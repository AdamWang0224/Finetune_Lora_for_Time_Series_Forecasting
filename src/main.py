import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
import wandb  # Import wandb for logging
import logging

# Import preprocessor functions and model loader
from .preprocessor import  load_and_preprocess
from .models.qwen import load_qwen
from .models.lora_linear import LoRALinear
from .utils.prepare_lora_data import process_sequences
# from .forecast_evaluation import (
#     split_sequence, forecast_sequence, evaluate_forecasts, 
#     visualize_forecasts, run_evaluation, print_metrics_table
# )


# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set CUDA device if needed

# -------------------------------
# Argument parser
# -------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Qwen2.5-Instruct on Time Series Forecasting")
    
    # Data and preprocessing
    parser.add_argument('--data_path', default='data/lotka_volterra_data.h5', type=str,
                        help='Path to the HDF5 file containing Lotka-Volterra time series data')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='Proportion of trajectories used for training (rest for validation)')
    parser.add_argument('--precision', default=2, type=int,
                        help='Decimal precision used for formatting numbers')
    parser.add_argument('--scale_percentile', default=0.95, type=float,
                        help='Percentile used to compute the scaling factor')
    
    # Model and LoRA hyperparameters
    parser.add_argument('--lora_rank', default=4, type=int,
                        help='LoRA rank (r)')
    parser.add_argument('--max_ctx_length', default=512, type=int,
                        help='Maximum context length for tokenization')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Training batch size')
    parser.add_argument('--learning_rate', default=1e-5, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', default=None, type=float,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_steps', default=10000, type=int,
                        help='Total number of training steps')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint_interval', default=1000, type=int,
                        help='Interval for saving checkpoints')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')


    # Resume training from a checkpoint
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to resume checkpoint')
    parser.add_argument('--no_resume_optimizer', action='store_true',
                        help='Do not resume optimizer state if resuming')

    # Validation frequency
    parser.add_argument('--val_interval', default=1000, type=int,
                        help='Interval (in steps) to run validation')
    
    # Wandb logging parameters.
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb_project', default='lora-finetune', type=str,
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', default='your entity', type=str,
                        help='Wandb entity name')
    parser.add_argument('--wandb_name', default='Xiaoye_Wang', type=str,
                        help='Wandb run name')
    parser.add_argument('--wandb_mode', default='online', type=str,
                        help='Wandb mode (online/offline)')
    parser.add_argument('--wandb_id', default=None, type=str,
                        help='Wandb run ID (if resuming a run)')

    # # Evaluation parameters
    # parser.add_argument('--eval', action='store_true',
    #                     help='Run evaluation instead of training')
    # parser.add_argument('--context_length', default=80, type=int,
    #                     help='Number of time steps to use as context for forecasting')
    # parser.add_argument('--target_length', default=20, type=int,
    #                     help='Number of time steps to forecast into the future')
    # parser.add_argument('--num_samples', default=100, type=int,
    #                     help='Number of trajectories to evaluate')

    return parser

# -------------------------------
# Set random seed for reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# -------------------------------
# Checkpoint save function
# -------------------------------
def save_checkpoint(model, optimizer, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    logger.info(f"Checkpoint saved at step {step} to {ckpt_path}")

# -------------------------------
# Checkpoint load function
# -------------------------------
def load_checkpoint(model, optimizer, resume_path, no_resume_optimizer=False):
    logger.info(f"Loading checkpoint from {resume_path}")

    loc = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(resume_path, map_location=loc)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    step = checkpoint.get('step', 0)

    if not no_resume_optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return step

# -------------------------------
# Validation function (dummy example)
# -------------------------------
def validate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for (batch,) in tqdm(val_loader, desc="Validation"):
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
            n_batches += 1
    model.train()
    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    return avg_loss



# -------------------------------
# Main training function
# -------------------------------
def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoint directory if not exists.
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize wandb (only on main process)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, 
                   name=args.wandb_name,
                   config=vars(args))
    
    # -------------------------------
    # Data Loading and Preprocessing
    # -------------------------------
    logger.info("Loading and preprocessing data...")
    # load_and_preprocess returns two lists of strings: train_texts and val_texts.
    # train_texts and val_texts are 720 and 180 sequences respectively, with each sequence having a length of 1000
    # this will be decoded into 1000 tokens.
    train_texts, val_texts = load_and_preprocess(args.data_path, train_ratio=args.train_ratio,
                                                  precision=args.precision,
                                                  scale_percentile=args.scale_percentile)
    # For tokenization we use sliding window if needed.
    max_ctx_length = args.max_ctx_length  

    # -------------------------------
    # Load Model and Apply LoRA
    # -------------------------------
    model, tokenizer = load_qwen()

    train_input_ids = process_sequences(train_texts, tokenizer, max_length=max_ctx_length, stride=max_ctx_length//2)
    val_input_ids = process_sequences(val_texts, tokenizer, max_length=max_ctx_length, stride=max_ctx_length)
    
    # -------------------------------
    # Apply LoRA
    # -------------------------------
    logger.info("Successfully loaded the model and applying LoRA now...")

    # Apply LoRA to each Transformer layer: wrap q_proj and v_proj.
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=args.lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=args.lora_rank)
    # At this point, only the LoRA matrices in q_proj and v_proj are trainable.

    # -------------------------------
    # Model Summary
    # -------------------------------
    logger.info("Model summary:")
    logger.info(model)
    # print the number of layers in the model.
    num_layers = len(model.model.layers)
    logger.info(f"Number of layers in the model: {num_layers}")
    # print the number of parameters in the model.
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters in the model: {num_params}")
    # print the number of trainable parameters in the model.
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters in the model: {num_trainable_params}")

    
    # -------------------------------
    # Prepare DataLoader and Optimizer
    # -------------------------------
    train_dataset = TensorDataset(train_input_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TensorDataset(val_input_ids)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Only trainable parameters (LoRA parameters) are optimized.
    if args.weight_decay is None:
        # Then we do not apply weight decay to optimizer.
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.learning_rate)
    else:
        # Apply weight decay to all parameters.
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), 
                                    lr=args.learning_rate, weight_decay=args.weight_decay)
        
    # Use Accelerator for device handling and (potentially) distributed training.
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    val_loader = accelerator.prepare(val_loader)

    # Resume checkpoint if provided.
    start_step = 0
    
    if args.resume is not None:
        start_step = load_checkpoint(model, optimizer, args.resume, args.no_resume_optimizer)
        logger.info(f"Successfully load the model from {args.resume} resumed from checkpoint at step {start_step}")
    
    # -------------------------------
    # Initial Validation (baseline)
    # -------------------------------
    init_val_loss = validate(model, val_loader, accelerator)
    logger.info(f"Initial Validation Loss: {init_val_loss:.4f}")
    if args.use_wandb:
        wandb.log({"val_loss": init_val_loss, "step": start_step})

    
    # -------------------------------
    # Training Loop
    # -------------------------------
    model.train()
    steps = start_step
    logger.info("Start training...")
    while steps < args.num_steps:
        progress_bar = tqdm(train_loader, desc=f"Training Step {steps}", leave=False)
        for (batch,) in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)  # forward pass, loss computed internally (e.g., cross-entropy)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            
            steps += 1
            progress_bar.set_postfix(loss=loss.item())

            # Log metrics to wandb every 100 steps
            if args.use_wandb:
                wandb.log({"train_loss": loss.item(), "step": steps})
            
            # Save checkpoint every 1000 steps
            if steps % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{steps}.pt")
                logger.info(f"Saving checkpoint at step {steps} to {ckpt_path}")
                torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)

            # Run validation every val_interval steps.
            if steps % args.val_interval == 0:
                val_loss = validate(model, val_loader, accelerator)
                logger.info(f"Validation Loss at step {steps}: {val_loss:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "step": steps})
            
            if steps >= args.num_steps:
                break
    logger.info("Training completed.")
    model.eval()
    
    # -------------------------------
    # Evaluate on Validation Set
    # -------------------------------
    logger.info("Evaluating on validation set...")
    
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Validation {steps}",)
        for (batch,) in progress_bar_val:
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    logger.info(f"Final Validation Loss: {avg_loss:.4f}")

    # Log validation loss to wandb
    if args.use_wandb:
        wandb.log({"val_loss": avg_loss, "step": steps})

    # Save final model checkpoint
    final_ckpt_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    logger.info(f"Saving final model checkpoint to {final_ckpt_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_ckpt_path)

    # Finish wandb run
    if args.use_wandb:
        wandb.log({"final_val_loss": avg_loss})
        # wandb.save(final_ckpt_path)
        wandb.run.finish()
        wandb.finish()




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
