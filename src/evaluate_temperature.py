import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
import logging

# Import the forecast evaluation functions
from .models.qwen import load_qwen
from .models.lora_linear import LoRALinear
from .preprocessor import load_data
from .forecast_evaluation_temperature import (
    run_sampling_evaluation, visualize_forecasts_with_uncertainty, print_metrics_table
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Set CUDA device if needed


def get_eval_args_parser(): 

    parser = argparse.ArgumentParser(description="Evaluation Script for Qwen2.5-Instruct Forecasting")
    # data parameters
    parser.add_argument('--data_path', default='./data/lotka_volterra_test.h5', type=str,
                        help='Path to the HDF5 file containing Lotka-Volterra time series data')

    # evaluation parameters
    parser.add_argument('--context_length', default=80, type=int,
                        help='Number of time steps to use as context for forecasting')
    parser.add_argument('--target_length', default=20, type=int,
                        help='Number of time steps to forecast into the future')
    parser.add_argument('--n_samples', default=5, type=int,
                        help='Number of trajectories to evaluate')
    parser.add_argument('--n_sampling', default=20, type=int,
                        help='Number of random samples to generate for each trajectory')
    parser.add_argument('--precision', default=2, type=int,
                        help='Decimal precision for formatting numbers')
    parser.add_argument('--scale_percentile', default=0.95, type=float,
                        help='Percentile used to compute the scaling factor for preprocessing')
    parser.add_argument('--lora', action='store_true',
                        help='Use LoRA for model evaluation')
    parser.add_argument('--vis_save_path', default=None, type=str,
                        help='Path to save the visualization results')
    parser.add_argument('--use_temperature', action='store_true',
                        help='Use temperature decoding for sampling, else use greedy decoding')
    
    
    # Model and LoRA hyperparameters
    parser.add_argument('--lora_rank', default=4, type=int,
                        help='LoRA rank (r)')
    parser.add_argument('--max_ctx_length', default=512, type=int,
                        help='Maximum context length for tokenization')

    # checkpoint parameters
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to a model checkpoint to load (if provided)')

    return parser

def set_seed(seed: int = 42): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def main(args):
    set_seed(42)

    # Load the data from the specified path
    logger.info("Loading data for evaluation...")
    trajectories, time_points = load_data(args.data_path)

    # load qwen model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_qwen()

    if args.lora:
        # Apply LoRA to each Transformer layer: wrap q_proj and v_proj.
        logger.info("Applying LoRA to the model...")
        for layer in model.model.layers:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=args.lora_rank)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=args.lora_rank)

    # Load the model checkpoint if provided.
    if args.resume is not None:
        logger.info(f"Loading model checkpoint from {args.resume}...")
        loc = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Checkpoint loaded.")

    model.eval()

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    logger.info("Running temperature evaluation, evey trajectory will be sampled {} times...".format(args.n_sampling))
    
    results = run_sampling_evaluation(
        model=model,
        tokenizer=tokenizer,
        trajectories=trajectories,
        context_length=args.context_length,
        target_length=args.target_length,
        n_samples=args.n_samples,
        n_sampling=args.n_sampling,
        precision=args.precision,
        scale_percentile=args.scale_percentile,
        max_ctx_length=args.max_ctx_length,
    )
    
    # results['mean_forecasts'] and results['std_forecasts'] are distribution statistics
    metrics = results["metrics"]
    print_metrics_table(metrics)
    
 
    context_sequences = []
    vis_context_length = args.context_length
    for idx in results['sample_indices']:
        context = trajectories[idx][:vis_context_length]
        context_sequences.append(context)

    visualize_forecasts_with_uncertainty(
        context_sequences=context_sequences,
        true_sequences=results["true_targets"],
        mean_forecasts=results["mean_forecasts"],
        std_forecasts=results["std_forecasts"],
        num_samples=5,
        save_path=args.vis_save_path,
    )

if __name__ == '__main__':
    parser = get_eval_args_parser()
    args = parser.parse_args()
    main(args)
