###################################
# Train with cpu or single gpu
###################################

# Basic settings: lr=1e-5, batch_size=4, lora_rank=4, num_steps=10000
python -m src.main \
    --data_path data/lotka_volterra_train.h5 \
    --train_ratio 0.8 \
    --precision 2 \
    --scale_percentile 0.95 \
    --lora_rank 4 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --num_steps 2000 \
    --checkpoint_interval 1000 \
    --checkpoint_dir checkpoints/checkpoints_basic_new2 \
    --seed 42 \
    --max_ctx_length 512 \
    --val_interval 100 \
    --use_wandb \
    --wandb_project lora-finetune \
    --wandb_entity wangxiaoye951-university-of-cambridge \
    --wandb_name basic_run_new2 \
    # --resume 
