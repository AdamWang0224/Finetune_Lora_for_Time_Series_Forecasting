# Best model setting: lr=1e-4, batch_size=4, lora_rank=8, max_ctx_length=512
python -m src.main \
    --data_path data/lotka_volterra_train.h5 \
    --train_ratio 0.8 \
    --precision 2 \
    --scale_percentile 0.95 \
    --lora_rank 8 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_steps 3000 \
    --checkpoint_interval 500 \
    --checkpoint_dir checkpoints/checkpoints_best_model \
    --seed 42 \
    --max_ctx_lengt 512 \
    --val_interval 200 \
    --use_wandb \
    --wandb_project lora-finetune \
    --wandb_entity wangxiaoye951-university-of-cambridge \
    --wandb_name best_model \
    --resume checkpoints/checkpoints_ctx_512/checkpoint_step_1000.pt # We start from our ablation checkpoint, and run for 2000 more steps

