#######################################################
### Script to run the LoRA ablation study for LR search
#######################################################

# You should choose one of the following settings to run the LoRA ablation study for LR search.

# # Basic setting: lr=1e-5, batch_size=4, lora_rank=4, num_steps=10000
# python -m src.main \
#     --data_path data/lotka_volterra_train.h5 \
#     --train_ratio 0.8 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora_rank 4 \
#     --batch_size 4 \
#     --learning_rate 1e-5 \
#     --weight_decay 1e-4 \
#     --num_steps 2000 \
#     --checkpoint_interval 1000 \
#     --checkpoint_dir checkpoints/checkpoints_basic \
#     --seed 42 \
#     --max_ctx_length 512 \
#     --val_interval 1000 \
#     --use_wandb \
#     --wandb_project lora-finetune \
#     --wandb_entity wangxiaoye951-university-of-cambridge \
#     --wandb_name basic_run \


# # lr search setting 2: lr=5e-5, batch_size=4, lora_rank=4, num_steps=10000
# python -m src.main \
#     --data_path data/lotka_volterra_train.h5 \
#     --train_ratio 0.8 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora_rank 4 \
#     --batch_size 4 \
#     --learning_rate 5e-5 \
#     --weight_decay 1e-4 \
#     --num_steps 2000 \
#     --checkpoint_interval 1000 \
#     --checkpoint_dir checkpoints/checkpoints_lr_5 \
#     --seed 42 \
#     --max_ctx_length 512 \
#     --val_interval 1000 \
#     --use_wandb \
#     --wandb_project lora-finetune \
#     --wandb_entity wangxiaoye951-university-of-cambridge \
#     --wandb_name ablation_run_lr_5e-5 \


# lr search setting 3: lr=1e-4, batch_size=4, lora_rank=4, num_steps=10000
python -m src.main \
    --data_path data/lotka_volterra_train.h5 \
    --train_ratio 0.8 \
    --precision 2 \
    --scale_percentile 0.95 \
    --lora_rank 4 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_steps 2000 \
    --checkpoint_interval 1000 \
    --checkpoint_dir checkpoints/checkpoints_lr_10 \
    --seed 42 \
    --max_ctx_length 512 \
    --val_interval 1000 \
    --use_wandb \
    --wandb_project lora-finetune \
    --wandb_entity wangxiaoye951-university-of-cambridge \
    --wandb_name ablation_run_lr_1e-4 \