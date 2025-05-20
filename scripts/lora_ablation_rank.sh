##############################################################
### Script to run the LoRA ablation study for Lora Rank search
###############################################################

# You should choose one of the following settings to run the LoRA ablation study for Lora Rank search.

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


# # rank search setting 2: lr=1e-5, batch_size=4, lora_rank=2, num_steps=10000
# python -m src.main \
#     --data_path data/lotka_volterra_train.h5 \
#     --train_ratio 0.8 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora_rank 2 \
#     --batch_size 4 \
#     --learning_rate 1e-5 \
#     --weight_decay 1e-4 \
#     --num_steps 2000 \
#     --checkpoint_interval 1000 \
#     --checkpoint_dir checkpoints/checkpoints_lora_rank_2 \
#     --seed 42 \
#     --max_ctx_length 512 \
#     --val_interval 1000 \
#     --use_wandb \
#     --wandb_project lora-finetune \
#     --wandb_entity wangxiaoye951-university-of-cambridge \
#     --wandb_name ablation_run_rank_2 \


# lr search setting 3: lr=1e-5, batch_size=4, lora_rank=8, num_steps=10000
python -m src.main \
    --data_path data/lotka_volterra_train.h5 \
    --train_ratio 0.8 \
    --precision 2 \
    --scale_percentile 0.95 \
    --lora_rank 8 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_steps 2000 \
    --checkpoint_interval 1000 \
    --checkpoint_dir checkpoints/checkpoints_lora_rank_8 \
    --seed 42 \
    --max_ctx_length 512 \
    --val_interval 1000 \
    --use_wandb \
    --wandb_project lora-finetune \
    --wandb_entity wangxiaoye951-university-of-cambridge \
    --wandb_name ablation_run_rank_8 \
