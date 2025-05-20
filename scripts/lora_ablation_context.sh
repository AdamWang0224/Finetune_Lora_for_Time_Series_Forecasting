###################################################################
### Script to run the LoRA ablation study for context length search
###################################################################

# You should choose one of the following settings to run the LoRA ablation study for context length search.

# Use best setting: lr=1e-4, batch_size=4, lora_rank=8, num_steps=10000, max_ctx_length=512
# python -m src.main \
#     --data_path data/lotka_volterra_train.h5 \
#     --train_ratio 0.8 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora_rank 8 \
#     --batch_size 4 \
#     --learning_rate 1e-4 \
#     --weight_decay 1e-4 \
#     --num_steps 1000 \
#     --checkpoint_interval 500 \
#     --checkpoint_dir checkpoints/checkpoints_ctx_512\
#     --seed 42 \
#     --max_ctx_length 512 \
#     --val_interval 200 \
#     --use_wandb \
#     --wandb_project lora-finetune \
#     --wandb_entity wangxiaoye951-university-of-cambridge \
#     --wandb_name ablation_run_ctx_512 \

# # context length search setting 1: lr=1e-5, batch_size=4, lora_rank=8, num_steps=10000, max_ctx_length=128
# python -m src.main \
#     --data_path data/lotka_volterra_train.h5 \
#     --train_ratio 0.8 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora_rank 8 \
#     --batch_size 4 \
#     --learning_rate 1e-4 \
#     --num_steps 1000 \
#     --checkpoint_interval 500 \
#     --checkpoint_dir checkpoints/checkpoints_ctx_128 \
#     --seed 42 \
#     --max_ctx_length 128 \
#     --val_interval 200 \
#     --use_wandb \
#     --wandb_project lora-finetune \
#     --wandb_entity wangxiaoye951-university-of-cambridge \
#     --wandb_name ablation_run_ctx_128 \

# # context length search setting 2: lr=1e-5, batch_size=4, lora_rank=4, num_steps=10000, max_ctx_length=768
python -m src.main \
    --data_path data/lotka_volterra_train.h5 \
    --train_ratio 0.8 \
    --precision 2 \
    --scale_percentile 0.95 \
    --lora_rank 8 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_steps 1000 \
    --checkpoint_interval 500 \
    --checkpoint_dir checkpoints/checkpoints_ctx_768 \
    --seed 42 \
    --max_ctx_length 768 \
    --val_interval 200 \
    --use_wandb \
    --wandb_project lora-finetune \
    --wandb_entity wangxiaoye951-university-of-cambridge \
    --wandb_name ablation_run_ctx_768 \
