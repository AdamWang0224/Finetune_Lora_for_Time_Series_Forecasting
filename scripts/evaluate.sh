###########################################
### script to evaluate the model
###########################################

# original qwen as benchmark
echo "Evaluating original qwen model settings"
python -m src.evaluate \
    --data_path data/lotka_volterra_test.h5 \
    --context_length 51 \
    --target_length 20 \
    --num_samples 100 \
    --precision 2 \
    --scale_percentile 0.95 \
    --vis_save_path "output/original_qwen" \


# # basic model
# echo "Evaluating basic model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 4 \
#     --resume checkpoints/checkpoints_basic/checkpoint_step_2000.pt  \
#     --vis_save_path "output/checkpoints_basic" \

# # lr 5e-5
# echo "Evaluating lr 5e-5 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 4 \
#     --resume checkpoints/checkpoints_lr_5/checkpoint_step_2000.pt  \
#     --vis_save_path "output/checkpoints_lr_5" \

# lr 1e-4
# echo "Evaluating lr 1e-4 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 4 \
#     --resume checkpoints/checkpoints_lr_10/checkpoint_step_2000.pt  \
#     --vis_save_path "output/checkpoints_lr_10" \

# rank 2
# echo "Evaluating lora rank 2 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 2 \
#     --resume checkpoints/checkpoints_lora_rank_2/checkpoint_step_2000.pt  \
#     --vis_save_path "output/checkpoints_lora_rank_2" \

# rank 8
# echo "Evaluating lora rank 8 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 8 \
#     --resume checkpoints/checkpoints_lora_rank_8/checkpoint_step_2000.pt  \
#     --vis_save_path "output/checkpoints_lora_rank_8" \

# # ctx 128
# echo "Evaluating context length 128 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 12 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 8 \
#     --resume checkpoints/checkpoints_ctx_128/checkpoint_step_1000.pt  \
#     --vis_save_path "output/checkpoints_ctx_128" \

# ctx 512
# echo "Evaluating context length 512 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 8 \
#     --resume checkpoints/checkpoints_ctx_512/checkpoint_step_1000.pt  \
#     --vis_save_path "output/checkpoints_ctx_512" \

# # ctx 768
# echo "Evaluating context length 768 model settings"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 76 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 8 \
#     --resume checkpoints/checkpoints_ctx_768/checkpoint_step_1000.pt  \
#     --vis_save_path "output/checkpoints_ctx_768" \

# # Best Model
# echo "Evaluating the best model!"
# python -m src.evaluate \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --num_samples 100 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 8 \
#     --resume checkpoints/checkpoints_best_model/checkpoint_step_3000.pt  \
#     --vis_save_path "output/checkpoints_best_model" \

# This is for evaluating every checkpoint in the checkpoints directory
#!/bin/bash
# for step in {1000..10000..1000}; do
#     checkpoint="checkpoints/checkpoints_basic/checkpoint_step_${step}.pt"
#     echo "Evaluating checkpoint at step ${step}: ${checkpoint}"
#     python -m src.evaluate \
#         --data_path data/lotka_volterra_test.h5 \
#         --context_length 51 \
#         --target_length 20 \
#         --num_samples 10 \
#         --precision 2 \
#         --scale_percentile 0.95 \
#         --lora \
#         --lora_rank 4 \
#         --resume "$checkpoint" \
#         --vis_save_path "output/checkpoints_basic/step_${step}"
# done
