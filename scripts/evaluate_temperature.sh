# # # Best Model
# echo "Evaluating the best model for temperature sampling"
# python -m src.evaluate_temperature \
#     --data_path data/lotka_volterra_test.h5 \
#     --context_length 51 \
#     --target_length 20 \
#     --n_samples 10 \
#     --n_sampling 20 \
#     --precision 2 \
#     --scale_percentile 0.95 \
#     --lora \
#     --lora_rank 8 \
#     --resume checkpoints/checkpoints_best_model/checkpoint_step_3000.pt  \
#     --vis_save_path "output/checkpoints_best_model_rerun_3000" \


# original qwen model
echo "Evaluating the original qwen model for temperature sampling"
python -m src.evaluate_temperature \
    --data_path data/lotka_volterra_test.h5 \
    --context_length 51 \
    --target_length 20 \
    --n_samples 10 \
    --n_sampling 20 \
    --precision 2 \
    --scale_percentile 0.95 \
    --vis_save_path "output/original_qwen" \