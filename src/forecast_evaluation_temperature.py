import os
import numpy as np
import torch
from tqdm import tqdm
import re
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from .preprocessor import rescale_and_format, tokenize_sequence


# ===============================
# 1) forecast_sequence_sampling
# ===============================

def forecast_sequence_sampling(
    model,
    tokenizer,
    context_sequence: np.ndarray,
    target_length: int,
    n_sampling: int = 20,
    precision: int = 2,
    scale_percentile: float = 0.95,
    max_new_tokens: int = 400,
    max_ctx_length: int = 512,
) -> np.ndarray:
    """
    Perform multiple random samples (n_sampling) for a given context sequence using LLMTIME scheme.
    Return an array of shape (n_sampling, target_length, features).

    Parameters:
        model, tokenizer: The Qwen2.5 model and tokenizer.
        context_sequence (np.ndarray): The context part of time series (T, features).
        target_length (int): Number of future time steps to predict.
        n_sampling (int): Number of random samples to generate.
        precision, scale_percentile: For LLMTIME encoding.
        max_new_tokens, max_ctx_length: Generation and tokenization constraints.

    Returns:
        np.ndarray of shape (n_sampling, target_length, features), each slice is one forecast sample.
    """
    # 1. Preprocess context
    preprocessed_context, scale = rescale_and_format(context_sequence,
                                                     precision=precision,
                                                     scale_percentile=scale_percentile)
    # 2. Tokenize context
    context_tokens = tokenize_sequence(preprocessed_context, tokenizer)
    if len(context_tokens) > max_ctx_length:
        context_tokens = context_tokens[:max_ctx_length]

    # 3. Convert to tensor
    input_ids = torch.tensor([context_tokens]).to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 4. For each sampling run, generate a forecast
    forecasts = []
    for _ in range(n_sampling):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,            
                pad_token_id=tokenizer.pad_token_id
            )

        # Extract newly generated tokens
        generated_tokens = output_ids[0][len(context_tokens):].tolist()
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Regex to extract valid time steps
        pattern = r'(-?\d+\.\d+,-?\d+\.\d+)'
        matches = re.findall(pattern, generated_text)

        # Keep only the first target_length matches
        matches = matches[:target_length]
        # Convert to float
        forecast_list = []
        for step_str in matches:
            try:
                features = [float(x) for x in step_str.split(',')]
                forecast_list.append(features)
            except ValueError:
                continue

        forecast_array = np.array(forecast_list) * scale
        # Pad or truncate
        if forecast_array.shape[0] < target_length:
            pad = np.zeros((target_length - forecast_array.shape[0], forecast_array.shape[1]))
            forecast_array = np.vstack([forecast_array, pad])
        elif forecast_array.shape[0] > target_length:
            forecast_array = forecast_array[:target_length]

        forecasts.append(forecast_array)

    return np.stack(forecasts, axis=0)  # shape (n_sampling, target_length, features)

# ===============================
# 2) run_sampling_evaluation
# ===============================

def run_sampling_evaluation(
    model,
    tokenizer,
    trajectories: np.ndarray,
    context_length: int,
    target_length: int,
    n_samples: int,
    n_sampling: int,
    precision: int = 2,
    scale_percentile: float = 0.95,
    max_ctx_length: int = 512,
):
    """
    For each selected sample in trajectories, do n_sampling times forecast, compute distribution statistics
    (mean, std), and compare with ground truth. Return metrics and some uncertainty measure.

    Returns:
        A dict with:
         - 'mean_forecasts': List of shape (n_samples, target_length, features)
         - 'std_forecasts':  List of shape (n_samples, target_length, features)
         - 'metrics': dictionary of overall MSE, MAE, R2, average uncertainty, etc.
         - 'all_samples': a structure storing all sampling results if needed
    """

    # 1. Subsample
    n_samples = min(n_samples, trajectories.shape[0])
    sample_indices = np.random.choice(trajectories.shape[0], n_samples, replace=False)

    # 2. For each sample, do repeated sampling
    mean_forecasts = []
    std_forecasts = []
    true_targets = []

    for idx in tqdm(sample_indices, desc="Sampling evaluation"):
        sequence = trajectories[idx]
        # Ensure we have enough length
        assert sequence.shape[0] >= context_length + target_length, "Sequence too short."

        context = sequence[:context_length]
        target = sequence[context_length:context_length + target_length]
        # shape: (target_length, features)

        # Repeated sampling
        forecast_samples = forecast_sequence_sampling(
            model, tokenizer, context,
            target_length=target_length,
            n_sampling=n_sampling,
            precision=precision,
            scale_percentile=scale_percentile,
            max_ctx_length=max_ctx_length,
        )  # shape: (n_sampling, target_length, features)

        # Compute mean, std
        mean_f = np.mean(forecast_samples, axis=0)  # (target_length, features)
        std_f = np.std(forecast_samples, axis=0)    # (target_length, features)

        mean_forecasts.append(mean_f)
        std_forecasts.append(std_f)
        true_targets.append(target)

    # 3. Flatten for overall metrics
    #   - We compare mean_forecasts with true_targets
    all_mean = np.vstack(mean_forecasts)
    all_std = np.vstack(std_forecasts)
    all_true = np.vstack(true_targets)

    mse = mean_squared_error(all_true, all_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_mean)
    r2 = r2_score(all_true, all_mean)

    # Average uncertainty: e.g. average standard deviation across all timesteps/features
    avg_uncertainty = np.mean(all_std)

    # Additional metrics
    # e.g. NRMSE or MAPE can be added here
    
    # Calculate normalized metrics
    # Normalized RMSE (NRMSE) as percentage of the range
    range_true = np.max(all_true) - np.min(all_true)
    nrmse = (rmse / range_true) * 100 if range_true > 0 else float('inf')

    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small epsilon or excluding zeros
    epsilon = 1e-10
    mape = np.mean(np.abs((all_true - all_mean) / (np.abs(all_true) + epsilon))) * 100

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "NRMSE": nrmse,
        "MAPE": mape,
        "AvgUncertainty": avg_uncertainty

    }

    return {
        "mean_forecasts": mean_forecasts,
        "std_forecasts": std_forecasts,
        "true_targets": true_targets,
        "metrics": metrics,
        "sample_indices": sample_indices,
        # Optionally store all sampling
        # "all_samples": all_samples
    }

# ===============================
# 3) visualize_forecasts_with_uncertainty
# ===============================

def visualize_forecasts_with_uncertainty(
    context_sequences: list,
    true_sequences: list,
    mean_forecasts: list,
    std_forecasts: list,
    features_names: list = None,
    num_samples: int = 5,
    save_path: str = None
):
    """
    Plot context + true + mean forecast, with a shaded area showing mean ± 1 std (or other bounds).
    
    context_sequences[i]: shape (context_length, features)
    true_sequences[i]:    shape (target_length, features)
    mean_forecasts[i]:    shape (target_length, features)
    std_forecasts[i]:     shape (target_length, features)
    """

    n_to_plot = min(num_samples, len(context_sequences))
    if features_names is None:
        n_features = context_sequences[0].shape[1]
        features_names = [f"Feature {i+1}" for i in range(n_features)]

    for i in range(n_to_plot):
        full_true = np.vstack([context_sequences[i], true_sequences[i]])
        full_mean = np.vstack([context_sequences[i], mean_forecasts[i]])
        full_std  = np.vstack([np.zeros_like(context_sequences[i]), std_forecasts[i]])
        
        time_idx = np.arange(full_true.shape[0])
        context_len = context_sequences[i].shape[0]

        fig, axes = plt.subplots(len(features_names), 1, figsize=(10, 3*len(features_names)), sharex=True)
        if len(features_names) == 1:
            axes = [axes]
        
        for f_idx, ax in enumerate(axes):
            # Plot context
            ax.plot(time_idx[:context_len], context_sequences[i][:, f_idx], 'b-', label='Context')
            # Plot true
            ax.plot(time_idx[context_len:], true_sequences[i][:, f_idx], 'g-', label='True')
            # Plot mean forecast
            ax.plot(time_idx[context_len:], mean_forecasts[i][:, f_idx], 'r--', label='Mean Forecast')

            # Plot uncertainty region: mean ± 1 std
            lower = full_mean[:, f_idx] - full_std[:, f_idx]
            upper = full_mean[:, f_idx] + full_std[:, f_idx]
            ax.fill_between(time_idx[context_len:], lower[context_len:], upper[context_len:], 
                            color='red', alpha=0.2, label='±1 Std Dev')

            # Vertical line
            ax.axvline(x=context_len-1, color='k', linestyle='--')
            ax.set_ylabel(features_names[f_idx])
            ax.legend(loc='best')
            # ax.legend(loc='upper left')

        fig.suptitle(f"Sample {i+1}: True vs Mean Forecast (+Uncertainty)")
        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"uncertainty_sample_{i+1}.png"), bbox_inches='tight')
        else:
            plt.show()

    plt.close('all')

def print_metrics_table(metrics: Dict[str, float]) -> None:
    """
    Prints a formatted table of evaluation metrics.
    
    Parameters:
        metrics (Dict[str, float]): Dictionary of evaluation metrics
    """
    print("\n===== Forecasting Performance Metrics =====")
    print(f"{'Metric':<15} {'Value':<15}")
    print("=" * 30)
    for metric, value in metrics.items():
        print(f"{metric:<15} {value:<15.6f}")


