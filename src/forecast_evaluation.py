import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Import necessary functions from your preprocessor module
from .preprocessor import rescale_and_format, tokenize_sequence, decode_sequence


def split_sequence(sequence: np.ndarray, context_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a time series sequence into context and target sections.
    
    Parameters:
        sequence (np.ndarray): 2D array of time series data (T, features)
        context_length (int): Number of time steps to use as context
        target_length (int): Number of time steps to predict
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Context and target sequences
    """
    assert context_length + target_length <= sequence.shape[0], "Context + target length exceeds sequence length"
    
    context = sequence[:context_length]
    target = sequence[context_length:context_length + target_length]
    
    return context, target


def forecast_sequence_v1(
        model, tokenizer, context_sequence: np.ndarray,
        target_length: int, precision: int = 2,
        scale_percentile: float = 0.95,
        max_new_tokens: int = 400,
        max_ctx_length: int = 512
        ) -> np.ndarray:
    """
    Uses the Qwen2.5 model to forecast future values given a context sequence.
    
    Parameters:
        model: The Qwen2.5 model with or without LoRA
        tokenizer: The tokenizer for Qwen2.5
        context_sequence (np.ndarray): The context part of the time series
        target_length (int): Number of time steps to predict
        precision (int): Decimal precision to use in formatting
        scale_percentile (float): Percentile for scaling
        max_new_tokens (int): Maximum number of new tokens to generate
        
    Returns:
        np.ndarray: The forecasted sequence
    """
    # Preprocess the context sequence using LLMTIME
    preprocessed_context, scale = rescale_and_format(context_sequence, 
                                                     precision=precision,
                                                     scale_percentile=scale_percentile)
    
    # Tokenize the context
    context_tokens = tokenize_sequence(preprocessed_context, tokenizer)

    if len(context_tokens) > max_ctx_length:
        context_tokens = context_tokens[:max_ctx_length]
    
    # Prepare input for model
    input_ids = torch.tensor([context_tokens]).to(model.device)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Generate future tokens using the model
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # add attention mask
            max_new_tokens=max_new_tokens,  # Set an upper limit to avoid excessive generation
            do_sample=False,  # Use greedy decoding for deterministic output
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Get only the newly generated tokens
    generated_tokens = output[0][len(context_tokens):].tolist()
    
    # Convert tokens back to text
    generated_text = tokenizer.decode(generated_tokens)
    
    # Parse the generated text to extract the forecasted values
    # We need to handle potential formatting issues in the generated text
    try:
        # First, try to directly decode the full string (context + generated)
        full_text = tokenizer.decode(output[0].tolist())
        
        # Extract just the generated part (after the context)
        context_text = tokenizer.decode(context_tokens)
        if full_text.startswith(context_text):
            forecast_text = full_text[len(context_text):]
        else:
            # Fallback if the above approach doesn't work
            forecast_text = generated_text
            
        # Count the expected semicolons for target_length time steps
        expected_semicolons = target_length - 1
        
        # Trim the forecast text to include the expected number of time steps
        semicolon_positions = [pos for pos, char in enumerate(forecast_text) if char == ';']
        if len(semicolon_positions) >= expected_semicolons:
            last_pos = semicolon_positions[expected_semicolons - 1] if expected_semicolons > 0 else len(forecast_text)
            trimmed_forecast = forecast_text[:last_pos + 1]  # Include the last semicolon
            
            # Ensure we have the complete last time step
            if expected_semicolons == 0:  # Only one time step
                # Find the first non-numeric, non-comma character
                for i, char in enumerate(trimmed_forecast):
                    if char not in "0123456789,.-":
                        trimmed_forecast = trimmed_forecast[:i]
                        break
            else:
                # Add characters until the next semicolon or non-numeric character
                i = last_pos + 1
                while i < len(forecast_text) and forecast_text[i] != ';' and forecast_text[i] in "0123456789,.-":
                    trimmed_forecast += forecast_text[i]
                    i += 1
                    
            forecast_text = trimmed_forecast
            
        # Combine context and forecast for proper decoding
        combined_text = preprocessed_context
        if not combined_text.endswith(';'):
            combined_text += ';'
        combined_text += forecast_text
            
        # Decode the combined text back to numeric values
        full_sequence = decode_sequence(combined_text, scale, precision)
        
        # Extract just the forecasted part
        forecast_sequence = full_sequence[context_sequence.shape[0]:context_sequence.shape[0] + target_length]
        
        # Ensure we have the expected number of time steps
        if forecast_sequence.shape[0] < target_length:
            # Pad with zeros or last value if needed
            padding = np.zeros((target_length - forecast_sequence.shape[0], forecast_sequence.shape[1]))
            forecast_sequence = np.vstack([forecast_sequence, padding])
        elif forecast_sequence.shape[0] > target_length:
            # Truncate if we generated too many steps
            forecast_sequence = forecast_sequence[:target_length]
            
        return forecast_sequence
        
    except Exception as e:
        print(f"Error in forecast parsing: {e}")
        # Return zeros as a fallback
        return np.zeros((target_length, context_sequence.shape[1]))

def forecast_sequence(
        model, 
        tokenizer, 
        context_sequence: np.ndarray,
        target_length: int, 
        precision: int = 2,
        scale_percentile: float = 0.95,
        max_new_tokens: int = 400,
        max_ctx_length: int = 512,
        use_temperature: bool = False
        ) -> np.ndarray:
    """
    Uses the Qwen2.5 model to forecast future values given a context sequence.
    
    The function preprocesses the context sequence using the LLMTIME scheme, generates new tokens,
    decodes the generated text, and uses a regular expression to extract predicted time steps.
    The extracted steps are then converted back into numeric values.
    
    Parameters:
        model: The Qwen2.5 model (with or without LoRA).
        tokenizer: The corresponding tokenizer.
        context_sequence (np.ndarray): The context part of the time series (shape: (T, features)).
        target_length (int): Number of time steps to predict.
        precision (int): Decimal precision for formatting.
        scale_percentile (float): Percentile for computing the scaling factor.
        max_new_tokens (int): Maximum number of new tokens to generate.
        max_ctx_length (int): Maximum context length for tokenization.
        
    Returns:
        np.ndarray: The forecasted sequence as a 2D array of shape (target_length, features).
    """
    # Preprocess the context sequence using LLMTIME: get formatted text and scaling factor.
    preprocessed_context, scale = rescale_and_format(context_sequence, precision=precision, scale_percentile=scale_percentile)
    
    # Tokenize the context text.
    context_tokens = tokenize_sequence(preprocessed_context, tokenizer)
    if len(context_tokens) > max_ctx_length:
        context_tokens = context_tokens[:max_ctx_length]
    
    # Prepare input tensor and attention mask.
    input_ids = torch.tensor([context_tokens]).to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Generate new tokens with greedy decoding.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=use_temperature,  # Use temperature sampling if specified, else greedy decoding
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract generated tokens (excluding context tokens).
    generated_tokens = output_ids[0][len(context_tokens):].tolist()
    # Decode generated tokens to text.
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Use regex to extract valid time steps from the generated text.
    pattern = r'(-?\d+\.\d+,-?\d+\.\d+)'
    matches = re.findall(pattern, generated_text)
    
    if len(matches) < target_length:
        print(f"Expected {target_length} time steps, but found only {len(matches)} in generated text.")
    
    # Take the first target_length matches.
    forecast_steps = matches[:target_length]
    
    forecast_list = []
    for step_str in forecast_steps:
        try:
            features = [float(x) for x in step_str.split(',')]
            forecast_list.append(features)
        except ValueError:
            continue
    
    forecast_sequence = np.array(forecast_list) * scale
    
    # If number of forecast steps is less than target_length, pad with zeros;
    # if more, truncate.
    if forecast_sequence.shape[0] < target_length:
        pad = np.zeros((target_length - forecast_sequence.shape[0], forecast_sequence.shape[1]))
        forecast_sequence = np.vstack([forecast_sequence, pad])
    elif forecast_sequence.shape[0] > target_length:
        forecast_sequence = forecast_sequence[:target_length]
    
    return forecast_sequence



def evaluate_forecasts(true_sequences: List[np.ndarray], 
                       forecast_sequences: List[np.ndarray]
                       ) -> Dict[str, float]:
    """
    Evaluates forecasting performance using appropriate metrics.
    
    Parameters:
        true_sequences (List[np.ndarray]): List of true target sequences
        forecast_sequences (List[np.ndarray]): List of forecasted sequences
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Flatten all sequences for overall evaluation
    all_true = np.vstack(true_sequences)
    all_forecast = np.vstack(forecast_sequences)
    
    # Calculate metrics
    mse = mean_squared_error(all_true, all_forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_forecast)
    
    # Calculate R² (coefficient of determination)
    # Note: R² can be negative if the model predictions are worse than a simple mean predictor
    r2 = r2_score(all_true, all_forecast)
    
    # Calculate normalized metrics
    # Normalized RMSE (NRMSE) as percentage of the range
    range_true = np.max(all_true) - np.min(all_true)
    nrmse = (rmse / range_true) * 100 if range_true > 0 else float('inf')
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small epsilon or excluding zeros
    epsilon = 1e-10
    mape = np.mean(np.abs((all_true - all_forecast) / (np.abs(all_true) + epsilon))) * 100
    
    # Return all metrics in a dictionary
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "NRMSE (%)": nrmse,
        "MAPE (%)": mape
    }
    
    return metrics


def visualize_forecasts(context_sequences: List[np.ndarray],
                        true_sequences: List[np.ndarray],
                        forecast_sequences: List[np.ndarray],
                        num_samples: int = 5, 
                        features_names: List[str] = None,
                        save_path: str = None
                        ) -> None:
    """
    Visualizes the true sequences versus the forecasted sequences.
    
    Parameters:
        context_sequences (List[np.ndarray]): List of context sequences
        true_sequences (List[np.ndarray]): List of true target sequences
        forecast_sequences (List[np.ndarray]): List of forecasted sequences
        num_samples (int): Number of sample sequences to plot
        features_names (List[str]): Names of the features for plotting
        save_path (str): Path to save the plots (if provided)
    """
    # Set default feature names if not provided
    if features_names is None:
        n_features = context_sequences[0].shape[1]
        features_names = [f"Feature {i+1}" for i in range(n_features)]
    
    # Plot at most num_samples sequences
    n_to_plot = min(num_samples, len(context_sequences))
    
    for i in range(n_to_plot):
        # Ensure we have full sequences for plotting (context + true)
        full_true = np.vstack([context_sequences[i], true_sequences[i]])
        full_forecast = np.vstack([context_sequences[i], forecast_sequences[i]])
        
        # Create time index for x-axis
        time_idx = np.arange(full_true.shape[0])
        context_len = context_sequences[i].shape[0]
        
        # Create a figure with subplots for each feature
        fig, axes = plt.subplots(len(features_names), 1, figsize=(12, 3*len(features_names)), sharex=True)
        if len(features_names) == 1:
            axes = [axes]  # Ensure axes is a list for consistent indexing
        
        for j, (ax, feature_name) in enumerate(zip(axes, features_names)):
            # Plot context (shared between true and forecast)
            ax.plot(time_idx[:context_len], context_sequences[i][:, j], 'b-', label='Context')
            
            # Plot true future values
            ax.plot(time_idx[context_len:], true_sequences[i][:, j], 'g-', label='True Future')
            
            # Plot forecasted values
            ax.plot(time_idx[context_len:], forecast_sequences[i][:, j], 'r--', label='Forecast')
            
            # Add vertical line to mark the split between context and forecast
            ax.axvline(x=context_len-1, color='k', linestyle='--')
            
            # Add labels and legend
            ax.set_ylabel(feature_name)

            # if feature_name == "Prey":
            #     ax.legend(loc='upper center')
            # else:
            ax.legend(loc='best')
            
            # Calculate metrics for this feature in this sample
            mse = mean_squared_error(true_sequences[i][:, j], forecast_sequences[i][:, j])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_sequences[i][:, j], forecast_sequences[i][:, j])
            
            # Add metrics as text
            ax.text(0.02, 0.95, f'RMSE: {rmse:.4f}, MAE: {mae:.4f}', 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        # Add overall title and x-label
        fig.suptitle(f'Sample {i+1}: True vs Forecasted Sequences', fontsize=16)
        axes[-1].set_xlabel('Time Steps')
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle

        
        # Save figure to file
        if save_path:
            fig.savefig(f"{save_path}/forecast_sample_{i+1}.png", bbox_inches='tight')
        else:
            # Optionally show the plot
            plt.show()



def run_evaluation(model, tokenizer, trajectories: np.ndarray,
                   context_length: int = 80,
                   target_length: int = 20,
                   num_samples: int = 100,
                   precision: int = 2,
                   scale_percentile: float = 0.95,
                   max_ctx_length: int = 512,
                   ) -> Dict[str, Any]:
    """
    Runs a complete evaluation of the model's forecasting ability.

    Parameters:
        model: The untrained Qwen2.5 model
        tokenizer: The tokenizer for Qwen2.5
        trajectories (np.ndarray): Array of time series trajectories
        context_length (int): Number of time steps to use as context
        target_length (int): Number of time steps to predict
        num_samples (int): Number of samples to evaluate
        precision (int): Decimal precision to use in formatting
        scale_percentile (float): Percentile for scaling

    Returns:
        Dict[str, Any]: Results dictionary containing metrics and sample data
    """
    # Limit the number of samples to evaluate
    num_samples = min(num_samples, trajectories.shape[0])
    sample_indices = np.random.choice(trajectories.shape[0], num_samples, replace=False)

    max_new_tokens = (target_length + 10) * 10

    # Lists to store context, true and forecasted sequences
    context_sequences = []
    true_sequences = []
    forecast_sequences = []

    # Process each sample
    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        # Get the sequence for this sample
        sequence = trajectories[idx]

        # Split into context and target
        context, target = split_sequence(sequence, context_length, target_length)

        # Generate forecast
        forecast = forecast_sequence(model, tokenizer, context, 
                                     target_length, precision, scale_percentile,
                                     max_new_tokens=max_new_tokens,
                                     max_ctx_length=max_ctx_length)

        # Store results
        context_sequences.append(context)
        true_sequences.append(target)
        forecast_sequences.append(forecast)

    # Calculate overall metrics
    metrics = evaluate_forecasts(true_sequences, forecast_sequences)

    # Create results dictionary
    results = {
        "metrics": metrics,
        "context_sequences": context_sequences,
        "true_sequences": true_sequences,
        "forecast_sequences": forecast_sequences,
        "sample_indices": sample_indices
    }
    
    return results


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
