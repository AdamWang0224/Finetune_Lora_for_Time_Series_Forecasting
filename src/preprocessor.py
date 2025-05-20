"""
Module: preprocessor
This module implements the LLMTIME preprocessing scheme for time series data.
It includes functions for loading time series data from an HDF5 file,
rescaling and formatting the data into a textual representation, and tokenizing the text using a given tokenizer.

Author: Your Name
Date: YYYY-MM-DD
"""

import re
import h5py
import numpy as np
from typing import List, Tuple
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads time series data from an HDF5 file.

    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset within the HDF5 file to load (default "trajectories").

    Returns:
        np.ndarray: Loaded data as a NumPy array.
    """
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
    logger.info(f"Loaded data shape: {trajectories.shape}")
    return trajectories, time_points

def rescale_and_format(sequence: np.ndarray, precision: int = 2, scale_percentile: float = 0.95) -> str:
    """
    Rescales and formats a time series sequence into a string suitable for LLMTIME.
    
    The sequence is expected to be a 2D numpy array of shape (T, features).
    It performs the following:
        1. Rescaling: scales the sequence so that the chosen percentile value is 1.
        2. Fixed precision formatting: formats each number to the given precision.
        3. Concatenates values with commas between features and semicolons between time steps.

    Parameters:
        sequence (np.ndarray): 2D array of time series data (T, features).
        precision (int): Number of decimal places to keep.
        scale_percentile (float): The percentile used for scaling (e.g., 0.95 means 95th percentile becomes 1).

    Returns:
        str: A string representation of the sequence.
    """
    # Compute scaling factor based on the given percentile for all values in the sequence.
    scale = np.percentile(np.abs(sequence), scale_percentile * 100)
    if scale == 0:
        logger.warning("Scaling factor is zero in rescale_and_format(), using default scale of 1.0")
        scale = 1.0  # avoid division by zero
    scaled_seq = sequence / scale

    # Format each value with fixed precision and remove decimal point if not needed.
    # Convert each time step into a string with comma separated values.
    time_steps = []
    fmt_str = f"{{:.{precision}f}}"
    for t in range(scaled_seq.shape[0]):
        # Format each feature in the time step
        formatted_features = [fmt_str.format(val) for val in scaled_seq[t]]

        # Join features with a comma
        time_step_str = ",".join(formatted_features)
        time_steps.append(time_step_str)

    # Join time steps with a semicolon
    final_str = ";".join(time_steps)

    # add a semicolon at the end to indicate the end of the sequence
    final_str += ";"
    return final_str, scale

def tokenize_sequence(text: str, tokenizer, 
                      use_prompt: bool=False) -> List[int]:
    """
    Tokenizes the preprocessed text sequence using the provided tokenizer.

    Parameters:
        text (str): The preprocessed time series string.
        tokenizer: A tokenizer object that implements a __call__ method (e.g., from Hugging Face's transformers).
    
    Returns:
        List[int]: A list of token IDs representing the input text.
    """
    prompt = text

    system_prompt = (
    "You are a helpful assistant that performs time series predictions. "
    "The user will provide a sequence and you will predict the remaining sequence. "
    "Return only numeric values (floats) separated by commas and semicolons. "
    "Do not add extra text, explanation, or units.\n")

    if use_prompt:
        # Optionally, use a chat template for better context.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt = text
    # Tokenizer should return a dict with 'input_ids'
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    # Convert to a list of integers
    token_ids = encoding.input_ids[0].tolist()
    return token_ids


def decode_sequence_v1(formatted_text: str, scale: float, expected_features: int = 2, precision: int = 2) -> np.ndarray:
    """
    Decodes a formatted time series string back into a numerical array by reversing
    the LLMTIME encoding process.

    Parameters:
        formatted_text (str): The text representation produced by rescale_and_format.
        scale (float): The scaling factor that was used during preprocessing.
        expected_features (int): The expected number of features in each time step.
        precision (int): The precision used during formatting.

    Returns:
        np.ndarray: A 2D array of the decoded time series data.
    """
    # Split the string into time steps using semicolon.
    # Remove any trailing semicolon and split by semicolon.
    time_steps = formatted_text.strip(";").split(";")
    sequence_list = []

    
    for step in time_steps:
        feature_strs = [val for val in step.split(",") if val]
        if len(feature_strs) != expected_features:
            print(f"skipping step due to unexpected number of features: {step}")
            continue
        features = [float(val) for val in feature_strs]
        sequence_list.append(features)
        
    decoded_sequence = np.array(sequence_list) * scale
    return decoded_sequence

def decode_sequence(formatted_text: str, scale: float, expected_features: int = 2, precision: int = 2) -> np.ndarray:
    """
    Decodes a formatted time series string back into a numerical array by reversing
    the LLMTIME encoding process.
    
    This improved version uses a regular expression to extract valid time steps
    matching the expected numeric format (e.g., "0.45,0.76") and is more robust
    to extra characters or noise.
    
    Parameters:
        formatted_text (str): The text representation produced by rescale_and_format.
        scale (float): The scaling factor used during preprocessing.
        expected_features (int): The expected number of features in each time step.
        precision (int): The precision used during formatting.
        
    Returns:
        np.ndarray: A 2D array of the decoded time series data.
    """
    # Regular expression pattern: matches two numbers (with optional negative sign) separated by a comma.
    pattern = r'(-?\d+\.\d+,-?\d+\.\d+)'
    matches = re.findall(pattern, formatted_text)
    
    sequence_list = []
    for match in matches:
        # Split by comma to get individual feature values.
        features = match.split(',')
        if len(features) != expected_features:
            continue
        try:
            features = [float(x) for x in features]
            sequence_list.append(features)
        except ValueError:
            continue

    if len(sequence_list) == 0:
        logger.warning("No valid time steps extracted from the text.")
        return np.empty((0, expected_features))
    
    decoded_sequence = np.array(sequence_list) * scale
    return decoded_sequence



def load_and_preprocess(file_path: str, train_ratio: float = 0.8, precision: int = 2, scale_percentile: float = 0.95) -> Tuple[List[str], List[str]]:
    """
    Loads the Lotka-Volterra data from an HDF5 file, preprocesses each trajectory using the LLMTIME scheme,
    and splits the results into training and validation sets.

    Parameters:
        file_path (str): Path to the HDF5 file containing the data.
        train_ratio (float): Proportion of trajectories to use for training (default 0.8).
        precision (int): Number of decimal places for formatting the time series (default 2).
        scale_percentile (float): Percentile used for scaling (default 0.95).

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - train_texts: List of preprocessed text strings for training.
            - val_texts: List of preprocessed text strings for validation.
    """
    # Load the data (trajectories and time points). We only use trajectories for preprocessing.
    trajectories, _ = load_data(file_path)
    
    num_trajectories = trajectories.shape[0]
    indices = np.arange(num_trajectories)
    
    # Shuffle indices for random train/validation split.
    np.random.shuffle(indices)
    
    # Determine the number of training trajectories.
    num_train = int(train_ratio * num_trajectories)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_texts = []
    for idx in train_indices:
        # Process each trajectory into a text string using the LLMTIME scheme.
        text, _ = rescale_and_format(trajectories[idx], precision=precision, scale_percentile=scale_percentile)
        train_texts.append(text)
    
    val_texts = []
    for idx in val_indices:
        text, _ = rescale_and_format(trajectories[idx], precision=precision, scale_percentile=scale_percentile)
        val_texts.append(text)
    
    return train_texts, val_texts

