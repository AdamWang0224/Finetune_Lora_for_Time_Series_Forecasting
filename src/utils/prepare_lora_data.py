import torch

# Use sliding window tokenization to process sequences:
def process_sequences(texts, tokenizer, max_length=512, stride=256):
    """
    Processes list of texts into chunks of token IDs using a sliding window approach.
    
    Parameters:
        texts (list[str]): List of preprocessed text sequences.
        tokenizer: Tokenizer from Hugging Face.
        max_length (int): Maximum sequence length per chunk.
        stride (int): Stride for sliding window.
        
    Returns:
        Tensor: A tensor of shape (num_chunks, max_length) of token IDs.
    """
    all_input_ids = []
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat([chunk, torch.full((max_length - len(chunk),), tokenizer.pad_token_id)])
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)