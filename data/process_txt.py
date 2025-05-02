import numpy as np
import torch

# This script works with train.txt, val.txt, and test.txt files. I do not use it currently as I use binary files instead.
# This script contains two functions:
# - stream_tokens: A generator that reads a text file line by line and yields tokens when the buffer reaches a certain size.
# - get_inf_batch: A generator that reads a text file line by line and yields batches of tokens of size (batch_size, block_size).
# I wrote this script to avoid loading the entire dataset into memory at once, which is not feasible for large datasets.

def stream_tokens(filename, tokenizer, min_chunk_chars):
    """
    This function opens a text file, reads line by line. When it reaches the minimum chunk size,
    it encodes the text into tokens. In then yields the tokens (yield --> returs a generator
    that will return a buffer of size <= min_chunk_chars every time you loop through it).
    Args:
        filename: Path to the input text file
        tokenizer: Tokenizer object for encoding the text
        min_chunk_chars: Minimum number of characters to read from the file at once
    Yields:
        list: A list of tokens
    """
    with open(filename, "r", encoding="utf-8") as f:
        buffer = "" # Initialize buffer
        for line in f:
            buffer += line.strip() + " " # Add space to separate lines
            if len(buffer) >= min_chunk_chars:
                tokens = tokenizer.encode(buffer)
                yield tokens
                buffer = ""
        # Leftover
        if buffer:
            yield tokenizer.encode(buffer)


def get_batch(filename, tokenizer, min_chunk_chars, block_size, batch_size, device, random=False):
    """
    This functions opens a text file, reads line by line. When it reaches the minimum chunk size,
    it encodes the text into tokens. In then yields batches of tokens of size (batch_size, block_size).
    Args:
        filename: Path to the input text file
        tokenizer: Tokenizer object for encoding the text
        min_chunk_chars: Minimum number of characters to read from the file at once
        block_size: Length of each sequence
        batch_size: Number of sequences in the batch
        device: Device to move the tensors to (e.g., "cuda" or "cpu")
    Yields:
        tuple: A tuple containing the input and target tensors (x, y)
    """
    buffer = [] # Initialize buffer
    token_stream = stream_tokens(filename, tokenizer, min_chunk_chars) # Stream tokens from the file generator
    # here: add while True loop to keep the generator running indefinitely
    for tokens in token_stream:
        buffer.extend(tokens) # Instead of [[tokens], [tokens]] -> [tokens, tokens]
        while len(buffer) >= block_size * batch_size + 1: # Check if we have enough tokens for a batch
            if random:
                start_index = np.random.randint(0, len(buffer) - (block_size*batch_size+1))
            else:
                start_index = 0
            chunk = buffer[start_index:start_index + block_size * batch_size + 1] # Get the chunk of tokens + 1 for the next token
            buffer = buffer[block_size * batch_size:] # Remove the chunk from the buffer
            x = np.array(chunk[:-1]).reshape(batch_size, block_size) # Reshape the chunk to (batch_size, block_size)
            y = np.array(chunk[1:]).reshape(batch_size, block_size) # Reshape the chunk to (batch_size, block_size) but shifted by 1
            x = torch.tensor(x, dtype=torch.long).to(device) # Convert to tensor and move to device
            y = torch.tensor(y, dtype=torch.long).to(device) # Convert to tensor and move to device
            yield x, y
