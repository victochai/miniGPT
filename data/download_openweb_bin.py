import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# This script downloads 10 % of the OpenWebText dataset and splits it into train, validation, and test sets
# It saves the splits as binary files in the current directory
# It uses memory mapping for efficient handling of large datasets (it means that the data is not loaded into RAM all at once)
# It also tokenizes the text using the GPT-2 tokenizer and saves the tokenized data as binary files
# Info from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

# Number of workers in .map() call
# Good number to use is ~order number of cpu cores // 2
# I have 6 cores (on Windows: Task Manager --> Performance --> CPU to know yours)
num_proc = 3

# Number of workers in load_dataset() call
# Best number might be different from num_proc above as it also depends on NW speed.
# It is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':

    # Loads only 10 % of Hugging Face OpenWebText data
    dataset = load_dataset(
        "openwebtext",
        cache_dir="./openwebtext",
        split="train[:10%]", # Load the training split
        num_proc=num_proc_load_dataset
        )

    print(f"\n\nNumber of samples in the dataset: {len(dataset)}")

    # OpenWebText is built on Apache Arrow and has saveral build-in methods
    # type(dataset) --> datasets.arrow_dataset.Dataset
    # IT IS LIKE A DICT WITH KEYS --> ONE OF THEM IS TEXT
    # YOU CANNOT MODIFY IT AS A DICT, YOU NEED TO USE BUILD-IN METHODS FROM APACHE ARROW:
    # dataset.train_test_split() --> copies the data to create the splits.
    #                                doesn't modify original dataset in-place
    #                                If you are working with a very limited amount of RAM,
    #                                this can cause higher memory usage since it's duplicating the dataset for the splits
    # dataset.select() --> allows to select rows using specific indices, doesn't copy anything.
    #                      Just creates a reference.
    # dataset.map() --> Allows you to add a function to every examples in the dataset.
    # (the are also others, but these are the most important ones)

    def split_dataset(dataset, train_ratio=0.95, val_ratio=0.025, seed=42):
        dataset = dataset.shuffle(seed=seed)
        total = len(dataset)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        return {
            "train": dataset.select(range(train_end)),
            "val": dataset.select(range(train_end, val_end)),
            "test": dataset.select(range(val_end, total))
        }

    splits = split_dataset(dataset) # Now this is a read dict that contains Arrow Datasets with one key "text" (list of strings)

    # We now want to tokenize the dataset using dataset.map()
    # First we need to write the mapping funciton
    def tokenize(example):
        ids = enc.encode_ordinary(example["text"]) # encode_ordinary ignores any special tokens (eos, pad, etc.)
        # We do not encode many special tokens (only add EOS) to keep things simple at first
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the datasets
    tokenized = {}
    for split, dset in splits.items(): # We iterate through our real dict and create a new one with the tokenized datasets
        print(f"Tokenizing {split} dataset...")
        tokenized[split] = dset.map(
            tokenize,
            remove_columns=['text'],
            desc=f"tokenizing {split} split",
            num_proc=num_proc,
        )
        print(f"Tokenizing {split} dataset finished.")

    # Now we have a dict with 3 datasets (train, val, test) that contain the tokenized Arrow Datasets
    # with keys "ids" (list of tokens) and "len" (losgth of the list of tokens)

    # This block is performing a concatenation of tokenized data into a single large binary file
    # using memory mapping for efficient handling of large datasets.

    for split, dset in tokenized.items():
        print(f"Writing {split} dataset to binary file...")
        arr_len = np.sum(dset['len'], dtype=np.uint64) # Total number of tokens in the dataset
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (Can do since enc.max_token_value == 50256 is < 2**16)
        # Now, instead of creating an in-memory array for these tokenized IDs, we create a memory-mapped file
        # This allows you to treat the file like an array in RAM, but it’s actually stored on hard drive.
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_chunks = 1024
        idx = 0
        for batch_idx in tqdm(range(total_chunks), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_chunks, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids']) # Concatenate all the tokenized IDs in the batch (list of lists)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush() # Flush the memory-mapped array to disk (funny name lol)
        print(f"Writing {split} dataset to binary file finished.")
