import torch
import tiktoken
from dataclasses import dataclass


@dataclass
class GPTConfig():
    ### model configuration
    tokenizer = tiktoken.get_encoding("r50k_base")
    vocab_size: int = tokenizer.n_vocab # Should be 50257
    block_size: int = 16 # T; The maximum length of the input sequence (number of tokens)
    n_layers: int = 8
    n_heads: int = 8
    embed_size: int = 256
    dropout: float = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn_dropout: float = 0.1
    att_dropout: float = 0.1
    res_dropout: float = 0.1
    assert embed_size % n_heads == 0, "Embedding size must be divisible by the number of heads."
    ### training hyperparameters
    max_iters = 15000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 1e-2 # 1e-3 --> 0.001
    weight_decay=1e-2
    batch_size = 8
    checkpoint_path = None # Path to the checkpoint file to load (if any)
    save_path = "./"
    train_file_train = "./dataloader/train_data.pkl"
    train_file_val = "./dataloader/val_data.pkl"
    save_every_checkpoint = False
    wandb = True # Whether to use wandb for logging
    wandb_project = "GPT-TinySheakespeare"
    wandb_run_name = "GPT-TinySheakespeare-1"

