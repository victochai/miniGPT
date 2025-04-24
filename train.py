from dataclasses import dataclass
import tiktoken
import torch
from GPT import GPT
from tqdm import tqdm
import os
import wandb as wb
from torch.utils.data import DataLoader
from data.dataset import GPTBinDataset


### -------------------------TRAINING CONFIG----------------------------- ###
@dataclass
class GPTConfig():
    ### model configuration
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size: int = tokenizer.n_vocab # Should be 50257
    block_size: int = 32 # T; The maximum length of the input sequence (number of tokens)
    n_layers: int = 8
    n_heads: int = 8 # Should be config.embed_size % config.n_heads == 0
    embed_size: int = 256
    dropout: float = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn_dropout: float = 0.1
    att_dropout: float = 0.1
    res_dropout: float = 0.1
    assert embed_size % n_heads == 0, "Embedding size must be divisible by the number of heads."
    ### training hyperparameters
    train_iters = 1000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 1e-4 # 1e-3 --> 0.001
    weight_decay=1e-5
    batch_size = 16
    checkpoint_path = None # Path to the checkpoint file to load (if any)
    save_path = "./"
    data_dir = "./data/"
    save_every_checkpoint = False
    min_chunk_chars = 5000 # Minimum number of characters to read from the file at once
    wandb = True # Whether to use wandb for logging
    wandb_project = "GPT-OpenWebText10percent"
    wandb_run_name = "GPT-OpenWebText10percent-1"
### --------------------------------------------------------------------- ###


def train(config):

    model = GPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"\n\nModel parameters: {sum(p.numel() for p in model.parameters())} \n\n")
    print(f"Model device: {next(model.parameters()).device} \n\n")

    # Load the dataset
    train_dataset = GPTBinDataset(os.path.join(config.data_dir, "train.bin"), config.block_size, config.device)
    val_dataset = GPTBinDataset(os.path.join(config.data_dir, "val.bin"), config.block_size, config.device)
    print(f"Train dataset length: {len(train_dataset)}\n\n")
    print(f"Val dataset length: {len(val_dataset)}\n\n")

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    start_iter = 0

    if config.checkpoint_path is not None:
        # Load the checkpkoint if it exists
        print(f"Loading checkpoint from {config.checkpoint_path}")
        if not os.path.exists(config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {config.checkpoint_path} not found.")
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        val_loss = checkpoint["val_loss"]
        start_iter = checkpoint["step"] + 1
        print(f"✅ Resumed training from checkpoint (Step {start_iter})")

    best_val_loss = float("inf")

    if config.wandb:
        wb.init(project=config.wandb_project, name=config.wandb_run_name)
        wb.config.update(config)
        wb.watch(model, log="all", log_graph=True)
        wb.log({"train_loss": 0, "val_loss": 0})

    for step in tqdm(range(start_iter, config.train_iters), desc="Training", unit="step", leave=False):

        X, Y = next(iter(train_dataloader))
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0:

            losses = estimate_val_loss(model, config, val_dataloader)
            train_loss = losses["train"]
            val_loss = losses["val"]
            if config.wandb:
                wb.log({"train_loss": train_loss, "val_loss": val_loss}, step=step)
            print(f"Step {step}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if step > 0:

                state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss
                    }

                if not os.path.exists(config.save_path):
                        os.makedirs(config.save_path)

                if config.save_every_checkpoint:
                    torch.save(
                        state,
                        os.path.join(config.save_path, "checkpoint_"+str(step)+".pt")
                        )
                    print(f"💽 Saved latest checkpoint at step {step}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        state,
                        os.path.join(config.save_path, "checkpoint_best1.pt")
                        )
                    print(f"🌟 Saved new BEST checkpoint (val loss: {val_loss:.4f}, step {step})")
                else:
                    print(f"❌ No improvement in val loss, not saving checkpoint")

    state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss
        }

    print(f"🚀 Training complete! Final val loss: {val_loss:.4f}")
    torch.save(
        state,
        os.path.join(config.save_path, "checkpoint_final.pt")
        )
    print(f"✅ Final model checkpoint saved!")


@torch.no_grad()
def estimate_val_loss(model, config, val_dataloader, train_dataloader):
    out = {}
    model.eval()
    losses = torch.zeros(config.eval_iters)
    split = ["train", "val"]
    for s in split:
        if s == "train":
            dataloader = train_dataloader
        else:
            dataloader = val_dataloader
        for i in range(config.eval_iters):
            X, Y = next(iter(dataloader))
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[s] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":

    config = GPTConfig()
    train(config)
