from dataclasses import dataclass
import tiktoken
import torch
from GPT import GPT
from tqdm import tqdm
import os
import wandb as wb
import numpy as np


### -------------------------TRAINING CONFIG----------------------------- ###
@dataclass
class GPTConfig():
    ### model configuration
    tokenizer = tiktoken.get_encoding("gpt2") # Used for my current bin files
    vocab_size: int = tokenizer.n_vocab # Should be 50257
    block_size: int = 256 # T; The maximum length of the input sequence (number of tokens)
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
    train_iters = 100000 # Number of training iterations
    eval_interval = 500 # How often to evaluate the model on the validation set
    eval_iters = 200 # Number of iterations to evaluate the model on the validation set and the training set
    max_new_tokens = 50 # Number of new tokens to generate during evaluation
    eval_sentences = [
        "The meaning of life is",
        "In a recent scientific discovery",
        "Once upon a time, there was a",
        "The president announced today that",
        "According to the latest research",
        "The top 10 reasons why people love",
        "In a shocking turn of events",
        "Here is how you can improve your coding skills"
        ]
    learning_rate = 1e-4 # 1e-4 --> 0.0001
    weight_decay=1e-5
    batch_size = 8
    checkpoint_path = None # Path to the checkpoint file to load (if any)
    save_path = "./"
    final_model_name = "checkpoint_final.pt"
    best_model_name = "checkpoint_best_test.pt"
    data_dir = "./data/"
    save_every_checkpoint = False
    wandb = True # Whether to use wandb for logging
    wandb_project = "GPT-OpenWebText10percent-test"
    wandb_run_name = "GPT-OpenWebText10percent-test"
### --------------------------------------------------------------------- ###


def get_batch(split, data_dir, block_size, batch_size, device):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint((len(data)-1) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        x, y = x.to(device), y.to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def train(config):

    model = GPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model device: {next(model.parameters()).device}\n")

    start_iter = 0

    if config.checkpoint_path is not None:
        # Load the checkpkoint if it exists
        print(f"\nLoading checkpoint from {config.checkpoint_path}")
        if not os.path.exists(config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {config.checkpoint_path} not found.")
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        val_loss = checkpoint["val_loss"]
        start_iter = checkpoint["step"]
        print(f"✅ Resumed training from checkpoint (Step {start_iter})\n")

    best_val_loss = float("inf")

    if config.wandb:
        wb.init(project=config.wandb_project, name=config.wandb_run_name)
        wb.config.update(config)
        wb.watch(model, log="all", log_graph=True)
        # wb.log({"train_loss": 0, "val_loss": 0})

    if config.eval_sentences is not None:
        tokenized_sentences = []
        for sentence in config.eval_sentences:
            tokens = config.tokenizer.encode(sentence)
            tokenized_sentences.append(torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(config.device))
        print(f"\nThe model will generate text during eval time for the following sentences:")
        for i, sentence in enumerate(config.eval_sentences):
            print(f"  {i+1}: {sentence}")
        print(f"\n")

    for step in tqdm(range(start_iter, config.train_iters), desc="Training", unit="step", leave=False):

        X, Y = get_batch("train", config.data_dir, config.block_size, config.batch_size, config.device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0:

            losses, predictions = estimate_val_loss(model, config, tokenized_sentences=tokenized_sentences)

            if predictions is not None:
                table = wb.Table(columns=["index", "prediction"])
                for i, pred in enumerate(predictions):
                    table.add_data(i, pred)
                wb.log({"predictions": table}, step=step)

            train_loss = losses["train"]
            val_loss = losses["val"]
            if config.wandb:
                wb.log({"train_loss": train_loss, "val_loss": val_loss}, step=step)
            print(f"Step {step}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Predictions:")
            for i, sentence in enumerate(config.eval_sentences):
                print(f"  {i+1}: {predictions[i]}")

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
                        os.path.join(config.save_path, config.best_model_name)
                        )
                    print(f"🌟 Saved new BEST checkpoint (val loss: {val_loss:.4f}, step {step})")
                else:
                    print(f"❌ No improvement in val loss, not saving checkpoint")

        if step >= config.train_iters:
            break

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
        os.path.join(config.save_path, config.final_model_name)
        )
    print(f"✅ Final model checkpoint saved!")


@torch.no_grad()
def estimate_val_loss(model, config, tokenized_sentences=None):
    out = {}
    model.eval()
    losses = torch.zeros(config.eval_iters)
    split = ["train", "val"]
    for s in split:
        for i in range(config.eval_iters):
            X, Y = get_batch(s, config.data_dir, config.block_size, config.batch_size, config.device)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[s] = losses.mean()
    predictions = None
    if tokenized_sentences is not None:
        predictions = []
        for sentence in tokenized_sentences:
            generated = model.generate(sentence, max_new_tokens=config.max_new_tokens)
            decoded = config.tokenizer.decode(generated[0].tolist())
            predictions.append(decoded)
    model.train()
    return out, predictions


if __name__ == "__main__":

    config = GPTConfig()
    train(config)
