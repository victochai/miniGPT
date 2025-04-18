from dataclasses import dataclass
import tiktoken
import torch
from GPT import GPT
from dataloader import get_batch, get_data
from config import GPTConfig
from tqdm import tqdm
import pickle
import os
import wandb as wb


def train(config):

    train_data, val_data = get_data(
        train_file_train=config.train_file_train,
        train_file_val=config.train_file_val,
    )

    model = GPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model device: {next(model.parameters()).device} \n\n")

    if config.checkpoint_path is not None:
        # Load the checkpoint if it exists
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

    for iter in tqdm(range(config.max_iters), desc="Training", unit="step"):

        X, Y = get_batch(
            train_data,
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=config.device
        )

        logits, loss = model(X, Y)  # Forward pass through the model
        optimizer.zero_grad(set_to_none=True)  # Zero the gradients
        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update the model parameters

        if iter % config.eval_interval == 0:

            losses = estimate_loss(model, train_data, val_data, config)
            train_loss = losses["train"]
            val_loss = losses["val"]
            if config.wandb:
                wb.log({"train_loss": train_loss, "val_loss": val_loss})
            print(f"Step {iter}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if iter > 0:

                state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": iter,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss
                    }

                if not os.path.exists(config.save_path):
                        os.makedirs(config.save_path)

                if config.save_every_checkpoint:
                    torch.save(
                        state,
                        os.path.join(config.save_path, "checkpoint_"+str(iter)+".pt")
                        )
                    print(f"💽 Saved latest checkpoint at step {iter}")

                # Save the best only if val loss improves

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        state,
                        os.path.join(config.save_path, "checkpoint_best.pt")
                        )
                    print(f"🌟 Saved new BEST checkpoint (val loss: {val_loss:.4f}, step {iter})")
                else:
                    print(f"❌ No improvement in val loss, not saving checkpoint")

        if iter == (config.max_iters-1):

            state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": iter,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss
                }

            print(f"🚀 Training complete! Final val loss: {val_loss:.4f}")
            torch.save(
                state,
                os.path.join(config.save_path, "checkpoint_final.pt")
                )
            print(f"✅ Final model checkpoint saved!")


@torch.no_grad()  # Disable gradient tracking for the following operations
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    data = {"train": train_data, "val": val_data}
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(
            data[split],
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=config.device
        )
            _, loss = model(X, Y)  # Forward pass through the model
            losses[k] = loss.item()  # Store the loss
        out[split] = losses.mean()  # Compute the mean loss for the split
    model.train()  # Set the model back to training mode
    return out  # Return the losses for both splits


if __name__ == "__main__":

    config = GPTConfig()
    train(config)
