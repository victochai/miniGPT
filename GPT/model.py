import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CausalAttention
from config import GPTConfig


class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:

        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embed_size)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,
                idx: torch.Tensor,
                targets: torch.Tensor = None,
                ) -> torch.Tensor:

        """
        Args:
            idx (torch.Tensor): Input tensor of shape [B, T]
            targets (torch.Tensor): Target tensor of shape [B, T]
        Returns:
            torch.Tensor: Output tensor of shape [B, T, vocab_size]
        """

        B, T = idx.size()
        tok_emb = self.token_embedding_table(idx) # [B, T, embed_size]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # [T, embed_size]
        x = self.dropout(tok_emb + pos_emb) # [B, T, embed_size] (broadcasting)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)    # [B, T, vocab_size]

        if targets is not None:
            logits = logits.view(B * T, -1) # [B * T, vocab_size]
            targets = targets.view(B * T) # [B * T]
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int = None,
                 ) -> torch.Tensor:

        """
        Args:
            idx (torch.Tensor): Input tensor of shape [B, T]
            max_new_tokens (int): Maximum number of new tokens to generate
        Returns:
            torch.Tensor: Output tensor of shape [B, T + max_new_tokens]
        """

        for _ in range(max_new_tokens):

            # Make sure idx is of shape [B, block_size (max sequence length)] or crop it
            idx_cond = idx[:, -self.config.block_size:]
            # Forward idx to the model to get the logits
            logits, _ = self(idx_cond) # [B, T, vocab_size]
            # We care only about the last time step to predict the next token
            # Temperature controls the randomness of the predictions
            # Lower temperature -> makes softmax sharper. The model is more confident about its predictions.
            # Higher temperature -> makes softmax flatter. The model is less confident,
            #                                  tokens have more equal probabilities.
            # Temerature = 1.0 means the model behaves normally, softmax-is
            logits = logits[:, -1, :] / temperature # [B, vocab_size]
            # TopK returns the k largest elements of the given input tensor along a given dimension
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution (not greedy sampling)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Block(nn.Module):

    def __init__(self, config: GPTConfig) -> None:

        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config.embed_size)
        self.attention = CausalAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size * 4),
            nn.ReLU(),
            nn.Linear(config.embed_size * 4, config.embed_size),
            nn.Dropout(config.ffn_dropout)
        )
        self.layer_norm2 = nn.LayerNorm(config.embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, embed_size]
                - B: Number of samples in the batch
                - T: Length of the input sequence (number of tokens)
                - embed_size: Size of the embedding vector for each token
        Returns:
            torch.Tensor: Output tensor of shape [B, T, embed_size]
        """

        x = x + self.attention(self.layer_norm1(x)) # [B, T, embed_size]
        x = x + self.feed_forward(self.layer_norm2(x)) # [B, T, embed_size]

        return x


if __name__ == "__main__":

    # Example usage
    config = GPTConfig()
    print(f"\n\nblock_size: {config.block_size}, embed_size: {config.embed_size}")
    print(f"vocab_size: {config.vocab_size}, n_layers: {config.n_layers}, n_heads: {config.n_heads}")
    print(f"device: {config.device}")

    model = GPT(config).to(config.device)

    idx = torch.randint(0, config.vocab_size, (2, config.block_size)).to(config.device)  # [2, block_size]
    logits, loss = model(idx)

    print("\n\nLogits shape:", logits.shape)  # [2, block_size, vocab_size]
    print("Loss:", loss)
