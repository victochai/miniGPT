import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import BaseConfig


class CausalAttention(nn.Module):

    """
    A single head of multi-head causal attention.
    """

    def __init__(self, config: BaseConfig) -> None:

        super().__init__()

        assert config.embed_size % config.n_heads == 0, "Embedding size must be divisible by the number of heads."
        self.n_heads = config.n_heads
        self.block_size = config.block_size
        self.d_k = config.embed_size // config.n_heads
        self.res_dropout = nn.Dropout(config.res_dropout)
        self.att_dropout = nn.Dropout(config.att_dropout)
        self.W_qkv = nn.Linear(config.embed_size, 3 * config.embed_size, bias=False)
        self.W_proj = nn.Linear(config.embed_size, config.embed_size)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).to(config.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, embed_size]
        Returns:
            torch.Tensor: Output tensor of shape [B, T, embed_size]
        """

        B, T, embed_size = x.size()  # Get the shape of the input tensor
        # assert T <= self.block_size, f"Input tensor must have shape [B, T, embed_size], but got {T} > {self.block_size}."
        QKV = self.W_qkv(x)  # Compute the keys; [B, T, 3 * embed_size]
        Q, K, V = QKV.split(embed_size, dim=-1)  # [B, T, embed_size]

        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2) # [B, n_heads, T, d_k]
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2) # [B, n_heads, T, d_k]
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2) # [B, n_heads, T, d_k]

        attention = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k) # [B, n_heads, T, d_k] @ [B, n_heads, d_k, T] --> [B, n_heads, T, T]
        attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # [B, n_heads, T, T]
        attention = F.softmax(attention, dim=-1) # [B, n_heads, T, T]
        attention = self.att_dropout(attention)
        attention = attention @ V # [B, n_heads, T, T] @ [B, n_heads, T, d_k] --> [B, n_heads, T, d_k]
        attention = attention.contiguous().view(B, T, embed_size) # [B, T, embed_size]
        attention = self.res_dropout(self.W_proj(attention)) # [B, T, embed_size] @ [embed_size, embed_size] --> [B, T, embed_size]

        return attention
