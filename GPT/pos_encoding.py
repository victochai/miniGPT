import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self,
                 embed_size: int,
                 block_size: int,
                 ) -> None:
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(block_size).float().unsqueeze(1)
        i = torch.arange(embed_size).float().unsqueeze(0)
        angle = 1 / torch.pow(10000,(2*i)/embed_size)
        angle = pos * angle
        pe = torch.zeros(block_size, embed_size)
        assert pe.size() == angle.size(), "Positional encoding shape mismatch."
        pe[:, 0::2] = torch.sin(angle[:, 0::2]) # 0::2 means every second element starting from 0
        pe[:, 1::2] = torch.cos(angle[:, 1::2]) # 1::2 means every second element starting from 1

        self.register_buffer("pe", pe)
        # register_buffer because: 1. we don't want to train it, 2. we want to save it in the model state_dict, 3. we want to move it to model.device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        pe shape: [seq_len, embed_size]
        x shape: [batch_size, seq_len, embed_size]
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_size]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_size]
        """
        return x + self.pe.unsqueeze(0)
