from typing import List

import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_units: List[int], dropout_rate: float = 0.1):
        super().__init__()
        layers = []
        original_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(hidden_units[-1], original_dim))
        layers.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class ViTEncoder(nn.Module):
    def __init__(self, num_heads: int, key_dim: int, mlp_units: List[int], dropout_rate: float = 0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(key_dim)
        self.attention = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(key_dim)
        self.mlp = MLPBlock(key_dim, mlp_units, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x
