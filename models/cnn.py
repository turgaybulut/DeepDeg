from typing import Any

import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config.model.cnn
        activation_fn = nn.LeakyReLU if self.config.activation == "leaky_relu" else nn.ReLU
        
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=self.config.filters, 
                kernel_size=self.config.kernel_size, 
                padding="same"
            ),
            activation_fn(),
            nn.MaxPool1d(kernel_size=self.config.pool_size),
            nn.Conv1d(
                in_channels=self.config.filters, 
                out_channels=self.config.filters, 
                kernel_size=self.config.kernel_size, 
                padding="same"
            ),
            activation_fn(),
            nn.MaxPool1d(kernel_size=self.config.pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.cnn_layers(x)
