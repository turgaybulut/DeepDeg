from typing import Any, List

import torch
import torch.nn as nn

class FeatureBranch(nn.Module):
    def __init__(self, config: Any, input_dim: int):
        super().__init__()
        self.config = config.model.features
        self.regularization = config.model.regularization

        dense_units = self.config.dense_units
        activation_fn = nn.LeakyReLU if self.config.activation == "leaky_relu" else nn.ReLU
        dropout_rate = self.regularization.dropout_rate

        layers: List[nn.Module] = []
        for units in dense_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        
        self.feature_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_mlp(x)
