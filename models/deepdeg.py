from typing import Any

import torch
import torch.nn as nn

from models.cnn import CNNBranch
from models.feature_branch import FeatureBranch
from models.vit import ViTBranch

class DeepDeg(nn.Module):
    def __init__(self, config: Any, feature_input_dim: int):
        super().__init__()
        self.config = config

        self.vit_branch = ViTBranch(config)
        self.cnn_branch = CNNBranch(config)
        self.feature_branch = FeatureBranch(config, feature_input_dim)

        final_config = self.config.model.final
        reg_config = self.config.model.regularization
        activation_fn = nn.LeakyReLU if final_config.activation == "leaky_relu" else nn.ReLU
        
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)

        vit_output_dim = self.config.model.vit.projection_dim
        cnn_output_dim = self.config.model.cnn.filters
        feature_output_dim = self.config.model.features.dense_units[-1]

        combined_dim = vit_output_dim + cnn_output_dim + feature_output_dim

        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, final_config.dense_units),
            activation_fn(),
            nn.Dropout(reg_config.dropout_rate),
            nn.Linear(final_config.dense_units, 1),
        )

    def forward(self, embedding_input: torch.Tensor, feature_input: torch.Tensor) -> torch.Tensor:
        vit_output = self.vit_branch(embedding_input)
        cnn_output_raw = self.cnn_branch(embedding_input)
        cnn_output_pooled = self.cnn_pool(cnn_output_raw).squeeze(-1)
        feature_output = self.feature_branch(feature_input)
        combined = torch.cat([vit_output, cnn_output_pooled, feature_output], dim=1)
        return self.final_mlp(combined)
