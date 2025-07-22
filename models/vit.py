from typing import Any

import torch
import torch.nn as nn

from models.modules import ViTEncoder

class ViTBranch(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config.model.vit
        self.patch_size = self.config.patch_size
        self.projection_dim = self.config.projection_dim

        self.patch_projection = nn.Linear(self.patch_size, self.projection_dim)

        max_seq_len = 4096
        num_patches = max_seq_len // self.patch_size
        self.positional_embedding = nn.Embedding(num_patches, self.projection_dim)

        self.encoder_layers = nn.ModuleList(
            [
                ViTEncoder(
                    num_heads=self.config.num_attention_heads,
                    key_dim=self.projection_dim,
                    mlp_units=self.config.mlp_units,
                    dropout_rate=self.config.dropout_rate,
                )
                for _ in range(self.config.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(2)

        _, seq_len = x.shape
        num_patches = seq_len // self.patch_size

        patches = x.unfold(1, self.patch_size, self.patch_size).contiguous()

        projected_patches = self.patch_projection(patches)

        positions = torch.arange(0, num_patches, device=x.device).unsqueeze(0)
        pos_embedding = self.positional_embedding(positions)
        vit_embeddings = projected_patches + pos_embedding

        for layer in self.encoder_layers:
            vit_embeddings = layer(vit_embeddings)

        output = torch.mean(vit_embeddings, dim=1)
        return output
