#!/usr/bin/env python3
"""
FunctionalCRSoftNet: CRSoft variant for functional covariates.

Adds a functional encoding layer that projects each functional covariate
X_j(s) from G grid points to q1 features via learned weight functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .crsoft import BaseCRSoftNet, ResidualBlock


class FunctionalCRSoftNet(BaseCRSoftNet):
    """Residual FFNN for competing risks with functional covariates.

    Adds a functional encoding layer (paper eq. 3) that projects each
    functional covariate X_j(s) from G grid points to q1 features via
    learned weight functions phi_{j,l'}(s).

    Architecture:
      X_j(s) -> Linear(G, q1) per covariate -> concat all + t -> backbone -> K logits

    Args:
        num_covariates: number of functional covariates (= p)
        n_grid: number of grid points per covariate (= G)
        embed_dim: embedding dimension per covariate (= q1)
        num_causes: number of competing event types (= K)
        hidden_dim: width of hidden layers
        num_blocks: number of residual blocks
    """

    def __init__(
        self,
        num_covariates: int,
        n_grid: int,
        embed_dim: int = 4,
        num_causes: int = 2,
        hidden_dim: int = 32,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.num_causes = num_causes
        self.num_covariates = num_covariates

        self.func_encoders = nn.ModuleList(
            [nn.Linear(n_grid, embed_dim) for _ in range(num_covariates)]
        )

        total_embed = num_covariates * embed_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(total_embed, hidden_dim),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, num_causes)

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> FunctionalCRSoftNet:
        """Reconstruct a trained FunctionalCRSoftNet from a checkpoint.

        Args:
            data: checkpoint dict with keys: state_dict, num_causes,
                num_covariates, n_grid, embed_dim, hidden_dim, num_blocks

        Returns:
            A FunctionalCRSoftNet instance ready for predict_cif_grid()
        """
        model = cls(
            num_covariates=data["num_covariates"],
            n_grid=data["n_grid"],
            embed_dim=data["embed_dim"],
            num_causes=data["num_causes"],
            hidden_dim=data["hidden_dim"],
            num_blocks=data["num_blocks"],
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model

    def forward(self, x_func: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)

        embeddings = []
        for j in range(self.num_covariates):
            h_j = self.func_encoders[j](x_func[:, j, :])
            embeddings.append(h_j)

        h = torch.cat(embeddings + [t], dim=1)
        h = self.input_proj(h)
        h = self.backbone(h)
        return self.output_layer(h)
