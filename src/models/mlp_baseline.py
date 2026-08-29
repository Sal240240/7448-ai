"""Deep MLP baseline (spec section 5.1): tapering feedforward net with
batchnorm + dropout, predicting the full metabolite vector jointly.

Unlike the per-metabolite elastic net baseline, this is a single multi-output
model -- the point of comparison is exactly "does sharing a representation
across correlated metabolites help," which is the spec's stated gap versus
existing linear/RF approaches (section 2).
"""
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, n_features: int, n_targets: int, hidden_dims=(3000, 2000, 1000, 500), dropout: float = 0.3):
        super().__init__()
        dims = [n_features, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE computed only over measured (mask=True) entries -- see standardize_metabolites.py
    for why unmeasured entries can't just be treated as zero."""
    sq_err = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return sq_err.sum() / denom
