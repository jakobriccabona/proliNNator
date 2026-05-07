from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class ProlineSiteGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.head(h).squeeze(-1)
        return logits


def masked_bce_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    masked_probs = probs[mask]
    masked_labels = labels[mask].float()

    if masked_probs.numel() == 0:
        return probs.new_tensor(0.0, requires_grad=True)

    masked_probs = masked_probs.clamp(min=1e-7, max=1.0 - 1e-7)
    loss = F.binary_cross_entropy(masked_probs, masked_labels, reduction="none")

    if pos_weight is None:
        return loss.mean()

    pw = torch.tensor(pos_weight, device=probs.device)
    class_weight = torch.where(masked_labels > 0.5, pw, torch.ones_like(masked_labels))
    return (loss * class_weight).mean()


def masked_focal_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    masked_probs = probs[mask]
    masked_labels = labels[mask].float()

    if masked_probs.numel() == 0:
        return probs.new_tensor(0.0, requires_grad=True)

    masked_probs = masked_probs.clamp(min=1e-7, max=1.0 - 1e-7)
    bce = F.binary_cross_entropy(masked_probs, masked_labels, reduction="none")
    pt = masked_probs * masked_labels + (1.0 - masked_probs) * (1.0 - masked_labels)
    focal_factor = (1.0 - pt).pow(gamma)

    loss = focal_factor * bce

    if alpha is not None:
        alpha_t = alpha * masked_labels + (1.0 - alpha) * (1.0 - masked_labels)
        loss = alpha_t * loss

    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=probs.device)
        class_weight = torch.where(masked_labels > 0.5, pw, torch.ones_like(masked_labels))
        loss = class_weight * loss

    return loss.mean()
