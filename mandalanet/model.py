"""MandalaNet: geometric GNN + chaos pyramid, with a reconstruction and a
classification head. One class, two tasks, so the ablation code can swap the
mesh topology underneath without touching the model.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .graph import EdgeGeometry, Graph
from .pyramid import ChaosPyramid


class MandalaNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 16, n_classes: int = 3,
                 dim: int = 64, depth: int = 3, blocks: int = 2, heads: int = 4,
                 harmonics: int = 4, rbf_bins: int = 8, edge_types: int = 3,
                 dropout: float = 0.05, cell: float = 0.125,
                 use_edge_gate: bool = True, mlp_ratio: float = 2.0):
        super().__init__()
        self.geom = EdgeGeometry(rbf_bins, harmonics, edge_types)
        self.inp = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.drop = nn.Dropout(dropout)
        self.pyramid = ChaosPyramid(dim, self.geom, depth=depth, blocks=blocks,
                                   heads=heads, dropout=dropout, cell=cell,
                                   use_edge_gate=use_edge_gate, mlp_ratio=mlp_ratio)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim)
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, n_classes)
        )
        self.dim = dim

    def forward(self, g: Graph) -> dict:
        h = self.drop(self.inp(g.x))
        out = self.pyramid(g, h)
        out["node_pred"] = self.recon_head(out["node"])
        out["logits"] = self.cls_head(out["pooled"])
        live = [gt for gt in out["gates"] if gt.numel()]
        out["gate_mean"] = (
            torch.cat([gt.mean(0) for gt in live]).mean()
            if live else torch.zeros((), device=h.device)
        )
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.new_zeros(())
    return ((pred[mask] - target[mask]) ** 2).mean()


def masked_corr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pearson r over all pixel values of the masked patches: does the model
    reproduce *structure* (ink here, paper there) rather than the local mean?"""
    if mask.sum() == 0:
        return pred.new_zeros(())
    p = pred[mask].reshape(-1)
    t = target[mask].reshape(-1)
    # a predictor that is the same at every node has no structure to score
    # (its values still differ per patch channel, so check across nodes)
    across_nodes = float(pred[mask].std(dim=0).max())
    scale = float(pred[mask].abs().mean().clamp(min=1e-9))
    if across_nodes < 1e-3 * scale:  # constant up to float noise
        return pred.new_zeros(())
    p = p - p.mean()
    t = t - t.mean()
    return (p * t).sum() / (p.norm() * t.norm() + 1e-9)


def total_loss(out: dict, target: torch.Tensor, mask: torch.Tensor,
               sparsity: float = 1e-3, cls_target: Optional[torch.Tensor] = None) -> dict:
    rec = masked_mse(out["node_pred"], target, mask)
    loss = rec + sparsity * out["gate_mean"]
    parts = {"recon": rec.item(), "gate": out["gate_mean"].item()}
    if cls_target is not None:
        ce = nn.functional.cross_entropy(out["logits"], cls_target)
        loss = loss + ce
        parts["ce"] = ce.item()
    return {"loss": loss, **parts}
