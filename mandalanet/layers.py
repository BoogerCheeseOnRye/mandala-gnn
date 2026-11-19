"""Geometric attention layer: the 'hex node + radial edge' message passing.

Key idea: attention is *geometry aware*. The logit has three parts

    q_i . k_j            content
  + B(edge_feat_ij)      learnable bias from rho-RBF, angular harmonics, radii, type
  + per-head k*cos/sin   folded out of the Angelic n-fold symmetry

and every message is multiplied by a learned edge gate g_ij in (0,1). The gate
is what lets the topology adapt: edges the drawing does not support get closed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import Graph, segment_softmax


class RadialAttention(nn.Module):
    def __init__(self, dim: int, edge_dim: int, heads: int = 4,
                 dropout: float = 0.0, use_edge_gate: bool = True):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.use_edge_gate = use_edge_gate
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.bias = nn.Sequential(
            nn.Linear(edge_dim, dim), nn.GELU(), nn.Linear(dim, heads)
        )
        self.gate = (
            nn.Sequential(nn.Linear(edge_dim, dim // 2), nn.GELU(),
                          nn.Linear(dim // 2, heads))
            if use_edge_gate else None
        )
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, g: Graph, edge_feat: torch.Tensor):
        n = h.shape[0]
        self.last_alpha = None
        src, dst = g.edge_index
        hn = self.norm(h)
        q, k, v = self.qkv(hn).view(n, 3, self.heads, self.dh).unbind(1)
        logit = (q[src] * k[dst]).sum(-1) / math.sqrt(self.dh)   # (E, H)
        logit = logit + self.bias(edge_feat)
        alpha = segment_softmax(logit, dst, n)                    # (E, H)
        alpha = self.drop(alpha)
        # Keep the attention this pass actually routed, for the live demo.
        # Eval only, detached: no graph retained, no change to the forward.
        if not self.training:
            self.last_alpha = alpha.detach()

        if self.gate is not None:
            gate = torch.sigmoid(self.gate(edge_feat))            # (E, H)
            alpha = alpha * gate
        else:
            gate = None

        msg = alpha.unsqueeze(-1) * v[dst]                        # (E, H, dh)
        agg = torch.zeros((n, self.heads, self.dh), dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, msg)
        return self.out(agg.reshape(n, self.dim)), gate


class RadialBlock(nn.Module):
    """Pre-norm residual geometric-attention block."""

    def __init__(self, dim: int, edge_dim: int, heads: int = 4,
                 dropout: float = 0.0, mlp_ratio: float = 2.0,
                 use_edge_gate: bool = True):
        super().__init__()
        self.attn = RadialAttention(dim, edge_dim, heads, dropout, use_edge_gate)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, g: Graph, edge_feat: torch.Tensor):
        upd, gate = self.attn(h, g, edge_feat)
        h = h + upd
        h = h + self.ff(h)
        return h, gate
