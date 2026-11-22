"""Causal sequence model with a pluggable pair bias.

This is the bridge between the drawing experiments and language. Every token
has a position; attention can be biased by the geometry between two positions

    B_ij = MLP( RBF(|p_i - p_j|), cos k dtheta, sin k dtheta )

which is the same term `RadialAttention` adds on graph edges, here computed for
every pair so it works on a sequence. What the model is allowed to know is
chosen by `pos`:

    none        nothing about position
    abs1d       learned embedding of the raster index
    abs2d       MLP of Fourier features of the absolute (x, y)
    geom        relative distance + angle from the true coordinates
                (translation invariant; the distance part is rotation
                invariant, the angle part is not)
    geom_rbf    distance only: the rotation-invariant half, as an ablation
    geom_shuf   the same machinery on shuffled coordinates: a capacity control

The point of `geom` is not that relative geometry is magic -- it is that
distance survives a rotation the absolute-position model has never seen. R0
tests exactly that, and separates the distance term from the angle term.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairGeometry(nn.Module):
    """RBF distance + angular harmonics for every ordered pair of positions."""

    def __init__(self, heads: int, bins: int = 8, harmonics: int = 4,
                 hidden: int = 64, use_angle: bool = True):
        super().__init__()
        self.harmonics = harmonics if use_angle else 0
        self.register_buffer("centres", torch.linspace(0.0, 2.9, bins))
        self.register_buffer("widths", torch.full((bins,), 0.28))
        self.out_dim = bins + 2 * self.harmonics
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, hidden), nn.GELU(), nn.Linear(hidden, heads)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def features(self, coords: torch.Tensor) -> torch.Tensor:
        d = coords[:, :, None, :] - coords[:, None, :, :]        # (B, N, N, 2)
        rho = d.norm(dim=-1, keepdim=True)
        rbf = torch.exp(-((rho - self.centres) / self.widths) ** 2)
        if self.harmonics == 0:
            return rbf
        ang = torch.atan2(d[..., 1], d[..., 0])
        k = torch.arange(1, self.harmonics + 1, device=ang.device, dtype=ang.dtype)
        harm = torch.cat([torch.cos(k * ang[..., None]),
                          torch.sin(k * ang[..., None])], -1)
        return torch.cat([rbf, harm], -1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(feats).permute(0, 3, 1, 2)                # (B, H, N, N)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0,
                 mlp_ratio: float = 4.0, bias: Optional[PairGeometry] = None):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        self.bias = bias
        self.drop = dropout

    def forward(self, x: torch.Tensor, feats: Optional[torch.Tensor],
                causal: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        mask = causal
        if self.bias is not None and feats is not None:
            mask = causal + self.bias(feats)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, -1))
        return x + self.ff(self.ln2(x))


class SeqLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 96, depth: int = 3, heads: int = 4,
                 max_len: int = 256, pos: str = "geom", dropout: float = 0.1,
                 harmonics: int = 4, rbf_bins: int = 8):
        super().__init__()
        assert pos in ("none", "abs1d", "abs2d", "geom", "geom_rbf",
                       "geom_shuf"), pos
        self.pos = pos
        self.tok = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(dropout)
        self.pos1d = nn.Embedding(max_len, dim) if pos == "abs1d" else None
        if pos == "abs2d":
            self.freqs = 6
            feat_dim = 2 + 4 * self.freqs
            self.pos2d = nn.Sequential(
                nn.Linear(feat_dim, dim), nn.GELU(), nn.Linear(dim, dim)
            )
        else:
            self.pos2d = None
        make_bias = pos in ("geom", "geom_rbf", "geom_shuf")
        self.blocks = nn.ModuleList([
            Block(dim, heads, dropout,
                  bias=PairGeometry(heads, rbf_bins, harmonics,
                                    use_angle=pos != "geom_rbf")
                  if make_bias else None)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.max_len = max_len

    def pos_features(self, coords: torch.Tensor) -> torch.Tensor:
        f = torch.arange(1, self.freqs + 1, device=coords.device, dtype=coords.dtype) * math.pi
        x, y = coords[..., 0:1], coords[..., 1:2]
        return torch.cat([coords,
                          torch.sin(f * x), torch.cos(f * x),
                          torch.sin(f * y), torch.cos(f * y)], -1)

    def forward(self, idx: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b, n = idx.shape
        coords = coords[:, :n]
        x = self.tok(idx)
        if self.pos == "abs1d":
            x = x + self.pos1d(torch.arange(n, device=idx.device))[None]
        elif self.pos == "abs2d":
            x = x + self.pos2d(self.pos_features(coords))
        x = self.drop(x)

        causal = torch.full((n, n), float("-inf"), device=idx.device, dtype=x.dtype)
        causal = torch.triu(causal, diagonal=1)[None, None]
        feats = None
        if self.pos in ("geom", "geom_rbf", "geom_shuf"):
            feats = self.blocks[0].bias.features(coords)
        for block in self.blocks:
            x = block(x, feats, causal)
        return self.head(self.ln_f(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def bits_per_token(logits: torch.Tensor, target: torch.Tensor) -> float:
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         target.reshape(-1), reduction="sum")
    return float(ce) / (target.numel() * math.log(2))
