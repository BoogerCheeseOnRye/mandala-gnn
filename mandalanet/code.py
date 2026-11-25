"""Byte-level code model with a pluggable positional mechanism.

R0 showed the drawing's geometry is inert as a *sequence* prior: an AR model
predicts token t+1, whose position is already fixed by the order, so there is
no query position for geometry to route to. R1 therefore tests the surviving
half of the idea — the *mechanism* (a learned pair bias with an RBF over
relative position, plus structure features) — on relations the corpus actually
has: indentation, bracket depth, line distance.

    rope          rotary position on q/k (the baseline to beat)
    abs           learned absolute position embedding
    bias_dist     learned bias from RBF(|i - j|) only; no RoPE, no abs
    bias_struct   learned bias from RBF(|i - j|) + indentation, bracket depth,
                  line distance, same-line flag; no RoPE, no abs
    bias_shuf     the same features with the per-token structural channels
                  permuted across positions: a capacity control

In the bias arms the bias is the *only* source of positional information, so
the comparison is "can this mechanism replace RoPE", not "can it decorate it".
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Rotary position embedding, applied to q and k."""

    def __init__(self, dh: int, max_len: int, base: float = 10000.0):
        super().__init__()
        assert dh % 2 == 0, "head dim must be even for RoPE"
        inv = 1.0 / (base ** (torch.arange(0, dh, 2).float() / dh))
        f = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        n = x.shape[2]                                    # (B, H, L, dh)
        c = self.cos[offset:offset + n][None, None]
        s = self.sin[offset:offset + n][None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class PairBias(nn.Module):
    """Per-head additive attention bias from a pair-feature tensor."""

    def __init__(self, heads: int, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, heads)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:   # (B, N, N, F)
        return self.mlp(feats).permute(0, 3, 1, 2)            # (B, H, N, N)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0,
                 rope: Optional[RoPE] = None, bias: Optional[PairBias] = None):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.rope = rope
        self.bias = bias
        self.drop = dropout

    def forward(self, x: torch.Tensor, feats: Optional[torch.Tensor],
                mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        m = mask if self.bias is None or feats is None else mask + self.bias(feats)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m, dropout_p=self.drop if self.training else 0.0
        )
        x = x + self.out(a.transpose(1, 2).reshape(b, n, -1))
        return x + self.ff(self.ln2(x))

    def forward_cached(self, x: torch.Tensor, past, offset: int):
        """One or more new tokens against a (k, v) cache. Returns x and the new cache."""
        b, n, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q, offset), self.rope(k, offset)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        if n == 1 and past is not None:
            attn_mask = None
        else:
            total = k.shape[2]
            mask = torch.zeros(n, total, device=x.device, dtype=x.dtype)
            mask[:, total - n:] = torch.triu(
                torch.full((n, n), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1)
            attn_mask = mask[None, None]
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop if self.training else 0.0)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, -1))
        return x + self.ff(self.ln2(x)), (k, v)


class CodeLM(nn.Module):
    def __init__(self, vocab: int = 256, dim: int = 96, depth: int = 3,
                 heads: int = 4, ctx: int = 256, pos: str = "rope",
                 feat_dim: int = 15, dropout: float = 0.1):
        super().__init__()
        assert pos in ("rope", "abs", "bias_dist", "bias_struct", "bias_shuf"), pos
        self.pos, self.ctx = pos, ctx
        self.tok = nn.Embedding(vocab, dim)
        self.drop = nn.Dropout(dropout)
        self.pos1d = nn.Embedding(ctx, dim) if pos == "abs" else None
        rope = RoPE(dim // heads, ctx) if pos == "rope" else None
        make_bias = pos in ("bias_dist", "bias_struct", "bias_shuf")
        self.feat_dim = 8 if pos == "bias_dist" else feat_dim
        self.blocks = nn.ModuleList([
            Block(dim, heads, dropout, rope=rope,
                  bias=PairBias(heads, self.feat_dim) if make_bias else None)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n = idx.shape
        x = self.tok(idx)
        if self.pos == "abs":
            x = x + self.pos1d(torch.arange(n, device=idx.device))[None]
        x = self.drop(x)
        mask = torch.triu(torch.full((n, n), float("-inf"), device=idx.device,
                                     dtype=x.dtype), diagonal=1)[None, None]
        for block in self.blocks:
            x = block(x, feats, mask)
        return self.head(self.ln_f(x))

    def forward_cached(self, idx: torch.Tensor, cache=None):
        """Incremental forward: (B, n) new tokens plus a list of (k, v) per block.

        Returns the logits for the new tokens and the updated cache. Positions
        come from the cache length, so the result matches the full forward.
        """
        assert self.pos in ("rope", "abs"), "cached decoding is only wired for rope/abs"
        b, n = idx.shape
        offset = 0 if cache is None else cache[0][0].shape[2]
        x = self.tok(idx)
        if self.pos == "abs":
            pos = torch.arange(offset, offset + n, device=idx.device)
            x = x + self.pos1d(pos)[None]
        x = self.drop(x)
        new = []
        for i, block in enumerate(self.blocks):
            past = None if cache is None else cache[i]
            x, kv = block.forward_cached(x, past, offset)
            new.append(kv)
        return self.head(self.ln_f(x)), new

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# pair features
# --------------------------------------------------------------------------- #
RBF_CENTRES = torch.logspace(0.0, math.log10(256.0), 8)


def pair_features(d: torch.Tensor, indent: torch.Tensor, bracket: torch.Tensor,
                  line: torch.Tensor, kind: str) -> torch.Tensor:
    """(B, N, N, F) pair features for a window.

    d        (B, N) byte offsets inside the window
    indent   (B, N) leading spaces of the line, in units of 4
    bracket  (B, N) running bracket depth, clipped to 0..15
    line     (B, N) line index inside the window
    """
    dist = (d[:, :, None] - d[:, None, :]).clamp(min=0).float()      # (B, N, N)
    rbf = torch.exp(-((torch.log1p(dist)[..., None] - RBF_CENTRES.view(1, 1, 1, -1)) / 0.35) ** 2)
    if kind == "dist":
        return rbf
    di = (indent[:, :, None] - indent[:, None, :]).float().clamp(-8, 8) / 8.0
    db = (bracket[:, :, None] - bracket[:, None, :]).float().clamp(-8, 8) / 8.0
    dl = (line[:, :, None] - line[:, None, :]).float().clamp(0, 32) / 32.0
    same = (line[:, :, None] == line[:, None, :]).float()
    return torch.cat([rbf, di[..., None], di.abs()[..., None],
                      db[..., None], db.abs()[..., None],
                      same[..., None], dl[..., None],
                      (dl == 0).float()[..., None]], -1)


def bits_per_byte(logits: torch.Tensor, target: torch.Tensor) -> float:
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         target.reshape(-1), reduction="sum")
    return float(ce) / (target.numel() * math.log(2))
