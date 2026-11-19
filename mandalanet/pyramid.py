"""Hierarchical residual pyramid -- the 'chaos engine'.

Level l sees the sketch at scale l: local mesh pattern at l=0, whole mandala
at l=depth-1. Each level is a stack of radial-attention blocks plus a learned
downsample; the way back up is a gated residual injection

    h_fine <- h_fine + sigmoid(MLP[h_up, h_fine]) * h_up

so a level keeps its local detail unless the coarse view disagrees with it.
Graphs are coarsened spatially (cell pooling), not resampled: the topology at
level l+1 is literally the merged cell adjacency, so it adapts to the sketch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .graph import EdgeGeometry, Graph, pool_graph, segment_mean
from .layers import RadialBlock


class ChaosPyramid(nn.Module):
    def __init__(self, dim: int, geom: EdgeGeometry, depth: int = 3,
                 blocks: int = 2, heads: int = 4, dropout: float = 0.0,
                 cell: float = 0.125, use_edge_gate: bool = True,
                 mlp_ratio: float = 2.0):
        super().__init__()
        assert depth >= 1
        self.depth, self.cell = depth, cell
        self.geom = geom
        ed = geom.out_dim

        self.levels = nn.ModuleList([
            nn.ModuleList([
                RadialBlock(dim, ed, heads, dropout, mlp_ratio, use_edge_gate)
                for _ in range(blocks)
            ]) for _ in range(depth)
        ])
        self.down = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.up = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(depth)
        ])
        self.fuse_block = nn.ModuleList([
            RadialBlock(dim, ed, heads, dropout, mlp_ratio, use_edge_gate)
            for _ in range(depth)
        ])
        self.scale_weight = nn.Parameter(torch.zeros(depth))

    def forward(self, g: Graph, h: torch.Tensor):
        feats = [self.geom(g)]
        hs = [h]
        gs = [g]
        assigns = []
        gates = []
        alphas = []

        # ---- descend ----
        for l in range(self.depth):
            for blk in self.levels[l]:
                hs[l], gate = blk(hs[l], gs[l], feats[l])
                if gate is not None:
                    gates.append(gate)
                if l == 0 and blk.attn.last_alpha is not None:
                    alphas.append(blk.attn.last_alpha)
            if l == self.depth - 1:
                break
            pooled = pool_graph(gs[l], cell=min(0.5, self.cell * (2 ** l)))
            if pooled is None:
                break
            coarse, assign = pooled
            feats.append(self.geom(coarse))
            hs.append(self.down[l](segment_mean(hs[l], assign, coarse.num_nodes)))
            gs.append(coarse)
            assigns.append(assign)

        used = len(hs)
        h = hs[-1]
        for l in range(used - 2, -1, -1):
            up = self.up[l](h)[assigns[l]]
            gate = torch.sigmoid(self.fuse[l](torch.cat([up, hs[l]], dim=-1)))
            h = hs[l] + gate * up
            h, _ = self.fuse_block[l](h, gs[l], feats[l])
            if l == 0 and self.fuse_block[l].attn.last_alpha is not None:
                alphas.append(self.fuse_block[l].attn.last_alpha)

        hs[0] = h  # the refined finest level, not the pre-climb one

        # ---- multi-scale read-out ----
        pooled = []
        for l in range(used):
            m = segment_mean(hs[l], gs[l].batch, gs[l].num_graphs)
            mx = torch.full((gs[l].num_graphs, hs[l].shape[1]), -1e9,
                            dtype=hs[l].dtype, device=hs[l].device)
            mx = mx.index_reduce_(0, gs[l].batch, hs[l], "amax")
            mx = torch.where(mx <= -1e8, torch.zeros_like(mx), mx)
            pooled.append(torch.cat([m, mx], -1))

        w = torch.softmax(self.scale_weight[:used], dim=0)
        # levels have different node counts; their pooled vectors already agree
        readout = sum(wi * p for wi, p in zip(w, pooled))

        return {
            "node": hs[0],
            "pooled": readout,
            "level_weights": w,
            "gates": gates,
            "alphas": alphas,
            "depth_used": used,
        }
