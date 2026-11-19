"""Minimal graph container + geometric helpers.

Deliberately PyTorch-only: no torch_geometric dependency in the critical path.
`mandalanet.pyg_adapter` converts these graphs to `torch_geometric.data.Data`
if PyG happens to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

EDGE_TYPES = ("radial", "circum", "spoke")  # hex neighbour / ring / long hub line


@dataclass
class Graph:
    x: torch.Tensor                      # (N, F)
    pos: torch.Tensor                    # (N, 2) normalised to roughly the unit disk
    edge_index: torch.Tensor             # (2, E) [src, dst], both directions
    edge_type: Optional[torch.Tensor] = None   # (E,) long in [0, len(EDGE_TYPES))
    batch: Optional[torch.Tensor] = None       # (N,) long, graph id per node

    def __post_init__(self) -> None:
        if self.batch is None:
            self.batch = torch.zeros(self.x.shape[0], dtype=torch.long)
        if self.edge_type is None:
            self.edge_type = torch.zeros(self.edge_index.shape[1], dtype=torch.long)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_graphs(self) -> int:
        return int(self.batch.max()) + 1 if self.batch.numel() else 0

    def to(self, device) -> "Graph":
        return Graph(
            self.x.to(device),
            self.pos.to(device),
            self.edge_index.to(device),
            self.edge_type.to(device),
            self.batch.to(device),
        )

    def with_x(self, x: torch.Tensor) -> "Graph":
        return Graph(x, self.pos, self.edge_index, self.edge_type, self.batch)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def dedupe_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    key = edge_index[0].long() * num_nodes + edge_index[1].long()
    uniq, inv = torch.unique(key, return_inverse=True)
    return torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0), inv


def knn_edges(pos: torch.Tensor, k: int = 6) -> torch.Tensor:
    """k nearest neighbours, symmetric, no self loops."""
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    idx = d.topk(min(k, pos.shape[0] - 1), largest=False).indices          # (N, k)
    src = torch.arange(pos.shape[0]).repeat_interleave(idx.shape[1])
    dst = idx.reshape(-1)
    e = torch.stack([src, dst])
    e = make_undirected(e)
    keep = e[0] != e[1]
    return dedupe_edges(e[:, keep], pos.shape[0])[0]


def radius_edges(pos: torch.Tensor, radius: float, max_k: int = 24) -> torch.Tensor:
    d = torch.cdist(pos, pos)
    n = pos.shape[0]
    mask = (d < radius) & ~torch.eye(n, dtype=torch.bool)
    src, dst = mask.nonzero(as_tuple=True)
    if src.numel() == 0:
        return knn_edges(pos, max_k)
    deg = torch.zeros(n).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    over = deg > max_k
    if over.any():
        keep = ~over[src]
        src, dst = src[keep], dst[keep]
    return dedupe_edges(torch.stack([src, dst]), n)[0]


def batch_graphs(graphs: list[Graph]) -> Graph:
    x = torch.cat([g.x for g in graphs], 0)
    pos = torch.cat([g.pos for g in graphs], 0)
    edge_type = torch.cat([g.edge_type for g in graphs], 0)
    edges, batches, offset = [], [], 0
    for i, g in enumerate(graphs):
        edges.append(g.edge_index + offset)
        batches.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        offset += g.num_nodes
    return Graph(x, pos, torch.cat(edges, 1), edge_type, torch.cat(batches))


# --------------------------------------------------------------------------- #
# pooling (used by the pyramid)
# --------------------------------------------------------------------------- #
def _per_graph_minmax(pos: torch.Tensor, batch: torch.Tensor, n_graphs: int):
    big = torch.full((n_graphs, 2), float("inf"))
    small = torch.full((n_graphs, 2), float("-inf"))
    mn = big.index_reduce_(0, batch, pos, "amin")
    mx = small.index_reduce_(0, batch, pos, "amax")
    return mn, mx


def segment_mean(x: torch.Tensor, assign: torch.Tensor, n_out: int) -> torch.Tensor:
    out = torch.zeros((n_out, x.shape[1]), dtype=x.dtype, device=x.device)
    out.index_add_(0, assign, x)
    cnt = torch.zeros(n_out, dtype=x.dtype, device=x.device).index_add_(
        0, assign, torch.ones_like(assign, dtype=x.dtype)
    )
    return out / cnt.clamp(min=1.0).unsqueeze(-1)


def pool_graph(g: Graph, cell: float, min_nodes: int = 6):
    """Spatial cell pooling.

    Returns (coarse_graph, assign) or None if the graph cannot be coarsened any
    further. `assign[i]` is the coarse node that fine node i feeds into.
    """
    n_graphs = g.num_graphs
    mn, mx = _per_graph_minmax(g.pos, g.batch, n_graphs)
    span = (mx - mn).clamp(min=1e-6)
    rel = ((g.pos - mn[g.batch]) / span[g.batch]).clamp(0.0, 0.999999)
    grid = max(2, int(round(1.0 / cell)))
    q = (rel * grid).floor().long()
    key = (g.batch * (grid * grid) + q[:, 0] * grid + q[:, 1]).long()
    uniq, assign = torch.unique(key, return_inverse=True)
    n_coarse = uniq.numel()
    if n_coarse < min_nodes:
        return None

    coarse_pos = segment_mean(g.pos, assign, n_coarse)
    coarse_batch = (uniq // (grid * grid)).long()

    src, dst = g.edge_index
    cs, cd = assign[src], assign[dst]
    keep = cs != cd
    cs, cd = cs[keep], cd[keep]
    key_e = cs * n_coarse + cd
    uniq_e = torch.unique(key_e)
    e = torch.stack([uniq_e // n_coarse, uniq_e % n_coarse])
    e = make_undirected(e)
    e, _ = dedupe_edges(e, n_coarse)

    coarse = Graph(
        x=segment_mean(g.x, assign, n_coarse),
        pos=coarse_pos,
        edge_index=e,
        edge_type=torch.zeros(e.shape[1], dtype=torch.long),
        batch=coarse_batch,
    )
    return coarse, assign


# --------------------------------------------------------------------------- #
# geometry features
# --------------------------------------------------------------------------- #
class EdgeGeometry(torch.nn.Module):
    """Turns (pos, edge_index, edge_type) into smooth geometric edge features.

    - log-spaced RBF over edge length / median edge length (scale free)
    - cos/sin of k * dtheta, the angular displacement of the edge: this is the
      term that carries the sketch's n-fold symmetry into attention
    - the two endpoints' radii from the sketch centre
    - one-hot edge type
    """

    def __init__(self, bins: int = 8, harmonics: int = 4, edge_types: int = 3):
        super().__init__()
        self.harmonics = harmonics
        self.edge_types = edge_types
        self.register_buffer(
            "centres", torch.logspace(-1.2, 0.9, bins), persistent=True
        )
        self.register_buffer(
            "widths", torch.full((bins,), 0.55), persistent=True
        )

    @property
    def out_dim(self) -> int:
        return self.centres.numel() + 2 * self.harmonics + 2 + self.edge_types

    def forward(self, g: Graph) -> torch.Tensor:
        if g.num_edges == 0:
            return torch.zeros((0, self.out_dim), dtype=g.pos.dtype, device=g.pos.device)
        src, dst = g.edge_index
        d = g.pos[dst] - g.pos[src]
        rho = d.norm(dim=-1, keepdim=True)
        med = rho.median().clamp(min=1e-6)
        rbf = torch.exp(-((rho / med - self.centres) ** 2) / (self.widths**2))

        ang = torch.atan2(d[:, 1], d[:, 0])
        k = torch.arange(1, self.harmonics + 1, device=ang.device, dtype=ang.dtype)
        ang_feat = torch.cat([torch.cos(k * ang[:, None]), torch.sin(k * ang[:, None])], -1)

        r_src = g.pos[src].norm(dim=-1, keepdim=True)
        r_dst = g.pos[dst].norm(dim=-1, keepdim=True)
        onehot = torch.nn.functional.one_hot(
            torch.clamp(g.edge_type, 0, self.edge_types - 1), self.edge_types
        ).to(ang.dtype)
        return torch.cat([rbf, ang_feat, r_src, r_dst, onehot], -1)


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """softmax over incoming edges, separately per node *and* per channel."""
    neg = torch.full((n, logits.shape[1]), -1e9, dtype=logits.dtype,
                     device=logits.device)
    mx = neg.index_reduce_(0, index, logits, "amax", include_self=True)
    mx = mx.masked_fill(mx <= -1e8, 0.0)
    e = (logits - mx[index]).exp()
    den = torch.zeros((n, logits.shape[1]), dtype=logits.dtype, device=logits.device)
    den = den.index_add_(0, index, e)
    return e / den[index].clamp(min=1e-9)
