"""A recurrent graph reasoner: the drawing as the computation graph.

The state lives on N nodes and is updated by message passing along a fixed
edge set. The same weights are reused at every step, so a task that needs k
propagation steps needs k iterations, and the graph is the only thing that
routes information.

This is the R2 substrate: swap the edge set for a chain, a star, a random
graph or the complete graph, keep the parameters identical, and ask whether
the drawing is a better place to compute.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphReasoner(nn.Module):
    def __init__(self, dim: int = 64, classes: int = 17, steps: int = 16,
                 in_dim: int = 1, out_dim: int | None = None,
                 edge_gate: bool = False):
        super().__init__()
        self.steps = steps
        self.inp = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.step = nn.Embedding(steps, dim)
        self.msg = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cell = nn.GRUCell(dim, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, classes if out_dim is None else out_dim))
        self.edge_gate = edge_gate
        if edge_gate:
            self.gate = nn.Sequential(nn.Linear(2 * dim, 1))

    def forward(self, source: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """source: (B, N) one-hot source, or (B, N, F) for F input channels.

        Returns (B, N, out_dim) -- one output per node.
        """
        if source.dim() == 2:
            source = source[..., None]
        b, n = source.shape[0], source.shape[1]
        h = self.inp(source)
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=n).clamp(min=1).to(h.dtype)[None, :, None]
        for t in range(self.steps):
            hi, hj = h[:, src], h[:, dst]
            m = self.msg(torch.cat([hi, hj], -1))
            if self.edge_gate:
                m = m * torch.sigmoid(self.gate(torch.cat([hi, hj], -1)))
            agg = torch.zeros(b, n, h.shape[-1], dtype=h.dtype, device=h.device)
            agg = agg.scatter_add_(1, dst[None, :, None].expand(b, -1, h.shape[-1]), m)
            agg = agg / deg
            h = h + self.step.weight[t][None, None]
            h = self.cell(agg.reshape(b * n, -1), h.reshape(b * n, -1)).reshape(b, n, -1)
        return self.out(h)


def adjacency(n: int, edge_index: torch.Tensor) -> torch.Tensor:
    a = torch.zeros(n, n, dtype=torch.float32)
    a[edge_index[0], edge_index[1]] = 1.0
    return a


def distances(n: int, edge_index: torch.Tensor, steps: int) -> torch.Tensor:
    """BFS from every node, capped at `steps`. dist[s, t] = hops, or `steps`."""
    a = adjacency(n, edge_index)
    a.fill_diagonal_(0.0)
    seen = torch.eye(n, dtype=torch.bool)
    dist = torch.full((n, n), steps, dtype=torch.long)
    dist[seen] = 0
    frontier = seen.clone()
    for t in range(1, steps):
        frontier = ((frontier.float() @ a) > 0) & ~seen
        dist[frontier] = t
        seen |= frontier
        if bool(frontier.sum() == 0):
            break
    return dist


def chain_edges(n: int) -> torch.Tensor:
    i = torch.arange(n - 1)
    e = torch.stack([i, i + 1])
    return torch.cat([e, e.flip(0)], 1)


def star_edges(n: int) -> torch.Tensor:
    i = torch.arange(1, n)
    e = torch.stack([torch.zeros_like(i), i])
    return torch.cat([e, e.flip(0)], 1)


def full_edges(n: int) -> torch.Tensor:
    i, j = torch.triu_indices(n, n, offset=1)
    e = torch.stack([i, j])
    return torch.cat([e, e.flip(0)], 1)


def random_edges(n: int, m: int, seed: int = 0) -> torch.Tensor:
    """A connected random graph with exactly m undirected edges.

    A uniform random graph at the drawings' density is usually disconnected,
    which turns BFS into a "which nodes are unreachable" problem and makes the
    control incomparable. A random spanning tree plus the remaining edges keeps
    the budget and the connectivity, and only the topology is random.
    """
    assert m >= n - 1
    g = torch.Generator().manual_seed(seed)
    seen: set[tuple[int, int]] = set()
    for i in range(1, n):
        j = int(torch.randint(0, i, (1,), generator=g))
        seen.add((j, i))
    while len(seen) < m:
        a = int(torch.randint(0, n, (1,), generator=g))
        b = int(torch.randint(0, n - 1, (1,), generator=g))
        if b >= a:
            b += 1
        seen.add((min(a, b), max(a, b)))
    e = torch.tensor(sorted(seen)).T
    return torch.cat([e, e.flip(0)], 1)


def rewired_edges(n: int, edge_index: torch.Tensor, seed: int = 0,
                  rounds: int = 10) -> torch.Tensor:
    """Double-edge swaps: identical degrees, identical edge count, no geometry.

    This is the control the reconstruction ablation used, generalised to any
    graph. If a rewired copy of the drawing computes like the drawing, then
    whatever the drawing contributes comes from its degree sequence, not from
    which node is wired to which.
    """
    e = edge_index[:, edge_index[0] < edge_index[1]]
    pairs = [tuple(sorted(p)) for p in e.T.tolist()]
    present = set(pairs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(rounds * len(pairs)):
        i = int(torch.randint(0, len(pairs), (1,), generator=g))
        j = int(torch.randint(0, len(pairs), (1,), generator=g))
        if i == j:
            continue
        a, b = pairs[i]
        c, d = pairs[j]
        if float(torch.rand((), generator=g)) < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        n1, n2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if n1 in present or n2 in present:
            continue
        present.discard(pairs[i])
        present.discard(pairs[j])
        present.update((n1, n2))
        pairs[i], pairs[j] = n1, n2
    t = torch.tensor(sorted(present), dtype=torch.long).T
    return torch.cat([t, t.flip(0)], 1)
