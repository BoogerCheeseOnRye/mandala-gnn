"""Lattices read off the three sketches.

`star-mandala`  -> concentric rings + radial spokes (Rings)
`hub-chart`     -> hub and long spoke lines, lens/triangle ribs (Chart)
`hex-tessellation` -> hexagonal cell centres, 6-fold radial edges (HexMesh)

Every builder returns positions in *pixel* coordinates (so a drawing can be
sampled at them) plus a normalised copy and an edge list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .graph import Graph, dedupe_edges, knn_edges, make_undirected

RADIAL, CIRCUM, SPOKE = 0, 1, 2


@dataclass
class Mesh:
    pos_px: torch.Tensor        # (N, 2) in [0, size]^2
    pos: torch.Tensor           # (N, 2) centred, unit disk
    edge_index: torch.Tensor    # (2, E)
    edge_type: torch.Tensor     # (E,)

    def graph(self, x: torch.Tensor) -> Graph:
        return Graph(x=x, pos=self.pos, edge_index=self.edge_index, edge_type=self.edge_type)


def _finish(pos_px: torch.Tensor, size: float, edges: torch.Tensor,
            types: torch.Tensor, dedupe: bool = True) -> Mesh:
    if dedupe:
        edges, inv = dedupe_edges(edges, pos_px.shape[0])
        # inv maps each *input* edge onto its unique representative; collapse it
        # so there is exactly one type per unique edge.
        collapsed = torch.zeros(edges.shape[1], dtype=types.dtype)
        collapsed[inv] = types
        types = collapsed
    pos = (pos_px - size / 2.0) / (size / 2.0)
    return Mesh(pos_px, pos, edges, types)


def hex_mesh(size: float = 128.0, spacing: float = 9.0, star_edges: bool = True) -> Mesh:
    """Pointy-top honeycomb, as in the hex-tessellation drawing."""
    s = spacing / math.sqrt(3.0)  # circumradius for a given centre spacing
    dx = s * math.sqrt(3.0)
    dy = s * 1.5
    pts, index = [], {}
    m = int(size / s) + 2
    for r in range(-m, m + 1):
        for q in range(-m, m + 1):
            x = dx * (q + r / 2.0)
            y = dy * r
            if abs(x) <= size / 2 - spacing * 0.5 and abs(y) <= size / 2 - spacing * 0.5:
                index[(q, r)] = len(pts)
                pts.append((x, y))
    pos_px = torch.tensor(pts, dtype=torch.float32) + size / 2.0

    near = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    far = [(0, 2), (0, -2), (2, -1), (-2, 1), (2, -2), (-2, 2)] if star_edges else []
    src, dst, typ = [], [], []
    for (q, r), i in index.items():
        for dq, dr in near:
            j = index.get((q + dq, r + dr))
            if j is not None:
                src.append(i); dst.append(j); typ.append(RADIAL)
        for dq, dr in far:
            j = index.get((q + dq, r + dr))
            if j is not None:
                src.append(i); dst.append(j); typ.append(SPOKE)
    e = torch.tensor([src, dst], dtype=torch.long)
    return _finish(pos_px, size, e, torch.tensor(typ, dtype=torch.long))


def grid_mesh(size: float = 128.0, spacing: float = 9.0, diagonals: bool = True) -> Mesh:
    n = int(size // spacing)
    pts = [((i + 0.5) * size / n, (j + 0.5) * size / n) for j in range(n) for i in range(n)]
    pos_px = torch.tensor(pts, dtype=torch.float32)
    idx = lambda i, j: j * n + i
    src, dst, typ = [], [], []
    for j in range(n):
        for i in range(n):
            for di, dj, t in ((1, 0, RADIAL), (0, 1, RADIAL),
                              (1, 1, CIRCUM), (1, -1, CIRCUM)):
                if 0 <= i + di < n and 0 <= j + dj < n:
                    src.append(idx(i, j)); dst.append(idx(i + di, j + dj)); typ.append(t)
    e = torch.tensor([src, dst], dtype=torch.long)
    return _finish(pos_px, size, e, torch.tensor(typ, dtype=torch.long))


def rings_mesh(size: float = 128.0, spokes: int = 12, n_rings: int = 5,
               radius: float | None = None, skip: int = 2) -> Mesh:
    """Concentric spokes as in the star-mandala sketch."""
    if radius is None:
        radius = 0.45 * size
    pts = [(size / 2.0, size / 2.0)]
    pos = [(0.0, 0.0)]
    ring_of = [0]
    for li in range(1, n_rings + 1):
        rr = radius * li / n_rings
        for k in range(spokes):
            a = 2 * math.pi * k / spokes
            pts.append((size / 2.0 + rr * math.cos(a), size / 2.0 + rr * math.sin(a)))
            pos.append((rr * math.cos(a), rr * math.sin(a)))
            ring_of.append(li)
    pos_px = torch.tensor(pts, dtype=torch.float32)
    idx = lambda li, k: 1 + (li - 1) * spokes + (k % spokes)
    src, dst, typ = [], [], []
    for k in range(spokes):                      # hub -> first ring
        src.append(0); dst.append(idx(1, k)); typ.append(SPOKE)
    for li in range(1, n_rings + 1):
        for k in range(spokes):
            src.append(idx(li, k)); dst.append(idx(li, k + 1)); typ.append(CIRCUM)
            src.append(idx(li, k)); dst.append(idx(li, k + skip)); typ.append(CIRCUM)
            if li < n_rings:
                src.append(idx(li, k)); dst.append(idx(li + 1, k)); typ.append(RADIAL)
    for k in range(spokes):                      # pointy star tips
        src.append(idx(n_rings, k)); dst.append(idx(n_rings, k + spokes // 2))
        typ.append(SPOKE)
    e = torch.tensor([src, dst], dtype=torch.long)
    return _finish(pos_px, size, e, torch.tensor(typ, dtype=torch.long))


def chart_mesh(size: float = 128.0, arms: int = 4, per_arm: int = 5) -> Mesh:
    """Hub-and-spoke chart: a centre, four diamond corners, long spokes that
    run corner-to-corner, and lens ribs between neighbouring arm nodes."""
    c = size / 2.0
    pts = [(c, c)]
    index = {}
    corners = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for a, (ux, uy) in enumerate(corners):
        for j in range(1, per_arm + 1):
            t = j / per_arm
            index[(a, j)] = len(pts)
            pts.append((c + ux * t * size * 0.46, c + uy * t * size * 0.46))
    # corner hubs of the outer diamond (as in the drawing's four dots)
    for a, (ux, uy) in enumerate(corners):
        index[(a, "hub")] = len(pts)
        pts.append((c + ux * size * 0.47, c + uy * size * 0.47))
    pos_px = torch.tensor(pts, dtype=torch.float32)
    src, dst, typ = [], [], []
    for a in range(arms):
        src.append(0); dst.append(index[(a, 1)]); typ.append(SPOKE)
        for j in range(1, per_arm):
            src.append(index[(a, j)]); dst.append(index[(a, j + 1)]); typ.append(RADIAL)
        src.append(index[(a, per_arm)]); dst.append(index[(a, "hub")]); typ.append(SPOKE)
        # lens ribs to the next arm
        b = (a + 1) % arms
        for j in range(1, per_arm):
            src.append(index[(a, j)]); dst.append(index[(b, j + 1)]); typ.append(CIRCUM)
            src.append(index[(a, j + 1)]); dst.append(index[(b, j)]); typ.append(CIRCUM)
        src.append(index[(a, "hub")]); dst.append(index[(b, per_arm)]); typ.append(SPOKE)
    # outer diamond edges between corners
    for a in range(arms):
        src.append(index[(a, "hub")]); dst.append(index[((a + 1) % arms, "hub")])
        typ.append(SPOKE)
    e = torch.tensor([src, dst], dtype=torch.long)
    return _finish(pos_px, size, e, torch.tensor(typ, dtype=torch.long))


def rewired_mesh(size: float = 128.0, spacing: float = 9.0, k: int = 6,
                 seed: int = 0) -> Mesh:
    """Honeycomb node positions, random edges. The strict topology control:
    identical sampling points and roughly the same degree, no lattice."""
    base = hex_mesh(size, spacing)
    n = base.pos_px.shape[0]
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(n).repeat_interleave(k)
    dst = torch.randint(0, n, (n * k,), generator=g)
    keep = src != dst
    e = make_undirected(torch.stack([src[keep], dst[keep]]))
    e = dedupe_edges(e, n)[0]
    return _finish(base.pos_px, size, e, torch.zeros(e.shape[1], dtype=torch.long))


def random_mesh(size: float = 128.0, n: int = 180, k: int = 6,
                seed: int = 0, jitter: float = 0.45, spacing: float = 9.0) -> Mesh:
    """Freehand control: jittered lattice / pure random points + kNN edges.

    The control matters: if the hand-drawn hex lattice does not beat this, the
    topology is doing nothing."""
    g = torch.Generator().manual_seed(seed)
    if jitter < 1.0:
        base = hex_mesh(size, spacing).pos_px
        pos_px = base + jitter * spacing * (torch.rand(base.shape, generator=g) - 0.5)
    else:
        pos_px = torch.rand((n, 2), generator=g) * (size - 4.0) + 2.0
    pos_px = pos_px.clamp(2.0, size - 2.0)
    e = knn_edges((pos_px - size / 2.0) / (size / 2.0), k)
    e = make_undirected(e)
    return _finish(pos_px, size, e, torch.zeros(e.shape[1], dtype=torch.long))


def empty_mesh(size: float = 128.0, spacing: float = 9.0) -> Mesh:
    """No edges at all: what the same network becomes without the sketch graph."""
    m = grid_mesh(size, spacing, diagonals=True)
    return Mesh(m.pos_px, m.pos, torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.long))


MESHES = {
    "hex": hex_mesh,
    "grid": grid_mesh,
    "rings": rings_mesh,
    "chart": chart_mesh,
    "random": random_mesh,
    "jitter": lambda size=128.0: random_mesh(size, jitter=0.7),
    "rewired": rewired_mesh,
    "hex_near": lambda size=128.0: hex_mesh(size, star_edges=False),
    "none": empty_mesh,
}


def build_mesh(kind: str, size: float = 128.0, **kw) -> Mesh:
    if kind not in MESHES:
        raise KeyError(f"unknown mesh {kind!r}; have {sorted(MESHES)}")
    return MESHES[kind](size=size, **kw)
