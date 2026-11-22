"""Synthetic drawings from the three lattices.

R0 needs thousands of drawings with the same topology so that held-out samples
are *new drawings*, not new crops of the same three photographs. This renders
each lattice as line art with a fresh hand every time: node jitter, rotation,
scale, stroke width, ink and paper. The geometry the model is given is the
geometry that was drawn, so the pair bias can be scored honestly.

`python -m mandalanet.synth --out runs/synth-preview.png` writes a contact
sheet of six samples per layout.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from PIL import Image, ImageDraw, ImageFilter

from .data import node_patches
from .lattices import RADIAL, Mesh, build_mesh

LAYOUTS = ("hex", "rings", "chart")
IMAGE = 128


def _transform(pos: torch.Tensor, size: float, rng: torch.Generator):
    p = pos - size / 2.0
    a = float(torch.rand((), generator=rng)) * 2.0 * math.pi
    s = 0.90 + float(torch.rand((), generator=rng)) * 0.18
    t = (torch.rand(2, generator=rng) - 0.5) * 5.0
    c, sn = math.cos(a), math.sin(a)
    x = (c * p[:, 0] - sn * p[:, 1]) * s
    y = (sn * p[:, 0] + c * p[:, 1]) * s
    return torch.stack([x, y], -1) + size / 2.0 + t


def _hex_polygons(mesh: Mesh) -> list[list[int]]:
    n = mesh.pos_px.shape[0]
    near: list[list[int]] = [[] for _ in range(n)]
    for k in range(mesh.edge_index.shape[1]):
        if int(mesh.edge_type[k]) == RADIAL:
            near[int(mesh.edge_index[0, k])].append(int(mesh.edge_index[1, k]))
    out = []
    for i, ns in enumerate(near):
        if len(ns) >= 5:
            cx, cy = mesh.pos_px[i]
            ns = sorted(ns, key=lambda j: math.atan2(
                float(mesh.pos_px[j, 1] - cy), float(mesh.pos_px[j, 0] - cx)))
            out.append(ns)
    return out


def render(mesh: Mesh, rng: torch.Generator, size: int = IMAGE,
           supersample: int = 2, layout: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """One drawing plus the pixel positions the ink was drawn at."""
    jitter = torch.randn(mesh.pos_px.shape, generator=rng) * 0.8
    pos = _transform(mesh.pos_px + jitter, size, rng)
    ss = supersample
    w = size * ss
    paper = int(238 + float(torch.rand((), generator=rng)) * 14)
    img = Image.new("L", (w, w), paper)
    d = ImageDraw.Draw(img)

    def ink(lo: int, hi: int) -> int:
        return int(lo + float(torch.rand((), generator=rng)) * (hi - lo))

    def width(lo: float, hi: float) -> int:
        return max(1, int(round((lo + float(torch.rand((), generator=rng)) * (hi - lo)) * ss)))

    if layout == "hex":
        pos_px = pos
        for ns in _hex_polygons(mesh):
            pts = [(float(pos[j, 0]) * ss, float(pos[j, 1]) * ss) for j in ns]
            d.polygon(pts, outline=ink(35, 125))
            d.line(pts + pts[:1], fill=ink(35, 110), width=width(0.7, 2.2))
        # a few of the long radial lines that cross the tessellation in the sketch
        for k in range(mesh.edge_index.shape[1]):
            if int(mesh.edge_type[k]) != RADIAL and float(torch.rand((), generator=rng)) < 0.3:
                a, b = int(mesh.edge_index[0, k]), int(mesh.edge_index[1, k])
                d.line([float(pos[a, 0]) * ss, float(pos[a, 1]) * ss,
                        float(pos[b, 0]) * ss, float(pos[b, 1]) * ss],
                       fill=ink(120, 185), width=width(0.3, 0.8))
    elif layout == "rings":
        radii = pos.norm(dim=-1)
        for r in sorted(set(round(float(v), 1) for v in radii)):
            if r < 6 or (radii - r).abs().min() > 0.6 or int((radii - r).abs().lt(0.6).sum()) < 6:
                continue
            box = [w / 2 - r * ss, w / 2 - r * ss, w / 2 + r * ss, w / 2 + r * ss]
            d.ellipse(box, outline=ink(40, 130), width=width(0.7, 1.8))
        for k in range(mesh.edge_index.shape[1]):
            a, b = int(mesh.edge_index[0, k]), int(mesh.edge_index[1, k])
            g = ink(50, 150)
            d.line([float(pos[a, 0]) * ss, float(pos[a, 1]) * ss,
                    float(pos[b, 0]) * ss, float(pos[b, 1]) * ss],
                   fill=g, width=width(0.4, 1.2))
    else:  # chart
        for k in range(mesh.edge_index.shape[1]):
            a, b = int(mesh.edge_index[0, k]), int(mesh.edge_index[1, k])
            d.line([float(pos[a, 0]) * ss, float(pos[a, 1]) * ss,
                    float(pos[b, 0]) * ss, float(pos[b, 1]) * ss],
                   fill=ink(25, 110), width=width(0.8, 2.6))
        for i in range(pos.shape[0]):
            if float(pos[i].norm()) > size * 0.40:
                r = (1.2 + float(torch.rand((), generator=rng)) * 1.6) * ss
                d.ellipse([float(pos[i, 0]) * ss - r, float(pos[i, 1]) * ss - r,
                           float(pos[i, 0]) * ss + r, float(pos[i, 1]) * ss + r],
                          fill=ink(30, 90))

    img = img.resize((size, size), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(0.3 + float(torch.rand((), generator=rng)) * 0.45))
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float()
    arr = (arr.reshape(1, 1, size, size) / 255.0
           + torch.randn(1, 1, size, size, generator=rng) * 0.012).clamp(0, 1)
    coords = (pos - size / 2.0) / (size / 2.0)
    return arr, coords


def raster_order(coords: torch.Tensor) -> torch.Tensor:
    """Scanline order: group rows in 2 px bands, then left to right."""
    y = coords[:, 1] * 64.0
    key = y.round() * 4096.0 + coords[:, 0] * 2.0
    return torch.argsort(key)


def sample(layout: str, rng: torch.Generator, size: int = IMAGE,
           win: int = 8, grid: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patches, positions and the drawing for one synthetic sample."""
    mesh = build_mesh(layout, size=size)
    img, coords = render(mesh, rng, size=size, layout=layout)
    pos_px = (coords + 1.0) * size / 2.0
    patches = node_patches(img, pos_px, win=win, grid=grid)
    order = raster_order(coords)
    return patches[order], coords[order], img


def preview(out: str, per_layout: int = 6, seed: int = 0) -> str:
    rng = torch.Generator().manual_seed(seed)
    w = IMAGE * per_layout
    sheet = Image.new("L", (w, IMAGE * len(LAYOUTS)), 255)
    for row, layout in enumerate(LAYOUTS):
        for col in range(per_layout):
            _, _, img = sample(layout, rng)
            arr = (img.squeeze().clamp(0, 1) * 255).byte()
            sheet.paste(Image.fromarray(arr.numpy()), (col * IMAGE, row * IMAGE))
    sheet.save(out)
    return out


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/synth-preview.png")
    p.add_argument("--per-layout", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(preview(a.out, a.per_layout, a.seed))
