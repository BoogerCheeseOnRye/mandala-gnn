"""Data: the sketches as signals on the mesh.

The task is masked-patch reconstruction. Every node owns a small square patch
of the drawing (a hand-drawn cell), so "reconstruct" literally means "redraw".
Two evaluation sets are kept apart from training:

  train_mask  random cells on 7/8 of the circle
  val_mask    *other* random cells on 7/8 of the circle   (generalisation)
  wedge_mask  every cell in the remaining 1/8 sector      (symmetry inpainting)

The wedge is never optimised: a model that has only memorised positions cannot
fill it in, a model that uses the sketch's angular symmetry can.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .graph import Graph
from .lattices import Mesh, build_mesh

IMAGE = 128
WIN = 8
GRID = 4

# (cx, cy, side): centre and side length, normalised -- side in units of image
# width, cy in units of image height. Photos of paper, not scans, so the crop is
# part of the sketch spec.
CROPS = {
    "star-mandala": (0.43, 0.47, 0.70),
    "hub-chart": (0.50, 0.50, 0.94),
    "hex-tessellation": (0.36, 0.70, 0.58),
}


def load_gray(path: str, size: int = IMAGE, crop: bool = True) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if crop:
        key = os.path.splitext(os.path.basename(path))[0]
        if key in CROPS:
            cx, cy, side = CROPS[key]
            w, h = img.size
            s = side * w
            left = cx * w - s / 2
            top = cy * h - s / 2
            img = img.crop((int(left), int(top), int(left + s), int(top + s)))
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.resize((size, size), Image.LANCZOS)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float()
    return (t.reshape(1, 1, size, size) / 255.0)


def rotate(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    theta = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]], dtype=img.dtype)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="zeros")


def rotate_pos(pos: torch.Tensor, size: float, angle_deg: float) -> torch.Tensor:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    p = pos - size / 2.0
    x = c * p[:, 0] - s * p[:, 1]
    y = s * p[:, 0] + c * p[:, 1]
    return torch.stack([x, y], -1) + size / 2.0


def node_patches(img: torch.Tensor, pos_px: torch.Tensor, win: int = WIN,
                 grid: int = GRID, circle: bool = False) -> torch.Tensor:
    """Sample a `win`x`win` window at every node, pooled to grid x grid values."""
    n, size = pos_px.shape[0], img.shape[-1]
    cx = pos_px[:, 0] / size * 2.0 - 1.0
    cy = pos_px[:, 1] / size * 2.0 - 1.0
    off = (torch.arange(win, dtype=img.dtype) + 0.5) / win - 0.5
    ox = off * 2.0 * win / size
    gx = cx[:, None, None] + ox[None, None, :]
    gy = cy[:, None, None] + ox[None, :, None]
    g = torch.stack([gx.expand(n, win, win), gy.expand(n, win, win)], -1)
    patches = F.grid_sample(img.expand(n, 1, size, size), g, align_corners=False,
                            padding_mode="border")
    if circle:
        # a disc, not a square: removes the "square patches tile a square grid
        # exactly" advantage that the grid control would otherwise get for free
        o = (torch.arange(win, dtype=img.dtype) + 0.5) / win - 0.5
        yy, xx = torch.meshgrid(o, o, indexing="ij")
        patches = patches * (xx**2 + yy**2 <= 0.25).to(img.dtype)
    p = F.adaptive_avg_pool2d(patches, (grid, grid))
    return p.reshape(n, -1)


def patches_to_image(patches: torch.Tensor, pos_px: torch.Tensor, size: int = IMAGE,
                     win: int = WIN) -> torch.Tensor:
    n = patches.shape[0]
    side = int(round(math.sqrt(patches.shape[1])))
    p = F.interpolate(patches.reshape(n, 1, side, side), size=(win, win),
                      mode="bilinear", align_corners=False)
    img = torch.zeros(1, 1, size, size, dtype=patches.dtype)
    wsum = torch.zeros(1, 1, size, size, dtype=patches.dtype)
    for i in range(n):
        x0 = int(round(pos_px[i, 0].item() - win / 2))
        y0 = int(round(pos_px[i, 1].item() - win / 2))
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(size, x0 + win), min(size, y0 + win)
        if xb <= xa or yb <= ya:
            continue
        img[0, 0, ya:yb, xa:xb] += p[i, 0, ya - y0:yb - y0, xa - x0:xb - x0]
        wsum[0, 0, ya:yb, xa:xb] += 1.0
    return img / wsum.clamp(min=1.0)


@dataclass
class ReconExample:
    graph: Graph
    target: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    wedge_mask: torch.Tensor
    mesh: Mesh
    img: torch.Tensor
    masked_input_img: torch.Tensor

    @property
    def patch_dim(self) -> int:
        return int(self.target.shape[1])


def with_mask(ex: "ReconExample", hidden: torch.Tensor) -> Graph:
    """Same drawing, same topology, a different set of hidden cells."""
    x_in = ex.target * (~hidden).float().unsqueeze(-1)
    feats = torch.cat([x_in, hidden.float().unsqueeze(-1)], dim=-1)
    return Graph(feats, ex.mesh.pos, ex.mesh.edge_index, ex.mesh.edge_type)


def angular_window(pos: torch.Tensor, width: float = 0.125, offset: float = 0.0) -> torch.Tensor:
    """A contiguous angular slice of the nodes (the held-out wedge).

    Defined by rank around the circle, not by sector index: a chart-like mesh
    with nodes only on the diagonals would otherwise get an empty sector.
    """
    n = pos.shape[0]
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    order = torch.argsort(theta)
    k = max(3, int(round(width * n)))
    start = int(round(offset * n)) % n
    idx = order[(start + torch.arange(k)) % n]
    out = torch.zeros(n, dtype=torch.bool)
    out[idx] = True
    return out


def sector_of(pos: torch.Tensor, n_sectors: int = 8) -> torch.Tensor:
    th = torch.atan2(pos[:, 1], pos[:, 0])
    s = ((th + math.pi) / (2 * math.pi) * n_sectors).floor().long()
    return s.clamp(0, n_sectors - 1)


def recon_example(drawing: str, mesh: str = "hex", size: int = IMAGE,
                  win: int = WIN, grid: int = GRID, n_sectors: int = 8,
                  val_sector: int = 0, mask_ratio: float = 0.2,
                  seed: int = 0, rotate_deg: float = 0.0,
                  edge_drop: float = 0.0, circle: bool = False) -> ReconExample:
    img = load_gray(drawing, size)
    m = build_mesh(mesh, size=size)
    if rotate_deg:
        img = rotate(img, rotate_deg)
        m.pos_px = rotate_pos(m.pos_px, size, rotate_deg)
        m.pos = (m.pos_px - size / 2.0) / (size / 2.0)

    x = node_patches(img, m.pos_px, win, grid, circle=circle)
    g = torch.Generator().manual_seed(seed)
    wedge = angular_window(m.pos, width=1.0 / n_sectors, offset=val_sector / n_sectors)
    train_pool = (~wedge).nonzero(as_tuple=True)[0]
    perm = train_pool[torch.randperm(train_pool.numel(), generator=g)]
    k = int(round(mask_ratio * train_pool.numel()))
    train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    train_mask[perm[:k]] = True
    val_mask = torch.zeros_like(train_mask)
    val_mask[perm[k:2 * k]] = True

    known = ~(train_mask | val_mask | wedge)
    x_in = x.clone()
    x_in[~known] = 0.0
    feats = torch.cat([x_in, (~known).float().unsqueeze(-1)], dim=-1)

    gi = m.graph(feats)
    if edge_drop > 0:
        keep = torch.rand(gi.num_edges, generator=g) > edge_drop
        gi = Graph(gi.x, gi.pos, gi.edge_index[:, keep], gi.edge_type[keep])

    masked_patches = x * known.float().unsqueeze(-1)
    return ReconExample(
        graph=gi,
        target=x,
        train_mask=train_mask,
        val_mask=val_mask,
        wedge_mask=wedge,
        mesh=m,
        img=img,
        masked_input_img=patches_to_image(masked_patches, m.pos_px, size, win),
    )


@dataclass
class ClassExample:
    graph: Graph
    labels: torch.Tensor


def classification_set(drawings: list[str], angles: list[float],
                       mesh: str = "hex", size: int = IMAGE,
                       noise: float = 0.02, seed: int = 0) -> ClassExample:
    """One graph per (drawing, rotation). Node features are the local patches,
    so the label is only recoverable from how the patches are *arranged*."""
    from .graph import batch_graphs

    g = torch.Generator().manual_seed(seed)
    graphs, labels = [], []
    for label, path in enumerate(drawings):
        base = load_gray(path, size)
        for a in angles:
            img = rotate(base, a) if a else base
            m = build_mesh(mesh, size=size)
            pos = rotate_pos(m.pos_px, size, a) if a else m.pos_px
            x = node_patches(img, pos)
            x = (x + noise * torch.randn(x.shape, generator=g)).clamp(0, 1)
            graphs.append(Graph(x, m.pos, m.edge_index, m.edge_type))
            labels.append(label)
    return ClassExample(batch_graphs(graphs), torch.tensor(labels))


def save_triptych(path: str, *imgs: torch.Tensor, scale: int = 3) -> None:
    arr = torch.cat(list(imgs), dim=-1)
    im = Image.fromarray((arr.clamp(0, 1).squeeze() * 255).byte().numpy())
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
