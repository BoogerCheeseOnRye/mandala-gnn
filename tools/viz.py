"""Draw the mesh the model sees, on top of the sketch it came from.

    python -m tools.viz --drawing drawings/star-mandala.jpg --mesh rings --out runs/mesh-rings.png
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.data import IMAGE, load_gray  # noqa: E402
from mandalanet.lattices import build_mesh  # noqa: E402

TYPE_COLOR = {0: (40, 120, 255), 1: (0, 180, 90), 2: (255, 90, 0)}


def render(drawing: str, mesh: str, out: str, size: int = IMAGE, scale: int = 4) -> str:
    img = load_gray(drawing, size)
    arr = (img.squeeze(0).repeat(3, 1, 1) * 255).byte()
    im = Image.fromarray(arr.permute(1, 2, 0).numpy()).resize(
        (size * scale, size * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    m = build_mesh(mesh, size=size)
    for k in range(m.edge_index.shape[1]):
        a, b = m.edge_index[0, k].item(), m.edge_index[1, k].item()
        c = TYPE_COLOR.get(int(m.edge_type[k]), (255, 0, 255))
        d.line([m.pos_px[a, 0] * scale, m.pos_px[a, 1] * scale,
                m.pos_px[b, 0] * scale, m.pos_px[b, 1] * scale], fill=c, width=2)
    r = 3
    for i in range(m.pos_px.shape[0]):
        x, y = m.pos_px[i, 0] * scale, m.pos_px[i, 1] * scale
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 90))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    im.save(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drawing", default="drawings/star-mandala.jpg")
    p.add_argument("--mesh", default="rings")
    p.add_argument("--out", default="")
    a = p.parse_args()
    out = a.out or f"runs/mesh-{a.mesh}.png"
    print(render(a.drawing, a.mesh, out))
