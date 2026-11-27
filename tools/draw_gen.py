"""Step 6: can these models draw, or only reconstruct?

The step-1 meter asked the 405 k GNN to fill a drawing from an empty canvas and
got r 0.360 against 0.287 for a constant image. That is one real drawing and one
checkpoint. This tool asks the question properly, on the frozen set: 36
drawings, 12 per layout, seeds 1000-1011, never used in training.

Two candidates, same drawings, same scoring:

  * the GNN's masked-fill loop, 8 and 16 passes from an empty canvas,
  * an AR patch decoder at matched parameters (tools/ar_patches.py).

Everything is scored on the rendered image: MSE and Pearson r against the
drawing, seconds per drawing, and the constant-image floor computed from
*train* drawings (fresh seeds), never from the eval set.

    .venv/bin/python tools/draw_gen.py --out runs/draw-gen

Gate (pre-registered in ROADMAP): a candidate must beat the constant floor by
2x MSE and finish in <= 2 s.

Two controls, because "2x the floor" is only meaningful next to what the
representation can express. The *floor* is the mean training image. The
*ceiling* is the drawing's own patches pushed through the same renderer: no
model that outputs one 4x4 patch per node can score better than that, since the
ink between nodes is not in its output space. The useful number is therefore
gain captured, (floor - mse) / (floor - ceiling).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# one OpenMP thread is not a detail on this device: a tiny matmul spends more
# time waking threads than computing. Step 4 measured 1,075 B/s at one thread
# and 754 at eight; the fill loop here is 0.84 s at one and 2.15 s at eight.
# Set before torch loads its OpenMP runtime, so the number is reproducible.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.data import patches_to_image  # noqa: E402
from mandalanet.graph import Graph  # noqa: E402
from mandalanet.lattices import build_mesh  # noqa: E402
from mandalanet.synth import sample  # noqa: E402
from mandalanet.train import Config, build  # noqa: E402
from tools.energy_experiment import pearson  # noqa: E402
from tools.make_eval import load as load_eval  # noqa: E402

CKPTS = {"hex": "runs/main-hex", "rings": "runs/main-rings", "chart": "runs/main-chart"}
ONE_MODEL: str | None = None      # --ckpt: a single model for all three layouts
IMAGE = 128
PASSES = (8, 16)
TRAIN_SEEDS = range(0, 64)          # floor and nothing else; eval seeds start at 1000


def load_gnn(ckpt: str):
    meta = json.load(open(os.path.join(ckpt, "metrics.json")))
    meta.setdefault("config", {})
    cfg = Config(**{k: v for k, v in meta["config"].items()
                    if k in Config.__dataclass_fields__})
    model = build(cfg, 16 + 1, 16)
    ck = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def floor_images(seeds=TRAIN_SEEDS) -> dict[str, torch.Tensor]:
    """The constant-image control: the pixel-wise mean of training drawings.

    Standard "predict the average image" baseline, and unlike a single colour
    it has a correlation to report. Computed from training seeds only.
    """
    out = {}
    for layout in CKPTS:
        imgs = [sample(layout, torch.Generator().manual_seed(s))[2] for s in seeds]
        out[layout] = torch.stack(imgs).mean(0)
    return out


def drawing(layout: str, seed: int):
    patches, coords, img = sample(layout, torch.Generator().manual_seed(seed))
    mesh = build_mesh(layout, size=IMAGE)
    return patches, coords, img, mesh


def graph_for(patches: torch.Tensor, coords: torch.Tensor, hidden: torch.Tensor,
              mesh) -> Graph:
    feats = torch.cat([patches, hidden.float().unsqueeze(-1)], 1)
    return Graph(feats, coords, mesh.edge_index, mesh.edge_type)


def render(patches: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rendered image plus a mask of the pixels any patch actually covered.

    patches_to_image leaves a pixel at zero when no patch overlaps it, which on
    a synthetic drawing (jittered, rotated, sometimes scaled below 1) is 27% of
    the canvas -- the corners. Scoring those pixels would score the renderer's
    hole, not the model, so both the fill and the floor are scored on the
    covered region only. Rendering a field of ones gives exactly that mask.
    """
    pos_px = (coords + 1.0) * IMAGE / 2.0
    out = patches_to_image(patches, pos_px, IMAGE)
    covered = patches_to_image(torch.ones_like(patches), pos_px, IMAGE) > 0.5
    return out, covered


def score(out: torch.Tensor, img: torch.Tensor, covered: torch.Tensor) -> dict:
    a, b = out[covered], img[covered]
    return {"mse": float(((a - b) ** 2).mean()), "r": pearson(a, b),
            "covered": float(covered.float().mean())}


def sheet(rows, out_path: str, scale: int = 2,
          labels=("drawing", "floor", "fill, 8 passes", "fill, 16 passes")) -> str:
    """One row per layout: the drawing, the constant-image floor, and the fills."""
    from PIL import Image, ImageDraw

    w, pad, head = IMAGE * scale, 8, 16
    im = Image.new("L", (len(labels) * (w + pad) + pad,
                         len(rows) * (w + head) + pad), 255)
    d = ImageDraw.Draw(im)
    for r, (layout, imgs) in enumerate(rows):
        for c, (label, t) in enumerate(zip(labels, imgs)):
            x, y = pad + c * (w + pad), pad + r * (w + head)
            d.text((x, y), f"{layout} - {label}", fill=0)
            arr = (t.clamp(0, 1).squeeze() * 255).byte().numpy()
            im.paste(Image.fromarray(arr).resize((w, w), Image.LANCZOS), (x, y + head))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    return out_path


@torch.no_grad()
def fill(model, patches, coords, mesh, passes: int,
         teacher: bool = False) -> tuple[torch.Tensor, float]:
    """Fill in `passes` rounds. `teacher` reveals the true patch after each
    round instead of the model's own, which isolates error compounding from
    error at the round itself: the loop conditions on its own output, and it was
    never trained to."""
    n = patches.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    x = torch.zeros_like(patches)      # the model's answer, what gets scored
    ctx = torch.zeros_like(patches)    # what the next pass is allowed to see
    known = torch.zeros(n, dtype=torch.bool)
    t0 = time.time()
    for i in range(passes):
        cells = order[i::passes]
        pred = model(graph_for(ctx, coords, ~known, mesh))["node_pred"]
        x[cells] = pred[cells]
        ctx[cells] = patches[cells] if teacher else pred[cells]
        known[cells] = True
    return x, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", dest="eval_path", default="runs/budget/eval.json")
    p.add_argument("--out", default="runs/draw-gen")
    p.add_argument("--ckpt", default="",
                   help="one checkpoint for all layouts (e.g. runs/synth-gnn)")
    p.add_argument("--limit", type=int, default=0, help="debug: first N drawings")
    p.add_argument("--sheet", default="", help="write a PNG of target | floor | fills")
    p.add_argument("--teacher", action="store_true",
                   help="diagnostic: same loop, true context after each pass")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)

    ev = load_eval(a.eval_path)
    items = ev["drawings"]["items"][:a.limit or None]
    ckpts = dict.fromkeys(CKPTS, a.ckpt) if a.ckpt else CKPTS
    models = {layout: load_gnn(c) for layout, c in ckpts.items()}

    floor = floor_images()

    rows = []
    seen: set[str] = set()
    pics = []
    for item in items:
        layout, seed = item["layout"], item["seed"]
        patches, coords, img, mesh = drawing(layout, seed)
        _, covered = render(patches, coords)
        row = {"layout": layout, "seed": seed, "nodes": int(patches.shape[0]),
               "floor": score(floor[layout], img, covered),
               "ceiling": score(render(patches, coords)[0], img, covered)}
        fills = []
        for passes in PASSES:
            x, secs = fill(models[layout], patches, coords, mesh, passes)
            out, _ = render(x, coords)
            fills.append(out)
            row[f"fill{passes}"] = {**score(out, img, covered), "seconds": secs}
        if a.teacher:
            xt, secs_t = fill(models[layout], patches, coords, mesh, PASSES[0],
                              teacher=True)
            row["fill8_teacher"] = {**score(render(xt, coords)[0], img, covered),
                                    "seconds": secs_t}
        if layout not in seen:
            seen.add(layout)
            pics.append((layout, [img, floor[layout], *fills]))
        rows.append(row)
        print(f"  {layout:6s} {seed} n={row['nodes']:3d} "
              f"floor mse {row['floor']['mse']:.4f} r {row['floor']['r']:.3f} · "
              f"fill8 mse {row['fill8']['mse']:.4f} r {row['fill8']['r']:.3f} "
              f"{row['fill8']['seconds']:.2f}s · "
              f"fill16 mse {row['fill16']['mse']:.4f} r {row['fill16']['r']:.3f} "
              f"{row['fill16']['seconds']:.2f}s", flush=True)

    def agg(layout: str, key: str) -> dict:
        xs = rows if layout == "all" else [r for r in rows if r["layout"] == layout]
        if not xs:
            return {}
        keys = ("mse", "r", "seconds") if key.startswith("fill") else ("mse", "r", "covered")
        return {k: sum(x[key][k] for x in xs) / len(xs) for k in keys}

    summary = {}
    for layout in (*CKPTS, "all"):
        summary[layout] = {"floor": agg(layout, "floor"), "ceiling": agg(layout, "ceiling")}
        for passes in PASSES:
            summary[layout][f"fill{passes}"] = agg(layout, f"fill{passes}")
        if not summary[layout]["floor"]:
            summary[layout]["nodes"] = 0
            continue
        summary[layout]["nodes"] = (sum(r["nodes"] for r in rows
                                        if layout in ("all", r["layout"]))
                                    / len([r for r in rows
                                           if layout in ("all", r["layout"])]))

    res = {"eval_sha256": ev["sha256"], "items": len(rows),
           "checkpoints": ckpts, "params": models["hex"].n_params(),
           "passes": list(PASSES), "summary": summary, "rows": rows}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    print()
    print(f"{'layout':6s} {'nodes':>5s} {'floor':>7s} {'ceiling':>7s} "
          f"{'fill8 mse':>9s} {'r':>6s} {'s':>5s} {'gain':>5s} "
          f"{'fill16 mse':>10s} {'r':>6s} {'s':>5s}")
    for layout in (*CKPTS, "all"):
        s = summary[layout]
        if not s["floor"]:
            continue
        n = s["nodes"]
        gain = ((s["floor"]["mse"] - s["fill8"]["mse"])
                / (s["floor"]["mse"] - s["ceiling"]["mse"]))
        print(f"{layout:6s} {n:5.0f} {s['floor']['mse']:7.4f} {s['ceiling']['mse']:7.4f} "
              f"{s['fill8']['mse']:9.4f} {s['fill8']['r']:6.3f} "
              f"{s['fill8']['seconds']:5.2f} {gain:5.0%} {s['fill16']['mse']:10.4f} "
              f"{s['fill16']['r']:6.3f} {s['fill16']['seconds']:5.2f}")

    if a.teacher:
        print()
        print("diagnostic: 8 passes with the true patch revealed after each one")
        for layout in (*CKPTS, "all"):
            xs = [r for r in rows if layout == "all" or r["layout"] == layout]
            if not xs:
                continue
            t = sum(r["fill8_teacher"]["mse"] for r in xs) / len(xs)
            f = sum(r["floor"]["mse"] for r in xs) / len(xs)
            c = sum(r["ceiling"]["mse"] for r in xs) / len(xs)
            print(f"{layout:6s} teacher mse {t:.4f} r "
                  f"{sum(r['fill8_teacher']['r'] for r in xs) / len(xs):.3f} "
                  f"gain {(f - t) / (f - c):5.0%}")

    if a.sheet:
        print("sheet    " + sheet(pics, a.sheet))

    print(f"covered  {summary['all']['floor']['covered']:.1%} of the canvas")
    g = summary["all"]
    for passes in PASSES:
        k = f"fill{passes}"
        print(f"gate {k}: mse {g[k]['mse'] / g['floor']['mse']:.2f}x floor "
              f"(need <=0.50x) · {g[k]['seconds']:.2f}s (need <=2s) · "
              f"gain {(g['floor']['mse'] - g[k]['mse']) / (g['floor']['mse'] - g['ceiling']['mse']):.0%} "
              f"of what the representation allows")


if __name__ == "__main__":
    main()
