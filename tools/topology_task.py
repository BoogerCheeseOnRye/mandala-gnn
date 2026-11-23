"""R2: is the drawing a good place to compute?

    .venv/bin/python tools/topology_task.py --layouts chart rings hex \
        --substrates drawing random chain star --train-steps 600 --seeds 0

The task is BFS distance: pick a source node, ask for the hop distance to every
other node, capped at T. A recurrent graph net keeps a state on the nodes and
passes messages along a fixed edge set for T steps, so a graph that needs long
propagation needs more iterations, and the edge set is the only routing.

The same weights and parameter count are used for every substrate. Only the
graph changes: the drawing, a random graph with the same node and edge count, a
chain and a star. If the drawing is a better computation graph than a random
graph with the same budget, it should show up as higher balanced accuracy at
the same number of edges.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.compute import (GraphReasoner, chain_edges, distances,  # noqa: E402
                                full_edges, random_edges, rewired_edges,
                                star_edges)
from mandalanet.graph import dedupe_edges, make_undirected  # noqa: E402
from mandalanet.lattices import build_mesh  # noqa: E402
from tools.energy_experiment import (BLUE, DIM, GRID, INK, ORANGE,  # noqa: E402
                                     _canvas, _font, GREEN)

LAYOUTS = ("chart", "rings", "hex")
SUBSTRATES = ("drawing", "random", "rewired", "chain", "star", "full")
COLORS = {"drawing": BLUE, "random": ORANGE, "chain": GREEN, "star": (200, 120, 200),
          "rewired": (230, 190, 90), "full": (150, 158, 172)}
MARKERS = {"chart": "hex", "rings": "rings", "hex": "hex"}


def edges_for(layout: str, substrate: str, seed: int):
    mesh = build_mesh(layout)
    n = int(mesh.pos_px.shape[0])
    drawn = dedupe_edges(make_undirected(mesh.edge_index), n)[0]
    m = int(drawn.shape[1]) // 2
    if substrate == "drawing":
        return n, drawn, m
    if substrate == "random":
        return n, random_edges(n, m, seed), m
    if substrate == "rewired":
        return n, rewired_edges(n, drawn, seed), m
    if substrate == "chain":
        return n, chain_edges(n), n - 1
    if substrate == "star":
        return n, star_edges(n), n - 1
    if substrate == "full":
        return n, full_edges(n), n * (n - 1) // 2
    raise KeyError(substrate)


def balanced_accuracy(pred: torch.Tensor, true: torch.Tensor) -> float:
    recalls = []
    for c in true.unique():
        m = true == c
        recalls.append(float((pred[m] == c).float().mean()))
    return sum(recalls) / len(recalls)


@torch.no_grad()
def evaluate(model, edges, dist, sources, batch=32):
    model.eval()
    preds = []
    for i in range(0, sources.numel(), batch):
        s = sources[i:i + batch]
        src = torch.zeros(s.numel(), dist.shape[0])
        src[torch.arange(s.numel()), s] = 1.0
        preds.append(model(src, edges).argmax(-1))
    pred = torch.cat(preds)
    true = dist[sources]
    return {"acc": float((pred == true).float().mean()),
            "within1": float(((pred - true).abs() <= 1).float().mean()),
            "bal_acc": balanced_accuracy(pred.reshape(-1), true.reshape(-1)),
            "mae": float((pred - true).abs().float().mean())}


def run(layout: str, substrate: str, seed: int, args) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 7)
    n, edges, m = edges_for(layout, substrate, seed)
    dist = distances(n, edges, args.steps)
    perm = torch.randperm(n, generator=g)
    n_val = max(4, n // 5)
    val, train = perm[:n_val], perm[n_val:]
    classes = args.steps + 1

    model = GraphReasoner(dim=args.dim, classes=classes, steps=args.steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    curve = []
    for step in range(args.train_steps):
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup) /
                                            max(1, args.train_steps - args.warmup)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        idx = train[torch.randint(0, train.numel(), (args.batch,), generator=g)]
        src = torch.zeros(args.batch, n)
        src[torch.arange(args.batch), idx] = 1.0
        logits = model(src, edges)
        loss = F.cross_entropy(logits.reshape(-1, classes), dist[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.train_steps // 3) == 0:
            ev = evaluate(model, edges, dist, val)
            curve.append({"step": step + 1, **ev})
            print(f"  {layout:6s} {substrate:8s} seed {seed} {step + 1:4d} "
                  f"bal {ev['bal_acc']:.3f} acc {ev['acc']:.3f} "
                  f"±1 {ev['within1']:.3f} · {time.time() - t0:.0f}s", flush=True)
    return {"layout": layout, "substrate": substrate, "seed": seed, "nodes": n,
            "edges": m, "steps": args.steps, "diameter": int(dist.max()),
            "params": sum(p.numel() for p in model.parameters()),
            "metrics": curve[-1], "curve": curve,
            "seconds": round(time.time() - t0, 1)}


def figure(path: str, results: list[dict]) -> str:
    im, d = _canvas(1180, 560, "R2: is the drawing a good place to compute?",
                    "BFS distance on a fixed edge set · same weights for every graph · 12 propagation steps")
    l, r, t, b = 100, 720, 140, 440
    d.rectangle([l, t, r, b], outline=(60, 66, 78), width=1)
    xs = [x["edges"] for x in results]
    ys = [x["metrics"]["bal_acc"] for x in results]
    x0, x1 = min(xs) * 0.8, max(xs) * 1.25
    y0, y1 = 0.0, 1.0

    def px(v):
        return l + (math.log(max(v, 1)) - math.log(x0)) / (math.log(x1) - math.log(x0)) * (r - l)

    def py(v):
        return b - (v - y0) / (y1 - y0) * (b - t)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = b - frac * (b - t)
        d.line([l, yy, r, yy], fill=GRID, width=1)
        d.text((l - 10, yy), f"{frac:.2f}", fill=(150, 158, 172), anchor="rm", font=_font(12))
    for v in (50, 100, 200, 500, 1000, 2000):
        if x0 <= v <= x1:
            xx = px(v)
            d.line([xx, t, xx, b], fill=GRID, width=1)
            d.text((xx, b + 16), str(v), fill=(150, 158, 172), anchor="ma", font=_font(12))
    d.text((r, b + 44), "undirected edges (log)", fill=DIM, anchor="ra", font=_font(12))
    d.text((l, t - 22), "balanced accuracy after 12 steps", fill=(190, 196, 208), font=_font(13))

    for x in results:
        cx, cy = px(x["edges"]), py(x["metrics"]["bal_acc"])
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=COLORS[x["substrate"]])
        d.text((cx + 10, cy), f"{x['layout']}", fill=(200, 205, 215), anchor="lm",
               font=_font(11))

    lx = 800
    for i, sub in enumerate([s for s in SUBSTRATES if any(x["substrate"] == s for x in results)]):
        yy = 160 + i * 28
        d.rectangle([lx, yy - 6, lx + 12, yy + 6], fill=COLORS[sub])
        d.text((lx + 22, yy), sub, fill=INK, anchor="lm", font=_font(13))
    d.text((96, 510), "Fewer edges and higher is better. The random graph has the drawing's node and edge "
           "count, so the gap is topology, not budget.", fill=DIM, font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    p.add_argument("--substrates", nargs="+", default=["drawing", "random", "chain", "star"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-steps", type=int, default=600)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="runs/topology")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    results = []
    for layout in a.layouts:
        for sub in a.substrates:
            for seed in a.seeds:
                print(f"[{layout} · {sub} · seed {seed}]", flush=True)
                results.append(run(layout, sub, seed, a))
                with open(os.path.join(a.out, "results.json"), "w") as f:
                    json.dump({"config": vars(a), "results": results}, f, indent=2)
    if len({(x["layout"], x["substrate"]) for x in results}) > 1:
        figure(os.path.join(a.out, "topology-task.png"), results)
    print(f"{'layout':7s} {'graph':8s} {'edges':>6s} {'bal':>6s} {'acc':>6s} {'±1':>6s}")
    for x in results:
        mm = x["metrics"]
        print(f"{x['layout']:7s} {x['substrate']:8s} {x['edges']:6d} "
              f"{mm['bal_acc']:6.3f} {mm['acc']:6.3f} {mm['within1']:6.3f}")


if __name__ == "__main__":
    main()
