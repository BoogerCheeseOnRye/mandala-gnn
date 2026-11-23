"""R2b: is the drawing a good place to compute? (balanced binary probe)

    .venv/bin/python tools/reach_task.py --layouts chart rings hex \
        --substrates drawing random chain star --train-steps 600 --seeds 0

The first R2 task (regress the BFS distance to every node) was not a fair
comparison: the drawings' diameter is 6, so the answer sat in a handful of
classes that a star or a random graph produced in a different mix, and balanced
accuracy mostly measured class priors.

This probe asks one binary question instead:

    is the target node within k hops of the source node?

`k` is chosen *per graph* as the value that makes the positive rate closest to
50%, and the model is given exactly `k` propagation steps, so the question is
answerable within that graph's own reach. Pairs are sampled half positive, half
negative, so a model cannot win by guessing the majority class.

The only thing that differs between substrates is the edge set: the drawing, a
*connected* random graph with exactly the same node and edge counts, a chain and
a star. Weights, parameter count, steps and batch are identical. Metrics are
accuracy on balanced pairs and AUC over every held-out pair (AUC is independent
of the class mix, which is the thing the distance task got wrong).

Held-out sources: 20% of source nodes are never trained on, so the model cannot
answer by memorising a particular start node.
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

from mandalanet.compute import GraphReasoner, distances  # noqa: E402
from tools.energy_experiment import (BG, BLUE, DIM, GRID, INK, ORANGE,  # noqa: E402
                                     _canvas, _font, GREEN)
from tools.topology_task import edges_for  # noqa: E402

LAYOUTS = ("chart", "rings", "hex")
SUBSTRATES = ("drawing", "random", "rewired", "chain", "star")
COLORS = {"drawing": BLUE, "random": ORANGE, "rewired": (230, 190, 90),
          "chain": GREEN, "star": (200, 120, 200)}
LABELS = {"drawing": "the drawing", "random": "random (matched)",
          "rewired": "rewired (same degrees)", "chain": "chain", "star": "star"}


# --------------------------------------------------------------------------- #
# task construction
# --------------------------------------------------------------------------- #
def choose_k(dist: torch.Tensor, k_max: int = 16) -> tuple[int, float]:
    """The hop budget whose positive rate is closest to 50%."""
    n = dist.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    vals = dist[off].float()
    best_k, best_rate = 1, -1.0
    for k in range(1, k_max + 1):
        rate = float((vals <= k).float().mean())
        if best_rate < 0 or abs(rate - 0.5) < abs(best_rate - 0.5):
            best_k, best_rate = k, rate
    return best_k, best_rate


def pair_sets(dist: torch.Tensor, k: int, sources: torch.Tensor):
    """All (source, target) pairs for the given sources, split by label."""
    n = dist.shape[0]
    pos, neg = [], []
    for s in sources.tolist():
        for t in range(n):
            if t == s:
                continue
            (pos if int(dist[s, t]) <= k else neg).append((s, t))
    if not pos or not neg:
        raise ValueError(f"degenerate k={k}: {len(pos)} positive, {len(neg)} negative")
    return (torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long))


def batch_of(pos: torch.Tensor, neg: torch.Tensor, b: int, g: torch.Generator):
    hp = b // 2
    p = pos[torch.randint(0, pos.shape[0], (hp,), generator=g)]
    q = neg[torch.randint(0, neg.shape[0], (b - hp,), generator=g)]
    pairs = torch.cat([p, q])
    y = torch.cat([torch.ones(hp), torch.zeros(b - hp)])
    return pairs, y


def encode(pairs: torch.Tensor, n: int) -> torch.Tensor:
    """(B, N, 2): one channel says 'this is the source', one 'this is the target'."""
    x = torch.zeros(pairs.shape[0], n, 2)
    rows = torch.arange(pairs.shape[0])
    x[rows, pairs[:, 0], 0] = 1.0
    x[rows, pairs[:, 1], 1] = 1.0
    return x


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank AUC with ties averaged, so a constant score is exactly 0.5."""
    npos = float(labels.sum())
    nneg = labels.numel() - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = torch.argsort(scores)
    s = scores[order]
    n = s.numel()
    _, inv = torch.unique(s, return_inverse=True)
    ranks = torch.arange(1, n + 1, dtype=torch.float32)
    sums = torch.zeros(int(inv.max()) + 1).index_add_(0, inv, ranks)
    counts = torch.zeros_like(sums).index_add_(0, inv, torch.ones(n))
    avg = (sums / counts)[inv]
    r = torch.empty(n, dtype=torch.float32)
    r[order] = avg
    return float((r[labels > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def evaluate(model, edges, n, pairs, y, chunk: int = 256) -> dict:
    model.eval()
    scores = []
    for i in range(0, pairs.shape[0], chunk):
        p = pairs[i:i + chunk]
        logits = model(encode(p, n), edges).squeeze(-1)
        scores.append(logits[torch.arange(p.shape[0]), p[:, 1]])
    s = torch.cat(scores)
    acc = float(((s > 0) == (y > 0.5)).float().mean())
    return {"acc": acc, "auc": auc(s, y), "logit_gap":
            float(s[y > 0.5].mean() - s[y <= 0.5].mean())}


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def run(layout: str, substrate: str, seed: int, args) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 11)
    n, edges, m = edges_for(layout, substrate, seed)
    dist = distances(n, edges, n)
    k, rate = choose_k(dist, args.k_max)

    perm = torch.randperm(n, generator=g)
    n_val = max(4, n // 5)
    val_src, train_src = perm[:n_val], perm[n_val:]
    pos, neg = pair_sets(dist, k, train_src)
    vpos, vneg = pair_sets(dist, k, val_src)
    # a fixed balanced held-out set, plus every held-out pair for AUC
    ev_pos = vpos[torch.randint(0, vpos.shape[0], (min(512, vpos.shape[0]),), generator=g)]
    ev_neg = vneg[torch.randint(0, vneg.shape[0], (min(512, vneg.shape[0]),), generator=g)]
    ev_pairs = torch.cat([ev_pos, ev_neg])
    ev_y = torch.cat([torch.ones(ev_pos.shape[0]), torch.zeros(ev_neg.shape[0])])
    all_pairs = torch.cat([vpos, vneg])
    all_y = torch.cat([torch.ones(vpos.shape[0]), torch.zeros(vneg.shape[0])])

    model = GraphReasoner(dim=args.dim, classes=1, steps=k, in_dim=2, out_dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    curve = []
    for step in range(args.train_steps):
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup) /
                                            max(1, args.train_steps - args.warmup)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        pairs, y = batch_of(pos, neg, args.batch, g)
        logits = model(encode(pairs, n), edges).squeeze(-1)
        logits = logits[torch.arange(pairs.shape[0]), pairs[:, 1]]
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.train_steps // 3) == 0:
            ev = evaluate(model, edges, n, ev_pairs, ev_y)
            ev["all"] = evaluate(model, edges, n, all_pairs, all_y)
            curve.append({"step": step + 1, "loss": float(loss.detach()), **ev})
            print(f"  {layout:6s} {substrate:8s} k={k:2d} {step + 1:4d} "
                  f"bal {ev['acc']:.3f} auc {ev['auc']:.3f} "
                  f"all-acc {ev['all']['acc']:.3f} · {time.time() - t0:.0f}s", flush=True)
    return {"layout": layout, "substrate": substrate, "seed": seed,
            "nodes": n, "edges": m, "k": k, "pos_rate": round(rate, 3),
            "steps": k, "diameter": int(dist.max()),
            "params": sum(p.numel() for p in model.parameters()),
            "metrics": {kk: vv for kk, vv in curve[-1].items() if kk != "all"},
            "all": curve[-1]["all"], "curve": curve,
            "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(path: str, results: list[dict]) -> str:
    im, d = _canvas(1180, 560, "R2: is the drawing a better place to compute?",
                    "\"is the target within k hops of the source?\" · k set per graph "
                    "for a 50/50 question · AUC on held-out sources")
    l, r, t, b = 96, 700, 140, 430
    subs = [s for s in SUBSTRATES if any(x["substrate"] == s for x in results)]
    lo, hi = 0.3, 1.0

    def py(v):
        return b - (v - lo) / (hi - lo) * (b - t)

    for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        yy = py(v)
        d.line([l, yy, r, yy], fill=GRID, width=1)
        d.text((l - 10, yy), f"{v:.1f}", fill=(150, 158, 172), anchor="rm", font=_font(12))
    d.line([l, py(0.5), r, py(0.5)], fill=(120, 126, 140), width=1)
    d.text((r + 8, py(0.5)), "chance", fill=(150, 158, 172), anchor="lm", font=_font(11))

    group = (r - l) / len(LAYOUTS)
    for gi, layout in enumerate(LAYOUTS):
        gx = l + gi * group
        rows = [x for x in results if x["layout"] == layout]
        d.text((gx + group / 2, b + 18), layout, fill=(190, 196, 208),
               anchor="ma", font=_font(13))
        if rows:
            draw = [x for x in rows if x["substrate"] == "drawing"]
            d.text((gx + group / 2, b + 34),
                   f"{draw[0]['nodes']} nodes · {draw[0]['edges']} edges", fill=DIM,
                   anchor="ma", font=_font(11))
        for ai, sub in enumerate(subs):
            vals = [x["metrics"]["auc"] for x in rows if x["substrate"] == sub]
            if not vals:
                continue
            cx = gx + (ai + 0.55) * group / len(subs)
            w = group / len(subs) * 0.62
            m = sum(vals) / len(vals)
            d.rectangle([cx - w / 2, py(m), cx + w / 2, b], fill=COLORS[sub])
            for v in vals:
                d.ellipse([cx - 3, py(v) - 3, cx + 3, py(v) + 3], fill=INK)
            k = [x["k"] for x in rows if x["substrate"] == sub][0]
            d.text((cx, b - 9), str(k), fill=BG, anchor="ms", font=_font(10))
    d.text((l, t - 24), "AUC, held-out sources · the number in each bar is its hop budget k",
           fill=(190, 196, 208), font=_font(13))

    lx = 790
    d.text((lx + 250, 128), "mean auc", fill=DIM, anchor="rm", font=_font(12))
    for i, sub in enumerate(subs):
        yy = 160 + i * 30
        d.rectangle([lx, yy - 6, lx + 12, yy + 6], fill=COLORS[sub])
        d.text((lx + 22, yy), LABELS[sub], fill=INK, anchor="lm", font=_font(13))
        vals = [x["metrics"]["auc"] for x in results if x["substrate"] == sub]
        m = sum(vals) / len(vals) if vals else float("nan")
        d.text((lx + 250, yy), f"{m:.3f}", fill=INK, anchor="rm", font=_font(13))

    d.text((96, 500), "k is the hop budget closest to a 50/50 question for that graph, and the model "
           "gets exactly k steps, so no substrate is asked", fill=DIM, font=_font(12))
    d.text((96, 520), "something its own wiring cannot answer. Weights and batch are matched; chain and "
           "star have n-1 edges by construction.", fill=DIM, font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    p.add_argument("--substrates", nargs="+", default=list(SUBSTRATES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--batch", type=int, default=32)
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
                with open(os.path.join(a.out, "reach.json"), "w") as f:
                    json.dump({"config": vars(a), "results": results}, f, indent=2)
    if len({(x["layout"], x["substrate"]) for x in results}) > 1:
        figure(os.path.join(a.out, "reach-task.png"), results)
    print(f"{'layout':7s} {'graph':8s} {'k':>3s} {'edges':>6s} {'diam':>5s} "
          f"{'auc':>6s} {'bal':>6s} {'all':>6s}")
    for x in results:
        print(f"{x['layout']:7s} {x['substrate']:8s} {x['k']:3d} {x['edges']:6d} "
              f"{x['diameter']:5d} {x['metrics']['auc']:6.3f} "
              f"{x['metrics']['acc']:6.3f} {x['all']['acc']:6.3f}")


if __name__ == "__main__":
    main()
