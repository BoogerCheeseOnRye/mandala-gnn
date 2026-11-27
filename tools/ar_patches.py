"""Step 6, candidate B: an autoregressive patch-code decoder.

The GNN fills a canvas in passes; this candidate draws left to right. Each
drawing is a sequence of 4x4 patch codes (k-means over training patches, 128
codes, the codebook R0 built), visited in scanline order, and a small causal
transformer predicts the next code given the previous ones and the *known*
coordinates of every node. Positions are the absolute (x, y) of each patch
(Fourier features), so generation is conditioned on geometry the same way a
GNN is -- only the topology is missing.

Matched to the GNN: dim 96, depth 3, heads 4.

    .venv/bin/python tools/ar_patches.py --out runs/ar-patches

Scores the same 36 frozen drawings as tools/draw_gen.py, with the same
rendering and the same constant-image floor, so the two candidates are
directly comparable.
"""
from __future__ import annotations

import argparse
import json
import math
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.seq import SeqLM  # noqa: E402
from mandalanet.synth import LAYOUTS, sample  # noqa: E402
from tools.draw_gen import floor_images, render, score, sheet  # noqa: E402
from tools.make_eval import load as load_eval  # noqa: E402
from tools.seq_experiment import load_data  # noqa: E402

BOS = 128


def train(payload: dict, args) -> tuple[SeqLM, dict]:
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed + 17)
    model = SeqLM(vocab=BOS + 1, dim=args.dim, depth=args.depth, heads=args.heads,
                  pos="abs2d", dropout=args.dropout, max_len=256)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    data = payload["data"]
    steps = args.epochs * args.steps_per_epoch
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for _ in range(args.steps_per_epoch):
            lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
                args.lr * 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup)
                                                / max(1, steps - args.warmup)))
            for gp in opt.param_groups:
                gp["lr"] = lr
            layout = LAYOUTS[int(torch.randint(0, len(LAYOUTS), (1,), generator=g))]
            d = data[layout]
            n = d["n"]
            b = max(1, round(args.tokens_per_step / n))
            idx = torch.randint(0, d["t_train"].shape[0], (b,), generator=g)
            codes = d["t_train"][idx]                       # (b, n)
            coords = d["c_train"][idx]
            seq = torch.cat([torch.full((codes.shape[0], 1), BOS), codes], 1)
            pos = torch.cat([coords[:, :1], coords], 1)
            out = model(seq[:, :-1], pos[:, :-1])
            loss = F.cross_entropy(out.reshape(-1, BOS + 1), seq[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        bits, acc = evaluate(model, payload)
        print(f"  epoch {epoch + 1}/{args.epochs} {bits:.3f} bits/patch-code "
              f"acc {acc:.1%} · {time.time() - t0:.0f}s", flush=True)
    return model, {"bits": bits, "acc": acc, "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def evaluate(model: SeqLM, payload: dict) -> tuple[float, float]:
    model.eval()
    ce = correct = count = 0
    for layout in LAYOUTS:
        d = payload["data"][layout]
        codes, coords = d["t_val"], d["c_val"]
        seq = torch.cat([torch.full((codes.shape[0], 1), BOS), codes], 1)
        pos = torch.cat([coords[:, :1], coords], 1)
        logits = model(seq[:, :-1], pos[:, :-1])
        flat = logits.reshape(-1, BOS + 1)
        target = seq[:, 1:].reshape(-1)
        ce += float(F.cross_entropy(flat, target, reduction="sum"))
        correct += int((flat.argmax(-1) == target).sum())
        count += target.numel()
    return ce / (count * math.log(2)), correct / count


@torch.no_grad()
def generate(model: SeqLM, coords: torch.Tensor, n: int, temp: float | None,
             seed: int = 0) -> torch.Tensor:
    """Codes for one drawing, conditioned on its node coordinates."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.full((1, 1), BOS, dtype=torch.long)
    logits, cache = model.forward_cached(idx, coords[:, :1])
    out = []
    for t in range(n):
        row = logits[0, -1]
        if temp is None:
            nxt = int(row.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(row / temp, -1), 1, generator=g))
        out.append(nxt)
        if t + 1 < n:
            logits, cache = model.forward_cached(
                torch.tensor([[nxt]]), coords[:, t + 1:t + 2], cache)
    return torch.tensor(out)


def leak_check(payload: dict, ev: dict) -> dict:
    """Is any frozen eval drawing in the training set?

    The eval drawings come from literal seeds (1000-1011); these training
    drawings come from a generator stream, so a collision is not expected --
    but "not expected" is not a measurement. For each eval drawing, the
    distance to the nearest training drawing, by patch values and by node
    coordinates, both as the largest per-node difference.
    """
    out = {}
    for layout, d in payload["data"].items():
        tr, ct = d["p_train"], d["c_train"]
        by_patch, by_coord = [], []
        for item in [i for i in ev["drawings"]["items"] if i["layout"] == layout]:
            p, c, _ = sample(layout, torch.Generator().manual_seed(item["seed"]))
            by_patch.append(float((tr - p[None]).abs().amax(dim=(1, 2)).min()))
            by_coord.append(float((ct - c[None]).abs().amax(dim=(1, 2)).min()))
        out[layout] = {"nearest_patch": min(by_patch),
                       "nearest_coord": min(by_coord),
                       "train_drawings": int(tr.shape[0])}
        print(f"  leak {layout:6s} nearest training drawing: "
              f"patches {out[layout]['nearest_patch']:.3f} "
              f"coords {out[layout]['nearest_coord']:.3f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/seq/data.pt")
    p.add_argument("--eval", dest="eval_path", default="runs/budget/eval.json")
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=120)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--tokens-per-step", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="runs/ar-patches")
    p.add_argument("--sheet", default="", help="write a PNG of target | floor | AR")
    p.add_argument("--resume", action="store_true",
                   help="score runs/<out>/model.pt instead of training")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)

    payload = load_data(a.data)
    ev = load_eval(a.eval_path)
    if a.resume:
        ck = torch.load(os.path.join(a.out, "model.pt"), map_location="cpu",
                        weights_only=False)
        c = ck["config"]
        model = SeqLM(vocab=c["vocab"], dim=c["dim"], depth=c["depth"],
                      heads=c["heads"], pos=c["pos"], dropout=c["dropout"])
        model.load_state_dict(ck["state_dict"])
        train_metrics = json.load(open(os.path.join(a.out, "results.json")))["train"] \
            if os.path.exists(os.path.join(a.out, "results.json")) else {}
        print(f"resumed {a.out}: {model.n_params():,} params", flush=True)
        score_all(model, payload, ev, a, train_metrics,
                  os.path.join(a.out, "results.json"))
        return
    model, train_metrics = train(payload, a)
    torch.save({"state_dict": model.state_dict(),
                "config": {"vocab": BOS + 1, "dim": a.dim, "depth": a.depth,
                           "heads": a.heads, "pos": "abs2d",
                           "dropout": a.dropout}},
               os.path.join(a.out, "model.pt"))
    print(f"{model.n_params():,} params · codebook quant MSE "
          f"{payload['quant_mse']:.5f}", flush=True)

    score_all(model, payload, ev, a, train_metrics, os.path.join(a.out, "results.json"))


def score_all(model: SeqLM, payload: dict, ev: dict, a, train_metrics: dict,
              out_json: str) -> dict:
    """Same drawings, renderer and controls as tools/draw_gen.py."""
    book = payload["book"]
    floor = floor_images()
    items = ev["drawings"]["items"][:a.limit or None]
    rows = []
    pics = []
    seen: set[str] = set()
    model.eval()
    # step 4's lesson, again: one token at a time through oneDNN costs 400-1400
    # us per matmul on this device, so decode with oneDNN off and one thread
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    for item in items:
        layout, seed = item["layout"], item["seed"]
        patches, coords, img = sample(layout, torch.Generator().manual_seed(seed))
        _, covered = render(patches, coords)
        t0 = time.time()
        codes = generate(model, coords[None], patches.shape[0], a.temp, seed)
        secs = time.time() - t0
        out, _ = render(book[codes], coords)
        row = {"layout": layout, "seed": seed, "nodes": int(patches.shape[0]),
               "floor": score(floor[layout], img, covered),
               "ceiling": score(render(patches, coords)[0], img, covered),
               "ar": {**score(out, img, covered), "seconds": secs}}
        rows.append(row)
        if layout not in seen:
            seen.add(layout)
            pics.append((layout, [img, floor[layout], out]))
        print(f"  {layout:6s} {seed} mse {row['ar']['mse']:.4f} "
              f"r {row['ar']['r']:.3f} {secs:.2f}s (floor {row['floor']['mse']:.4f}, "
              f"ceiling {row['ceiling']['mse']:.4f})", flush=True)

    def agg(layout: str) -> dict:
        xs = rows if layout == "all" else [r for r in rows if r["layout"] == layout]
        return {"ar": {k: sum(x["ar"][k] for x in xs) / len(xs)
                       for k in ("mse", "r", "seconds")},
                "floor": {k: sum(x["floor"][k] for x in xs) / len(xs)
                          for k in ("mse", "r", "covered")},
                "ceiling": {k: sum(x["ceiling"][k] for x in xs) / len(xs)
                            for k in ("mse", "r", "covered")},
                "nodes": sum(x["nodes"] for x in xs) / len(xs)}

    summary = {layout: agg(layout) for layout in (*LAYOUTS, "all")}
    res = {"eval_sha256": ev["sha256"], "items": len(rows),
           "params": model.n_params(), "temp": a.temp,
           "leak": leak_check(payload, ev),
           "train": train_metrics, "summary": summary, "rows": rows}
    if getattr(a, "sheet", ""):
        res["sheet"] = sheet(pics, a.sheet,
                             labels=("drawing", "floor", "AR patch codes"))
        print("sheet    " + res["sheet"])
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    print()
    print(f"{'layout':6s} {'floor':>7s} {'ceiling':>7s} {'AR mse':>7s} "
          f"{'AR r':>6s} {'AR s':>5s} {'gain':>5s}")
    for layout in (*LAYOUTS, "all"):
        s = summary[layout]
        gain = ((s["floor"]["mse"] - s["ar"]["mse"])
                / (s["floor"]["mse"] - s["ceiling"]["mse"]))
        print(f"{layout:6s} {s['floor']['mse']:7.4f} {s['ceiling']['mse']:7.4f} "
              f"{s['ar']['mse']:7.4f} {s['ar']['r']:6.3f} {s['ar']['seconds']:5.2f} "
              f"{gain:5.0%}")
    g = summary["all"]
    print(f"gate: mse {g['ar']['mse'] / g['floor']['mse']:.2f}x floor "
          f"(need <=0.50x) · {g['ar']['seconds']:.2f}s (need <=2s) · "
          f"gain {(g['floor']['mse'] - g['ar']['mse']) / (g['floor']['mse'] - g['ceiling']['mse']):.0%} "
          f"of what the representation allows")
    return res


if __name__ == "__main__":
    main()
