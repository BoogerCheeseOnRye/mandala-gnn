"""R1: does the pair-bias mechanism transfer to code?

    .venv/bin/python tools/code_experiment.py --arms rope abs bias_dist bias_struct bias_shuf \
        --seeds 0 1 --steps 1200 --out runs/code

Corpus: a pinned 2 MB subset of CPython 3.12.0's `Lib/`, byte-level, split by
*file* so held-out bytes come from files the model never saw. Structural
features (indentation, bracket depth, line index) are computed per byte.

Arms differ only in how attention learns about position:

    rope          rotary position on q/k
    abs           learned absolute position embedding
    bias_dist     learned bias from RBF(|i-j|), no RoPE and no abs
    bias_struct   learned bias from RBF(|i-j|) + indentation, bracket depth,
                  line distance, same-line flag; no RoPE and no abs
    bias_shuf     the same features with the structural channels permuted
                  across positions: a capacity control

The question is narrow on purpose. R0 killed the drawing as a positional code;
this asks whether the *mechanism* that survived it is a code prior at all.
Metric is bits/byte on held-out files, not loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.code import CodeLM, bits_per_byte, pair_features  # noqa: E402
from tools.energy_experiment import (BG, BLUE, DIM, GRID, INK, ORANGE,  # noqa: E402
                                     _canvas, _font, GREEN)

ARMS = ("rope", "abs", "bias_dist", "bias_struct", "bias_shuf")
COLORS = {"rope": BLUE, "abs": (120, 140, 200), "bias_dist": GREEN,
          "bias_struct": ORANGE, "bias_shuf": (200, 120, 200)}
LABELS = {"rope": "RoPE", "abs": "absolute", "bias_dist": "bias: distance only",
          "bias_struct": "bias: distance + structure",
          "bias_shuf": "bias: shuffled (control)"}
XLABELS = {"rope": "RoPE", "abs": "absolute", "bias_dist": "distance\nonly",
           "bias_struct": "distance\n+ structure", "bias_shuf": "shuffled\n(control)"}

SKIP = ("/test/", "/tests/", "idle_test", "/test_")
TARGET = 2_000_000


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def _line_features(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.frombuffer(data, dtype=np.uint8)
    n = b.size
    newline = np.flatnonzero(b == 10)
    starts = np.concatenate([[0], newline + 1])
    starts = starts[starts < n]
    line_of = np.zeros(n, dtype=np.int32)
    line_of[starts] = 1
    line_of = np.cumsum(line_of) - 1
    # indent: leading spaces of each line, in units of 4, clipped to 0..8
    indent_per_line = np.zeros(starts.size, dtype=np.uint8)
    for i, s in enumerate(starts):
        j = s
        while j < n and b[j] == 32 and j - s < 40:
            j += 1
        indent_per_line[i] = min((j - s) // 4, 8)
    indent = indent_per_line[line_of]
    # bracket depth: crude running balance, strings and comments ignored
    delta = np.zeros(n, dtype=np.int32)
    for ch, dv in ((40, 1), (91, 1), (123, 1), (41, -1), (93, -1), (125, -1)):
        delta[b == ch] = dv
    depth = np.clip(np.cumsum(delta), 0, 15).astype(np.uint8)
    return indent, depth, line_of


def build_corpus(src: str, target: int = TARGET) -> dict:
    files = []
    for root, _, names in os.walk(src):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src)
            if any(s in "/" + rel for s in SKIP):
                continue
            size = os.path.getsize(path)
            if size < 2000 or size > 80000:
                continue
            files.append((rel, path, size))
    files.sort()
    if not files:
        raise SystemExit(f"no .py files under {src}")
    avg = sum(f[2] for f in files) / len(files)
    stride = max(1, round(len(files) * avg / (target * 1.25)))
    picked, total = [], 0
    for i in range(0, len(files), stride):
        picked.append(files[i])
        total += files[i][2]
        if total >= target:
            break
    picked.sort()

    split = {"train": ([], [], [], [], []), "val": ([], [], [], [], [])}
    manifest = []
    for i, (rel, path, size) in enumerate(picked):
        which = "val" if i % 10 == 9 else "train"
        with open(path, "rb") as fh:
            data = fh.read()
        indent, depth, line = _line_features(data)
        split[which][0].append(np.frombuffer(data, dtype=np.uint8))
        split[which][1].append(indent)
        split[which][2].append(depth)
        split[which][3].append(line)
        manifest.append({"path": rel, "bytes": size, "split": which})

    out = {"files": manifest}
    for which, parts in split.items():
        for key, arrs in zip(("bytes", "indent", "bracket", "line"), parts):
            joined = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.uint8)
            dtype = torch.int32 if key == "line" else torch.uint8
            out[f"{which}_{key}"] = torch.from_numpy(joined.astype(np.int64 if key == "line" else np.uint8))
        del dtype
    out["sha256"] = hashlib.sha256(
        out["train_bytes"].numpy().tobytes() + out["val_bytes"].numpy().tobytes()
    ).hexdigest()
    return out


def load_corpus(path: str, src: str, rebuild: bool = False) -> dict:
    if rebuild and os.path.exists(path):
        os.remove(path)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    print(f"building {path} from {src} ...", flush=True)
    payload = build_corpus(src)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"  train {payload['train_bytes'].numel():,} bytes · "
          f"val {payload['val_bytes'].numel():,} bytes · "
          f"{len(payload['files'])} files · sha {payload['sha256'][:12]}", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def feats_for(arm: str, corpus: dict, which: str, start: int, n: int,
              g: torch.Generator | None = None) -> torch.Tensor | None:
    if arm not in ("bias_dist", "bias_struct", "bias_shuf"):
        return None
    kind = "dist" if arm == "bias_dist" else "struct"
    indent = corpus[f"{which}_indent"][start:start + n].float()
    bracket = corpus[f"{which}_bracket"][start:start + n].float()
    line = corpus[f"{which}_line"][start:start + n].float()
    if arm == "bias_shuf":
        perm = torch.randperm(n, generator=g)
        indent, bracket, line = indent[perm], bracket[perm], line[perm]
    d = torch.arange(n).float()
    return pair_features(d[None], indent[None], bracket[None], line[None], kind)


def run_arm(arm: str, seed: int, corpus: dict, args) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 101)
    train = corpus["train_bytes"]
    n_train = int(train.numel())
    model = CodeLM(vocab=256, dim=args.dim, depth=args.depth, heads=args.heads,
                   ctx=args.ctx, pos=arm, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    curve = []
    for step in range(args.steps):
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup) /
                                            max(1, args.steps - args.warmup)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        starts = torch.randint(0, n_train - args.ctx - 1, (args.batch,), generator=g)
        x = torch.stack([train[s:s + args.ctx] for s in starts]).long()
        y = torch.stack([train[s + 1:s + args.ctx + 1] for s in starts]).long()
        if arm in ("bias_dist", "bias_struct", "bias_shuf"):
            feats = torch.cat([feats_for(arm, corpus, "train", int(s), args.ctx, g)
                               for s in starts], 0)
        else:
            feats = None
        logits = model(x, feats)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) in (args.steps // 2, args.steps):
            limit = 64_000 if step + 1 < args.steps else None
            ev = evaluate(model, arm, corpus, args, limit)
            curve.append({"step": step + 1, "train_loss": float(loss.detach()), **ev})
            print(f"  {arm:12s} seed {seed} {step + 1:4d} train {float(loss.detach()):.3f} "
                  f"val {ev['bits_per_byte']:.4f} b/B · {time.time() - t0:.0f}s", flush=True)
    last = curve[-1]
    return {"arm": arm, "seed": seed, "params": model.n_params(),
            "bits_per_byte": last["bits_per_byte"], "curve": curve,
            "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def evaluate(model, arm: str, corpus: dict, args, limit: int | None = None) -> dict:
    model.eval()
    data = corpus["val_bytes"]
    n = int(data.numel()) if limit is None else min(int(data.numel()), limit)
    ce, count = 0.0, 0
    for start in range(0, n - args.ctx - 1, args.ctx):
        x = data[start:start + args.ctx].long()[None]
        y = data[start + 1:start + args.ctx + 1].long()[None]
        feats = feats_for(arm, corpus, "val", start, args.ctx)
        logits = model(x, feats)
        ce += float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum"))
        count += y.numel()
    model.train()
    return {"bits_per_byte": ce / (count * math.log(2)), "bytes": count}


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(path: str, results: list[dict], corpus: dict) -> str:
    """Bits/byte relative to RoPE: the only scale on which the arms separate."""
    im, d = _canvas(1180, 560, "R1: does the pair-bias mechanism transfer to code?",
                    "CPython 3.12.0 Lib subset, byte level, held-out files · "
                    "bits per byte relative to RoPE · lower is better")
    arms = [a for a in ARMS if any(x["arm"] == a for x in results)]
    base = [x["bits_per_byte"] for x in results if x["arm"] == "rope"]
    bm = sum(base) / len(base)
    delta = {a: [x["bits_per_byte"] - bm for x in results if x["arm"] == a] for a in arms}
    lo = min(min(v) for v in delta.values() if v) - 0.02
    hi = max(max(v) for v in delta.values() if v) + 0.20
    l, r, t, b = 110, 700, 140, 430

    def py(v):
        return b - (v - lo) / (hi - lo) * (b - t)

    for i in range(5):
        v = lo + i * (hi - lo) / 4
        yy = py(v)
        d.line([l, yy, r, yy], fill=GRID, width=1)
        d.text((l - 10, yy), f"{v:+.2f}", fill=(150, 158, 172), anchor="rm", font=_font(12))
    d.line([l, py(0.0), r, py(0.0)], fill=(150, 158, 172), width=1)
    d.text((r + 8, py(0.0)), f"RoPE {bm:.3f}", fill=(190, 196, 208), anchor="lm",
           font=_font(11))

    width = (r - l) / len(arms)
    for i, arm in enumerate(arms):
        vals = delta[arm]
        cx = l + (i + 0.5) * width
        m = sum(vals) / len(vals)
        d.rectangle([cx - width * 0.28, py(m), cx + width * 0.28, py(0.0)], fill=COLORS[arm])
        for v in vals:
            d.ellipse([cx - 3, py(v) - 3, cx + 3, py(v) + 3], fill=INK)
        d.text((cx, b + 18), XLABELS[arm], fill=(190, 196, 208),
               anchor="ma", font=_font(11), align="center")
        y = min(py(m), py(0.0)) - 5
        d.text((cx, y), f"{m:+.3f}", fill=INK, anchor="mb", font=_font(12))
        d.text((cx, y - 15), f"{bm + m:.3f}", fill=DIM, anchor="mb", font=_font(11))
    d.text((l, t - 24), "held-out bits/byte relative to RoPE (absolute value above it)",
           fill=(190, 196, 208), font=_font(13))

    lx = 800
    d.text((lx + 250, 128), "delta", fill=DIM, anchor="rm", font=_font(12))
    d.text((lx + 330, 128), "b/B", fill=DIM, anchor="rm", font=_font(12))
    for i, arm in enumerate(arms):
        yy = 160 + i * 30
        d.rectangle([lx, yy - 6, lx + 12, yy + 6], fill=COLORS[arm])
        d.text((lx + 22, yy), LABELS[arm], fill=INK, anchor="lm", font=_font(13))
        vals = delta[arm]
        m = sum(vals) / len(vals)
        d.text((lx + 250, yy), f"{m:+.3f}", fill=INK, anchor="rm", font=_font(13))
        d.text((lx + 330, yy), f"{bm + m:.3f}", fill=INK, anchor="rm", font=_font(13))
    d.text((96, 500), f"{corpus['train_bytes'].numel():,} train / "
           f"{corpus['val_bytes'].numel():,} held-out bytes from "
           f"{len(corpus['files'])} files, split by file. Same parameters and steps in "
           f"every arm;", fill=DIM, font=_font(12))
    d.text((96, 520), "in the bias arms the bias is the only source of position, so this "
           "asks whether the mechanism can replace RoPE, not decorate it.", fill=DIM,
           font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--corpus", default="runs/code/corpus.pt")
    p.add_argument("--corpus-src", default="/tmp/opencode/cpy/cpython-3.12.0/Lib")
    p.add_argument("--rebuild-corpus", action="store_true")
    p.add_argument("--out", default="runs/code")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    corpus = load_corpus(a.corpus, a.corpus_src, a.rebuild_corpus)
    print(f"train {corpus['train_bytes'].numel():,} · val {corpus['val_bytes'].numel():,} · "
          f"sha {corpus['sha256'][:12]}", flush=True)

    results = []
    for arm in a.arms:
        for seed in a.seeds:
            print(f"[{arm} seed {seed}]", flush=True)
            results.append(run_arm(arm, seed, corpus, a))
            with open(os.path.join(a.out, "results.json"), "w") as f:
                json.dump({"config": vars(a), "corpus_sha256": corpus["sha256"],
                           "files": corpus["files"], "results": results}, f, indent=2)
    figure(os.path.join(a.out, "code-r1.png"), results, corpus)
    print(f"{'arm':12s} {'b/B':>7s} {'params':>9s}")
    for arm in a.arms:
        xs = [x["bits_per_byte"] for x in results if x["arm"] == arm]
        ps = [x["params"] for x in results if x["arm"] == arm]
        print(f"{arm:12s} {sum(xs) / len(xs):7.4f} {ps[0]:9,}")


if __name__ == "__main__":
    main()
