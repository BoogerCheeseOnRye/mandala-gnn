"""Step 1 of the images + code plan: the meter.

Every later step is scored as a delta against this table, so the meter has to
be honest about what it is measuring. Two artifacts, one envelope:

    code    the R1 byte transformer (RoPE, 385 k parameters) trained from
            scratch on the pinned CPython `Lib/` subset, then used as a
            *generator*: fixed prompts, bytes generated, bytes/s, compiled or
            not, and bits/byte on held-out files. An order-2 byte n-gram is the
            floor: it is cheap, it never compiles, and any code model that does
            not clear it by a wide margin is not a code model.

    image   the hex GNN (405 k parameters, frozen checkpoint) as a generator:
            cost of one forward pass, and an eight-pass progressive fill that
            starts from a completely hidden drawing. Scored as MSE and r
            against the drawing and against the constant-image floor (every
            node emits the mean patch).

Nothing here is a new model. The point is the number that later steps must
beat, measured on this phone with one command:

    .venv/bin/python tools/gen_meter.py --out runs/budget

The checkpoint is saved to runs/budget/code-rope.pt (gitignored, ~1.5 MB) so
steps 3-9 can reuse it; `--reuse` skips the ~11 minute training run.
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

from mandalanet.code import CodeLM  # noqa: E402
from mandalanet.data import IMAGE, patches_to_image  # noqa: E402
from mandalanet.graph import Graph  # noqa: E402
from mandalanet.model import masked_corr, masked_mse  # noqa: E402
from tools.code_experiment import load_corpus  # noqa: E402
from tools.energy_experiment import load, pearson  # noqa: E402

PROMPTS = [
    # every prompt is left-padded to the longest, so one batch can share a
    # window; padding is a newline byte, which the model has seen plenty of
    b"def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef median(xs):\n    ",
    b"class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, x):\n        ",
    b"def fib(n):\n    if n < 2:\n        return n\n    return ",
    b"import math\n\n\ndef norm(v):\n    total = 0.0\n    for x in v:\n        total += x * x\n    return ",
]
N_NEW = 96


# --------------------------------------------------------------------------- #
# code
# --------------------------------------------------------------------------- #
def train_code(corpus: dict, steps: int, seed: int = 0) -> CodeLM:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 101)
    train = corpus["train_bytes"]
    model = CodeLM(vocab=256, dim=96, depth=3, heads=4, ctx=256, pos="rope",
                   dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        lr = 1e-3 * (step + 1) / 50 if step < 50 else \
            1e-3 * 0.5 * (1.0 + math.cos(math.pi * (step - 50) / max(1, steps - 50)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        starts = torch.randint(0, train.numel() - 257, (4,), generator=g)
        x = torch.stack([train[s:s + 256] for s in starts]).long()
        y = torch.stack([train[s + 1:s + 257] for s in starts]).long()
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 300 == 0:
            print(f"  code step {step + 1:4d} train {float(loss.detach()):.3f} "
                  f"· {time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def heldout_bpb(model: CodeLM, data: torch.Tensor, ctx: int = 256) -> tuple[float, int]:
    ce, n = 0.0, 0
    for start in range(0, data.numel() - ctx - 1, ctx):
        x = data[start:start + ctx].long()[None]
        y = data[start + 1:start + ctx + 1].long()[None]
        ce += float(F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1),
                                    reduction="sum"))
        n += y.numel()
    return ce / (n * math.log(2)), n


@torch.no_grad()
def generate(model: CodeLM, prompts: list[bytes], n_new: int, seed: int,
             temp: float | None) -> list[bytes]:
    torch.manual_seed(seed)
    L = max(len(p) for p in prompts)
    idx = torch.full((len(prompts), L), 10, dtype=torch.long)
    for i, p in enumerate(prompts):
        idx[i, L - len(p):] = torch.tensor(list(p))
    for _ in range(n_new):
        logits = model(idx)[:, -1]
        if temp is None:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(logits / temp, -1)
            sp, si = probs.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < 0.95
            sp = (sp * keep) / (sp * keep).sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)
        idx = torch.cat([idx, nxt[:, None]], 1)
    return [bytes(row[L:].tolist()) for row in idx]


def compiles(text: bytes) -> bool:
    try:
        compile(text.decode("utf-8", "replace"), "<generated>", "exec")
        return True
    except SyntaxError:
        return False


def bigram(corpus: dict) -> dict:
    b = corpus["train_bytes"].to(torch.long)
    flat = b[0] * 256 + b[1:]
    counts = torch.bincount(flat, minlength=65536).reshape(256, 256).float()
    bi = (counts + 0.5) / (counts + 0.5).sum(1, keepdim=True)
    uni = (torch.bincount(b, minlength=256) + 0.5).float()
    uni = uni / uni.sum()
    probs = 0.8 * bi + 0.2 * uni[None]          # Jelinek-Mercer, no zeros
    val = corpus["val_bytes"].to(torch.long)
    lp = torch.log2(probs[val[:-1], val[1:]])
    return {"probs": probs, "bits_per_byte": float(-lp.mean())}


def bigram_samples(probs: torch.Tensor, prompts: list[bytes], n_new: int,
                   seed: int) -> list[bytes]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for p in prompts:
        cur = list(p)
        for _ in range(n_new):
            cur.append(int(torch.multinomial(probs[cur[-1]], 1, generator=g)))
        out.append(bytes(cur))
    return out


# --------------------------------------------------------------------------- #
# image
# --------------------------------------------------------------------------- #
def image_meter(ckpt: str, passes: int = 8) -> dict:
    cfg, ex, model = load(ckpt)
    model.eval()

    def graph_for(x_in: torch.Tensor, hidden: torch.Tensor) -> Graph:
        feats = torch.cat([x_in, hidden.float().unsqueeze(-1)], 1)
        return Graph(feats, ex.mesh.pos, ex.mesh.edge_index, ex.mesh.edge_type)

    n = ex.target.shape[0]

    # cost of one forward pass, no visible cells (the generation case)
    x_in = torch.zeros_like(ex.target)
    hidden = torch.ones(n, dtype=torch.bool)
    gg = graph_for(x_in, hidden)
    with torch.no_grad():
        for _ in range(3):
            model(gg)
        t0 = time.time()
        for _ in range(10):
            model(gg)
        ms = (time.time() - t0) / 10 * 1000

    # progressive fill from nothing: eight passes, each fills the next eighth
    order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    x_fill = torch.zeros_like(ex.target)
    known = torch.zeros(n, dtype=torch.bool)
    t0 = time.time()
    with torch.no_grad():
        for i in range(passes):
            cells = order[i::passes]
            pred = model(graph_for(x_fill, ~known))["node_pred"]
            x_fill[cells] = pred[cells]
            known[cells] = True
    seconds = time.time() - t0

    def score(patches: torch.Tensor) -> dict:
        img = patches_to_image(patches, ex.mesh.pos_px, IMAGE)
        return {"mse": float(((img - ex.img) ** 2).mean()),
                "r": pearson(img.flatten(), ex.img.flatten())}

    # reference: the in-distribution completion the checkpoint was trained
    # for (the held-out wedge), scored and timed the same way
    with torch.no_grad():
        t0 = time.time()
        wedge_pred = model(graph_for(ex.target * (~ex.wedge_mask).float().unsqueeze(-1),
                                     ex.wedge_mask))["node_pred"]
        wedge_s = time.time() - t0
    wedge = {"r": float(masked_corr(wedge_pred, ex.target, ex.wedge_mask)),
             "mse": float(masked_mse(wedge_pred, ex.target, ex.wedge_mask))}

    mean_patch = ex.target.mean(0, keepdim=True).repeat(n, 1)
    return {
        "checkpoint": ckpt,
        "drawing": cfg.drawing,
        "nodes": n,
        "params": model.n_params(),
        "forward_ms": ms,
        "passes": passes,
        "fill_seconds": seconds,
        "fill": score(x_fill),
        "completion": {"cells": int(ex.wedge_mask.sum()),
                       "mse": wedge["mse"], "r": wedge["r"],
                       "seconds": wedge_s},
        "constant": score(mean_patch),
        "one_pass_ms_projected": ms * passes / 1000,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--corpus", default="runs/code/corpus.pt")
    p.add_argument("--corpus-src", default="/tmp/opencode/cpy/cpython-3.12.0/Lib")
    p.add_argument("--ckpt", default="runs/main-hex")
    p.add_argument("--out", default="runs/budget")
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, "code-rope.pt")

    corpus = load_corpus(a.corpus, a.corpus_src)
    print(f"corpus sha {corpus['sha256'][:12]} · train "
          f"{corpus['train_bytes'].numel():,} · val {corpus['val_bytes'].numel():,}",
          flush=True)

    torch.set_num_threads(8)
    if a.reuse and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CodeLM(**ck["config"])
        model.load_state_dict(ck["state_dict"])
        print("reused", ckpt_path, flush=True)
    else:
        t0 = time.time()
        model = train_code(corpus, a.steps)
        print(f"trained {time.time() - t0:.0f}s", flush=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": dict(vocab=256, dim=96, depth=3, heads=4,
                                   ctx=256, pos="rope", dropout=0.1),
                    "steps": a.steps}, ckpt_path)
    model.eval()

    t0 = time.time()
    bpb, n_val = heldout_bpb(model, corpus["val_bytes"])
    eval_s = time.time() - t0
    size_mb = sum(x.numel() * x.element_size() for x in model.parameters()) / 1e6

    torch.set_num_threads(1)
    samples = []
    t0 = time.time()
    total_bytes = 0
    for mode, temp in (("greedy", None), ("sample t=0.8 top-p 0.95", 0.8)):
        seeds = [0] if temp is None else [0, 1, 2]
        for seed in seeds:
            outs = generate(model, PROMPTS, N_NEW, seed, temp)
            took = time.time() - t0
            total_bytes += sum(len(o) for o in outs)
            for i, (pr, out) in enumerate(zip(PROMPTS, outs)):
                text = pr + out
                samples.append({"mode": mode, "seed": seed, "prompt": pr.decode(),
                                "generated": out.decode("utf-8", "replace"),
                                "compiles": compiles(text)})
            print(f"  {mode:22s} seed {seed} {len(outs) * N_NEW / took:6.0f} B/s",
                  flush=True)
    bytes_per_s = total_bytes / (time.time() - t0)

    bg = bigram(corpus)
    bg_samples = []
    for seed in (0, 1, 2):
        for pr, out in zip(PROMPTS, bigram_samples(bg["probs"], PROMPTS, N_NEW, seed)):
            bg_samples.append({"mode": "bigram", "seed": seed,
                               "prompt": pr.decode(),
                               "generated": out.decode("utf-8", "replace"),
                               "compiles": compiles(pr + out)})

    def rate(rows: list[dict]) -> float:
        return sum(r["compiles"] for r in rows) / len(rows)

    res = {
        "envelope": {
            "params": model.n_params(),
            "size_mb_fp32": size_mb,
            "bytes_per_s_1thread": bytes_per_s,
            "heldout_bpb": bpb,
            "heldout_bytes": n_val,
            "eval_seconds": eval_s,
        },
        "code": {
            "model": {"dim": 96, "depth": 3, "heads": 4, "ctx": 256, "pos": "rope",
                      "steps": a.steps},
            "heldout_bpb": bpb,
            "generation": {"prompts": len(PROMPTS), "bytes_each": N_NEW,
                           "bytes_per_s": bytes_per_s,
                           "compile_rate": rate(samples),
                           "compiled": sum(r["compiles"] for r in samples),
                           "n": len(samples)},
            "bigram": {"heldout_bpb": bg["bits_per_byte"],
                       "compile_rate": rate(bg_samples),
                       "compiled": sum(r["compiles"] for r in bg_samples),
                       "n": len(bg_samples)},
            "samples": samples + bg_samples,
        },
        "image": image_meter(a.ckpt),
    }
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out, "samples.txt"), "w") as f:
        for r in samples + bg_samples:
            f.write(f"=== {r['mode']} · seed {r['seed']} · "
                    f"{'COMPILES' if r['compiles'] else 'syntax error'} ===\n")
            f.write(r["prompt"] + r["generated"] + "\n\n")

    print()
    print(f"{'code':22s} {'params':>9s} {'b/B':>6s} {'B/s':>6s} {'compiles':>10s}")
    print(f"{'  RoPE byte LM':22s} {model.n_params():9,d} {bpb:6.3f} "
          f"{bytes_per_s:6.0f} {rate(samples):9.1%}")
    print(f"{'  bigram floor':22s} {0:9d} {bg['bits_per_byte']:6.3f} {'—':>6s} "
          f"{rate(bg_samples):9.1%}")
    im = res["image"]
    print(f"{'image':22s} {'ms/pass':>9s} {'fill s':>6s} {'MSE':>6s} {'r':>6s}")
    print(f"  {os.path.basename(im['drawing']):20s} {im['forward_ms']:9.0f} "
          f"{im['fill_seconds']:6.1f} {im['fill']['mse']:6.4f} {im['fill']['r']:6.3f}")
    print(f"{'  constant floor':22s} {'—':>9s} {'—':>6s} "
          f"{im['constant']['mse']:6.4f} {im['constant']['r']:6.3f}")
    print(f"  {'wedge completion':20s} {'—':>9s} {im['completion']['seconds']:6.2f} "
          f"{im['completion']['mse']:6.4f} {im['completion']['r']:6.3f}")


if __name__ == "__main__":
    main()
