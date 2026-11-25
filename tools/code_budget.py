"""Step 3: the code model under a fixed token budget.

The step-2 baseline (385 k, 1,200 steps, 1.2 M tokens) compiles a real body on
0 of 40 tasks and passes 0. Two explanations: too little training, or too
little capacity. This separates them by giving the *same* model three times
the tokens, with checkpoints at the old budget, and scoring the frozen set at
each one. A capacity arm is only worth its 40-100 minutes if tokens alone do
not move the number.

Protocol, pre-registered:

  * same corpus, same seed, same task set (runs/budget/eval.json, sha checked)
  * fixed token budget: `--steps` windows of 4 x 256 bytes
  * score at every checkpoint: held-out bits/byte, body-compiles and pass@1
    on the 40 tasks, greedy and one sampled seed
  * decision: if body-compile is < 4/40 at 3x tokens, the next step is
    capacity (dim 128-192), not more time on this size; if it is >= 4/40,
    the sweep starts from the longer schedule.

    .venv/bin/python tools/code_budget.py --dim 96 --depth 3 --steps 3600 \
        --ckpt-every 1200 --out runs/code-budget
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
from tools.code_experiment import load_corpus  # noqa: E402
from tools.gen_meter import (compiles_body, generate, heldout_bpb,  # noqa: E402
                             run_task)
from tools.make_eval import load as load_eval  # noqa: E402


def score_tasks(model: CodeLM, tasks: list[dict], seeds: tuple[int, ...],
                bytes_each: int = 96) -> dict:
    model.eval()
    rows = []
    prompts = [t["prompt"].encode() for t in tasks]
    modes = [("greedy", None, 0)] + [("sample t=0.8", 0.8, s) for s in seeds]
    for mode, temp, seed in modes:
        t0 = time.time()
        outs = generate(model, prompts, bytes_each, seed, temp)
        for task, prompt, out in zip(tasks, prompts, outs):
            full = prompt + out
            body = compiles_body(full, task["name"])
            passed = False
            if body:
                passed = run_task(full.decode("utf-8", "replace") + "\n"
                                  + "\n".join(task["tests"]) + "\n")
            rows.append({"mode": mode, "name": task["name"],
                         "compiles": body, "passes": passed,
                         "generated": out.decode("utf-8", "replace")})
        print(f"    {mode:16s} {len(tasks)} prompts · {time.time() - t0:.0f}s",
              flush=True)
    model.train()

    def rate(mode: str, key: str) -> float:
        xs = [r[key] for r in rows if r["mode"] == mode]
        return sum(xs) / len(xs)

    return {
        "greedy": {"compile_rate": rate("greedy", "compiles"),
                   "pass_at_1": rate("greedy", "passes")},
        "sampled": {"compile_rate": rate("sample t=0.8", "compiles"),
                    "pass_at_1": rate("sample t=0.8", "passes")},
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--ckpt-every", type=int, default=1200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-seeds", nargs="*", type=int, default=[0])
    p.add_argument("--corpus", default="runs/code/corpus.pt")
    p.add_argument("--corpus-src", default="/tmp/opencode/cpy/cpython-3.12.0/Lib")
    p.add_argument("--eval", dest="eval_path", default="runs/budget/eval.json")
    p.add_argument("--out", default="runs/code-budget")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(8)

    ev = load_eval(a.eval_path)
    tasks = ev["code"]["tasks"]
    corpus = load_corpus(a.corpus, a.corpus_src)
    print(f"eval sha {ev['sha256'][:12]} · {len(tasks)} tasks · corpus "
          f"{corpus['train_bytes'].numel():,} bytes", flush=True)

    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed + 101)
    model = CodeLM(vocab=256, dim=a.dim, depth=a.depth, heads=a.heads, ctx=256,
                   pos="rope", dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train = corpus["train_bytes"]
    total_tokens = a.steps * a.batch * 256
    print(f"arm dim {a.dim} depth {a.depth} · {model.n_params():,} params · "
          f"{a.steps} steps · {total_tokens:,} tokens", flush=True)

    out = {"config": vars(a), "eval_sha256": ev["sha256"],
           "params": model.n_params(), "total_tokens": total_tokens,
           "checkpoints": []}
    t0 = time.time()
    next_ck = a.ckpt_every
    for step in range(a.steps):
        lr = 1e-3 * (step + 1) / 50 if step < 50 else \
            1e-3 * 0.5 * (1.0 + math.cos(math.pi * (step - 50)
                                        / max(1, a.steps - 50)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        starts = torch.randint(0, train.numel() - 257, (a.batch,), generator=g)
        x = torch.stack([train[s:s + 256] for s in starts]).long()
        y = torch.stack([train[s + 1:s + 257] for s in starts]).long()
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) == next_ck or (step + 1) == a.steps:
            t1 = time.time()
            bpb, n_val = heldout_bpb(model, corpus["val_bytes"])
            ts = score_tasks(model, tasks, tuple(a.sample_seeds))
            ck = {"step": step + 1, "tokens": (step + 1) * a.batch * 256,
                  "train_loss": float(loss.detach()), "bits_per_byte": bpb,
                  "eval_bytes": n_val, "tasks": ts,
                  "seconds": round(time.time() - t0, 1)}
            out["checkpoints"].append(ck)
            name = os.path.join(a.out, f"d{a.dim}x{a.depth}-s{a.seed}-{step + 1}.pt")
            torch.save({"state_dict": model.state_dict(),
                        "config": dict(vocab=256, dim=a.dim, depth=a.depth,
                                       heads=a.heads, ctx=256, pos="rope",
                                       dropout=0.1)}, name)
            with open(os.path.join(a.out, "results.json"), "w") as f:
                json.dump(out, f, indent=2)
            print(f"  step {step + 1:4d} loss {float(loss.detach()):.3f} "
                  f"b/B {bpb:.3f} · body {ts['greedy']['compile_rate']:.1%} "
                  f"(sampled {ts['sampled']['compile_rate']:.1%}) · pass "
                  f"{ts['greedy']['pass_at_1']:.1%}/"
                  f"{ts['sampled']['pass_at_1']:.1%} · {time.time() - t1:.0f}s "
                  f"eval · {time.time() - t0:.0f}s total", flush=True)
            next_ck += a.ckpt_every

    print()
    print(f"{'step':>6s} {'tokens':>10s} {'b/B':>6s} {'body':>6s} {'pass':>6s}")
    for ck in out["checkpoints"]:
        print(f"{ck['step']:6d} {ck['tokens']:10,d} {ck['bits_per_byte']:6.3f} "
              f"{ck['tasks']['greedy']['compile_rate']:6.1%} "
              f"{ck['tasks']['greedy']['pass_at_1']:6.1%}")


if __name__ == "__main__":
    main()
