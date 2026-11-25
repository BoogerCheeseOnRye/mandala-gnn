"""Step 4: make byte generation fast enough to use.

The meter measured 65-83 B/s single-thread because every new byte re-runs the
whole window. Two things were tried:

  * a KV cache in torch (mandalanet/code.py: forward_cached),
  * the same weights decoded in numpy (mandalanet/npcode.py).

On this device torch's per-op overhead, not arithmetic, is the floor: a
(1, 96) x (96, 288) matmul costs 400-1400 us through oneDNN and ~24 us with
oneDNN disabled, and one cached step makes ~230 aten calls. The numpy sampler
makes the same ~230 numpy calls but each costs ~2.8 us, and folding the
layer-norm affine into the following matmul and rope into one small matmul
takes it under 1 ms per byte.

This tool

  * checks the numpy sampler against torch exactly: logits, greedy bytes on the
    four meter prompts, and greedy bytes on all 40 frozen tasks,
  * measures bytes/s for torch full recompute, torch with the KV cache (oneDNN
    on and off), and numpy single-thread, one stream and four streams,
  * measures the int8 cost in bits/byte and the sizes on disk,
  * applies the step-4 gate: >= 1 kB/s single-thread, <= 1% bits/byte
    regression, <= 25 MB.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        .venv/bin/python tools/fast_sampling.py --out runs/budget

The OpenMP/BLAS thread counts matter: with 8 BLAS threads every tiny matmul
pays a thread-pool wake-up and the sampler runs at 754 B/s instead of 1126.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.code import CodeLM  # noqa: E402
from mandalanet.npcode import NumpyLM, export  # noqa: E402
from tools.code_experiment import load_corpus  # noqa: E402
from tools.gen_meter import PROMPTS, heldout_bpb  # noqa: E402
from tools.make_eval import load as load_eval  # noqa: E402

N_NEW = 96


def load_model(path: str) -> CodeLM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CodeLM(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def size_mb(model: torch.nn.Module) -> float:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "w.pt")
        torch.save(model.state_dict(), p)
        return os.path.getsize(p) / 1e6


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def measure(fn, n_bytes: int, reps: int = 3) -> dict:
    """Bytes/s, median of `reps` runs, with all runs kept."""
    runs = [n_bytes / timed(fn) for _ in range(reps)]
    return {"median": sorted(runs)[len(runs) // 2], "runs": runs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/budget/code-rope.pt")
    p.add_argument("--corpus", default="runs/code/corpus.pt")
    p.add_argument("--corpus-src", default="/tmp/opencode/cpy/cpython-3.12.0/Lib")
    p.add_argument("--eval", dest="eval_path", default="runs/budget/eval.json")
    p.add_argument("--out", default="runs/budget")
    p.add_argument("--no-tasks", action="store_true",
                   help="skip the 40-task torch/numpy agreement run (about a minute)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)

    model = load_model(a.ckpt)
    corpus = load_corpus(a.corpus, a.corpus_src)
    npz = os.path.join(a.out, "code-rope.npz")
    info = export(a.ckpt, npz, int8=True)
    lm = NumpyLM(npz)
    ev = load_eval(a.eval_path)

    # ---- equivalence ------------------------------------------------------
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        ref = model(x).numpy()
    max_logit = float(np.abs(ref - lm.logits(x.numpy())).max())

    from tools.gen_meter import generate as generate_torch
    t_ref = generate_torch(model, PROMPTS, N_NEW, 0, None)
    n_ref = lm.generate(PROMPTS, N_NEW, None)
    prompts_identical = t_ref == n_ref

    tasks = ev["code"]["tasks"]
    tasks_identical = None
    if not a.no_tasks:
        t0 = time.time()
        t_task = generate_torch(model, [t["prompt"].encode() for t in tasks], N_NEW, 0, None)
        n_task = lm.generate([t["prompt"].encode() for t in tasks], N_NEW, None)
        tasks_identical = t_task == n_task
        print(f"  torch 40 tasks in {time.time() - t0:.0f}s", flush=True)

    # ---- speed ------------------------------------------------------------
    n4 = len(PROMPTS) * N_NEW
    generate_torch(model, PROMPTS, 8, 0, None)                      # warm up
    speed = {}
    speed["torch_full_recompute"] = measure(
        lambda: generate_torch(model, PROMPTS, N_NEW, 0, None), n4)
    speed["torch_cached_mkldnn"] = measure(
        lambda: generate_torch_cached(model, PROMPTS, N_NEW), n4)
    torch.backends.mkldnn.enabled = False
    speed["torch_cached"] = measure(
        lambda: generate_torch_cached(model, PROMPTS, N_NEW), n4)

    lm.generate(PROMPTS, 8, None)                                   # warm up
    speed["numpy_4_streams"] = measure(
        lambda: lm.generate(PROMPTS, N_NEW, None), n4)
    speed["numpy_4_streams_sampled"] = measure(
        lambda: lm.generate(PROMPTS, N_NEW, 0.8, seed=0), n4)
    speed["numpy_1_stream"] = measure(
        lambda: lm.generate(PROMPTS[:1], N_NEW, None), N_NEW)

    # ---- quantisation cost and size ---------------------------------------
    bpb_fp32, n_val = heldout_bpb(model, corpus["val_bytes"])
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    bpb_int8, _ = heldout_bpb(q, corpus["val_bytes"])
    sizes = {"fp32_mb": size_mb(model), "npz_int8_mb": os.path.getsize(npz) / 1e6}

    # the fp32 export is the "same arithmetic" claim; int8 is a storage choice
    tmp_fp32 = os.path.join(a.out, ".code-rope-fp32.npz")
    export(a.ckpt, tmp_fp32, int8=False)
    with torch.no_grad():
        fp32_diff = float(np.abs(model(x).numpy()
                                 - NumpyLM(tmp_fp32).logits(x.numpy())).max())
    os.remove(tmp_fp32)

    res = {
        "checkpoint": a.ckpt,
        "params": info["params"],
        "eval_sha256": ev["sha256"],
        "equivalence": {
            "fp32_max_abs_logit_diff": fp32_diff,
            "int8_max_abs_logit_diff": max_logit,
            "greedy_4_prompts_identical": prompts_identical,
            "greedy_40_tasks_identical": tasks_identical,
            "n_prompts": N_NEW,
        },
        "speed_bytes_per_s": speed,
        "bits_per_byte": {"fp32": bpb_fp32, "int8": bpb_int8,
                          "delta": (bpb_int8 - bpb_fp32) / bpb_fp32,
                          "val_bytes": n_val},
        "size_mb": sizes,
        "threads": {"torch": torch.get_num_threads(),
                    "openblas": os.environ.get("OPENBLAS_NUM_THREADS", "default"),
                    "omp": os.environ.get("OMP_NUM_THREADS", "default")},
    }
    res["gate"] = {
        "speed_ok": speed["numpy_1_stream"]["median"] >= 1000.0,
        "bpb_ok": abs(res["bits_per_byte"]["delta"]) <= 0.01,
        "size_ok": sizes["npz_int8_mb"] <= 25.0,
    }
    with open(os.path.join(a.out, "speed.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"cache    numpy vs torch: fp32 logits {fp32_diff:.1e} · "
          f"int8 logits {max_logit:.2f} · greedy bytes "
          f"{'identical' if prompts_identical else 'DIFFER'} on 4 prompts, "
          f"{'identical' if tasks_identical else tasks_identical} on 40 tasks")
    print(f"speed    torch full {speed['torch_full_recompute']['median']:6.0f} B/s · "
          f"torch cached (oneDNN) {speed['torch_cached_mkldnn']['median']:6.0f} · "
          f"torch cached {speed['torch_cached']['median']:6.0f}")
    print(f"         numpy 1 stream {speed['numpy_1_stream']['median']:6.0f} B/s · "
          f"4 streams {speed['numpy_4_streams']['median']:6.0f} · "
          f"4 streams sampled {speed['numpy_4_streams_sampled']['median']:6.0f}")
    print(f"b/B      fp32 {bpb_fp32:.4f} · int8 {bpb_int8:.4f} "
          f"({100 * res['bits_per_byte']['delta']:+.2f}%)")
    print(f"size     fp32 {sizes['fp32_mb']:.2f} MB · int8 npz {sizes['npz_int8_mb']:.2f} MB")
    print(f"gate     speed {res['gate']['speed_ok']} · bpb {res['gate']['bpb_ok']} "
          f"· size {res['gate']['size_ok']}")


@torch.no_grad()
def generate_torch_cached(model: CodeLM, prompts: list[bytes], n_new: int) -> list[bytes]:
    L = max(len(p) for p in prompts)
    idx = torch.full((len(prompts), L), 10, dtype=torch.long)
    for i, p in enumerate(prompts):
        idx[i, L - len(p):] = torch.tensor(list(p))
    logits, cache = model.forward_cached(idx)
    out = [[] for _ in prompts]
    for _ in range(n_new):
        nxt = logits[:, -1].argmax(-1)
        for i in range(len(prompts)):
            out[i].append(int(nxt[i]))
        logits, cache = model.forward_cached(nxt[:, None], cache)
    return [bytes(o) for o in out]


if __name__ == "__main__":
    main()
