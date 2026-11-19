"""Ablation sweep over the sketch-derived hyperparameters.

    python -m mandalanet.ablate --axis mesh harmonics depth edge_gate --seeds 2
    python -m mandalanet.ablate --axis all --steps 250 --workers 4

Each (config, seed) is a full training run in its own process (the model is tiny
but per-op overhead on a phone CPU is not). Writes results.json and report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from .train import Config, run_recon

AXES: dict[str, list] = {
    "mesh": ["hex", "hex_near", "rewired", "jitter", "random", "grid", "rings",
             "chart", "none"],
    "harmonics": [0, 2, 4, 8],
    "rbf_bins": [4, 8, 16],
    "depth": [1, 2, 3, 4],
    "edge_gate": [True, False],
}


def _job(job: tuple) -> tuple:
    import torch

    torch.set_num_threads(1)
    axis, val, seed, steps, drawing, extra = job
    cfg = replace(Config(drawing=drawing, steps=steps, **extra), seed=seed)
    setattr(cfg, axis, val)
    res = run_recon(cfg, save=False)
    return axis, str(val), seed, res["metrics"]


def _agg(vals: list[float]) -> dict:
    return {"mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", nargs="+", default=["all"])
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--drawing", default=Config.drawing)
    p.add_argument("--only", nargs="+", default=None,
                   help="restrict the axis values, e.g. --only hex grid")
    p.add_argument("--round-patches", action="store_true")
    p.add_argument("--out", default="runs/ablations")
    a = p.parse_args()

    axes = list(AXES) if "all" in a.axis else a.axis
    base = Config(drawing=a.drawing, steps=a.steps)

    extra = {"round_patches": True} if a.round_patches else {}
    vals = {ax: ([v for v in AXES[ax] if str(v) in a.only] if a.only else AXES[ax])
            for ax in axes}
    jobs = [(axis, val, seed, a.steps, a.drawing, extra)
            for axis in axes for val in vals[axis] for seed in range(a.seeds)]
    print(f"{len(jobs)} runs, {a.workers} workers", flush=True)
    results: dict[str, dict[str, dict]] = {axis: {} for axis in axes}
    raw: list[dict] = []
    done: set[tuple] = set()

    def record(res: tuple) -> None:
        axis, val, seed, metrics = res
        raw.append({"axis": axis, "value": val, "seed": seed, **metrics})
        results[axis].setdefault(val, []).append(metrics)
        done.add((axis, val, seed))
        print(f"  {axis}={val} seed={seed}: val {metrics['val_mse']:.5f} wedge "
              f"{metrics['wedge_mse']:.5f} corr {metrics['wedge_corr']:.3f}", flush=True)

    # process pool when asked; fall back to this process if the box refuses
    # (proot + little RAM can kill a worker)
    if a.workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures = [pool.submit(_job, j) for j in jobs]
                for f in as_completed(futures):
                    record(f.result())
        except Exception as err:  # noqa: BLE001
            print(f"pool died ({type(err).__name__}: {err}); finishing in-process",
                  flush=True)
    for j in jobs:
        if (j[0], str(j[1]), j[2]) not in done:
            record(_job(j))

    table: dict[str, dict[str, dict]] = {}
    for axis, entries in results.items():
        table[axis] = {}
        for val, runs in entries.items():
            if not runs:
                continue
            table[axis][val] = {
                "val_mse": _agg([r["val_mse"] for r in runs]),
                "wedge_mse": _agg([r["wedge_mse"] for r in runs]),
                "params": runs[0]["params"],
                "nodes": runs[0]["nodes"],
                "wedge_corr": _agg([r["wedge_corr"] for r in runs]),
                "gate_mean": _agg([r["gate_mean"] for r in runs]),
                "seconds": runs[0]["seconds"],
            }

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"base": base.__dict__, "table": table, "runs": raw}, f, indent=2)

    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("# Ablations\n\n")
        f.write(f"{a.seeds} seeds x {a.steps} steps, drawing `{a.drawing}`, "
                f"mean +- sd over seeds.\n\n")
        allruns = [r for runs in results.values() for rs in runs.values() for r in rs]
        f.write("`val` = held-out cells, `wedge` = one 45 deg sector never trained on. "
                "The do-nothing mean-patch predictor scores val "
                f"{st.mean([r['mean_baseline_val_mse'] for r in allruns]):.4f}, wedge "
                f"{st.mean([r['mean_baseline_wedge_mse'] for r in allruns]):.4f} MSE.\n\n")
        for axis, entries in table.items():
            f.write(f"## {axis}\n\n| value | val_mse | wedge_mse | wedge_corr | params | nodes | gate |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for k, v in entries.items():
                f.write(f"| {k} | {v['val_mse']['mean']:.5f} +- {v['val_mse']['sd']:.5f} "
                        f"| {v['wedge_mse']['mean']:.5f} +- {v['wedge_mse']['sd']:.5f} "
                        f"| {v['wedge_corr']['mean']:.3f} +- {v['wedge_corr']['sd']:.3f} "
                        f"| {v['params']} | {v['nodes']} | {v['gate_mean']['mean']:.3f} |\n")
            f.write("\n")
        f.write("## How to read this\n\n"
                "- `mesh=none` is the same network with every edge deleted: an MLP.\n"
                "- `jitter`/`random` keep node count and degree but break the honeycomb.\n"
                "- `harmonics=0` removes the angular symmetry bias, `edge_gate=False`\n"
                "  removes topology adaptation, `depth=1` removes the pyramid.\n")
    print(f"wrote {a.out}/report.md")


if __name__ == "__main__":
    main()
