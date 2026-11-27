"""Step 6, candidate A: a GNN trained to draw, not to reconstruct one drawing.

The main checkpoints were trained on three photographs, so asking them to fill a
synthetic drawing from nothing measures transfer, not ability. This trains the
same 405 k architecture on synthetic drawings from the three lattices, with the
mask ratio drawn from 0.1 to 1.0 each step -- the schedule a fill-from-empty
loop needs, since a model trained to fill 20% of a canvas has never been asked
for the other 80%.

The first version of this trained masks up to 0.9 only, so the loop's first
pass, on a fully empty canvas, was outside its training distribution; it scored
0.55x the constant-image floor. `--mask-max 0.9` reproduces that run.

Held-out drawings are the frozen eval seeds (1000-1011 per layout); training
uses seeds from 200000 up, which cannot collide.

    .venv/bin/python tools/synth_gnn.py --steps 1500 --out runs/synth-gnn

Writes runs/synth-gnn/{model.pt,metrics.json} in the same layout as the main
checkpoints, so tools/draw_gen.py --ckpt runs/synth-gnn scores it on the frozen
set exactly like the others.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.graph import Graph  # noqa: E402
from mandalanet.lattices import build_mesh  # noqa: E402
from mandalanet.synth import LAYOUTS, sample  # noqa: E402
from mandalanet.train import Config, build  # noqa: E402

TRAIN_SEED = 200000
VAL_SEED = 300000
VAL_PER_LAYOUT = 6


def drawing(layout: str, seed: int, size: int = 128):
    patches, coords, img = sample(layout, torch.Generator().manual_seed(seed))
    mesh = build_mesh(layout, size=size)
    return patches, coords, mesh


def graph_for(patches: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor,
              mesh) -> Graph:
    x = patches * (~mask).float().unsqueeze(-1)
    return Graph(torch.cat([x, mask.float().unsqueeze(-1)], 1), coords,
                 mesh.edge_index, mesh.edge_type)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mask-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/synth-gnn")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)

    cfg = Config(dim=a.dim, depth=a.depth, blocks=a.blocks, heads=a.heads,
                 lr=a.lr, steps=a.steps, seed=a.seed, threads=a.threads,
                 drawing="synthetic (3 layouts)")
    model = build(cfg, 16 + 1, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps,
                                                       eta_min=cfg.lr * 0.1)
    gen = torch.Generator().manual_seed(a.seed + 7)
    print(f"{model.n_params():,} params · {a.steps} steps · "
          f"{a.threads} threads", flush=True)

    t0 = time.time()
    history = []
    for step in range(a.steps):
        model.train()
        layout = LAYOUTS[step % len(LAYOUTS)]
        patches, coords, mesh = drawing(layout, TRAIN_SEED + step)
        n = patches.shape[0]
        ratio = 0.1 + (a.mask_max - 0.1) * float(torch.rand((), generator=gen))
        k = max(1, int(round(ratio * n)))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.randperm(n, generator=gen)[:k]] = True
        out = model(graph_for(patches, coords, mask, mesh))
        loss = ((out["node_pred"][mask] - patches[mask]) ** 2).mean()
        loss = loss + cfg.sparsity * out["gate_mean"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, a.steps // 10) == 0:
            history.append({"step": step, "loss": float(loss.detach()),
                            "seconds": round(time.time() - t0, 1)})
            print(f"  step {step:4d} loss {float(loss.detach()):.4f} "
                  f"{time.time() - t0:5.0f}s", flush=True)
            # a restart should cost minutes, not the run
            torch.save({"state_dict": model.state_dict(), "config": vars(cfg)},
                       os.path.join(a.out, "model.pt"))

    # held-out drawings at a fixed 50% mask: the model has to interpolate
    model.eval()
    val = []
    with torch.no_grad():
        for layout in LAYOUTS:
            for i in range(VAL_PER_LAYOUT):
                patches, coords, mesh = drawing(layout, VAL_SEED + 100 * i + len(layout))
                n = patches.shape[0]
                mask = torch.zeros(n, dtype=torch.bool)
                mask[torch.randperm(n, generator=torch.Generator().manual_seed(i))[:n // 2]] = True
                pred = model(graph_for(patches, coords, mask, mesh))["node_pred"]
                val.append({"layout": layout, "mse": float(((pred[mask] - patches[mask]) ** 2).mean()),
                            "mean_mse": float(((patches[~mask].mean(0) - patches[mask]) ** 2).mean())})
    val_mse = sum(v["mse"] for v in val) / len(val)
    mean_mse = sum(v["mean_mse"] for v in val) / len(val)
    print(f"held-out 50% mask: model mse {val_mse:.4f} · "
          f"mean-patch {mean_mse:.4f} ({mean_mse / val_mse:.2f}x)", flush=True)

    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, 
               os.path.join(a.out, "model.pt"))
    metrics = {"config": vars(cfg), "params": model.n_params(),
               "history": history, "val_mse_50pct": val_mse,
               "val_mean_patch_mse": mean_mse,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
