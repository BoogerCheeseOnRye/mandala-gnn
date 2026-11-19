"""Train MandalaNet.

    python -m mandalanet.train --drawing drawings/star-mandala.jpg --mesh hex
    python -m mandalanet.train --task classify
    python -m mandalanet.train --mesh grid --out runs/grid   # topology ablation
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import torch

from .data import (IMAGE, classification_set, patches_to_image, recon_example,
                   save_triptych, with_mask)
from .model import MandalaNet, masked_corr, masked_mse, total_loss

DRAWINGS = ["drawings/star-mandala.jpg",
            "drawings/hub-chart.jpg",
            "drawings/hex-tessellation.jpg"]


@dataclass
class Config:
    drawing: str = DRAWINGS[0]
    mesh: str = "hex"
    task: str = "recon"
    dim: int = 64
    depth: int = 3
    blocks: int = 2
    heads: int = 4
    harmonics: int = 4
    rbf_bins: int = 8
    cell: float = 0.125
    dropout: float = 0.1
    edge_gate: bool = True
    round_patches: bool = False
    mask_ratio: float = 0.2
    n_sectors: int = 8
    val_sector: int = 0
    sparsity: float = 1e-3
    lr: float = 3e-3
    weight_decay: float = 5e-3
    steps: int = 400
    seed: int = 0
    out: str = ""
    threads: int = 1


def build(cfg: Config, in_dim: int, out_dim: int, n_classes: int = 3) -> MandalaNet:
    torch.manual_seed(cfg.seed)
    return MandalaNet(in_dim=in_dim, out_dim=out_dim, n_classes=n_classes,
                      dim=cfg.dim, depth=cfg.depth, blocks=cfg.blocks,
                      heads=cfg.heads, harmonics=cfg.harmonics,
                      rbf_bins=cfg.rbf_bins, dropout=cfg.dropout, cell=cfg.cell,
                      use_edge_gate=cfg.edge_gate)


def _opt(model, cfg: Config):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr * 0.1)
    return opt, sched


def run_recon(cfg: Config, save: bool = True, save_weights: bool = True) -> dict:
    torch.set_num_threads(cfg.threads)
    ex = recon_example(cfg.drawing, mesh=cfg.mesh, mask_ratio=cfg.mask_ratio,
                       n_sectors=cfg.n_sectors, val_sector=cfg.val_sector,
                       seed=cfg.seed, circle=cfg.round_patches)
    model = build(cfg, ex.patch_dim + 1, ex.patch_dim)
    opt, sched = _opt(model, cfg)
    # a fresh random mask every step: the model has to interpolate, not memorise
    pool = (~ex.wedge_mask).nonzero(as_tuple=True)[0]
    k = int(round(cfg.mask_ratio * pool.numel()))
    gen = torch.Generator().manual_seed(cfg.seed + 991)
    g = ex.graph
    t0 = time.time()
    history = []
    for step in range(cfg.steps):
        model.train()
        perm = pool[torch.randperm(pool.numel(), generator=gen)]
        g = with_mask(ex, ex.wedge_mask.index_fill(0, perm[:k], True))
        out = model(g)
        loss = total_loss(out, ex.target, ex.wedge_mask.index_fill(0, perm[:k], True),
                          cfg.sparsity)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, cfg.steps // 8) == 0:
            history.append((step, float(loss.detach())))
    secs = time.time() - t0

    model.eval()
    with torch.no_grad():
        g = with_mask(ex, ex.train_mask | ex.val_mask | ex.wedge_mask)
        out = model(g)
        pred = out["node_pred"]
        known = ~(ex.train_mask | ex.val_mask | ex.wedge_mask)
        mean_patch = ex.target[known].mean(0, keepdim=True).expand_as(ex.target)
        metrics = {
            "train_mse": float(masked_mse(pred, ex.target, ex.train_mask)),
            "val_mse": float(masked_mse(pred, ex.target, ex.val_mask)),
            "wedge_mse": float(masked_mse(pred, ex.target, ex.wedge_mask)),
            "val_corr": float(masked_corr(pred, ex.target, ex.val_mask)),
            "wedge_corr": float(masked_corr(pred, ex.target, ex.wedge_mask)),
            "mean_baseline_val_mse": float(masked_mse(mean_patch, ex.target, ex.val_mask)),
            "mean_baseline_wedge_mse": float(masked_mse(mean_patch, ex.target, ex.wedge_mask)),
            "gate_mean": float(out["gate_mean"]),
            "level_weights": [round(float(w), 4) for w in out["level_weights"]],
            "depth_used": int(out["depth_used"]),
            "params": model.n_params(),
            "nodes": g.num_nodes,
            "edges": g.num_edges,
            "seconds": round(secs, 1),
        }
        pred_img = patches_to_image(pred, ex.mesh.pos_px, IMAGE)
        unknown = ex.train_mask | ex.val_mask | ex.wedge_mask
        unknown_img = patches_to_image(1.0 - unknown.float().unsqueeze(-1),
                                       ex.mesh.pos_px, IMAGE)
        composite = torch.where(unknown_img > 0.5, pred_img, ex.img)

    cfg_dict = asdict(cfg)
    if save and cfg.out:
        os.makedirs(cfg.out, exist_ok=True)
        save_triptych(os.path.join(cfg.out, "recon.png"), ex.img,
                      ex.masked_input_img, pred_img, composite)
        with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
            json.dump({"config": cfg_dict, "metrics": metrics,
                       "history": history}, f, indent=2)
        if save_weights:
            torch.save({"config": cfg_dict, "state_dict": model.state_dict(),
                        "metrics": metrics},
                       os.path.join(cfg.out, "model.pt"))
    return {"config": cfg_dict, "metrics": metrics, "history": history}


def run_classify(cfg: Config) -> dict:
    torch.set_num_threads(cfg.threads)
    tr = classification_set(DRAWINGS, [0, 90, 180, 270], mesh=cfg.mesh)
    te = classification_set(DRAWINGS, [45, 135, 225, 315], mesh=cfg.mesh)
    model = build(cfg, tr.graph.x.shape[1], 16, n_classes=len(DRAWINGS))
    opt, sched = _opt(model, cfg)
    t0 = time.time()
    for step in range(cfg.steps):
        model.train()
        out = model(tr.graph)
        loss = torch.nn.functional.cross_entropy(out["logits"], tr.labels)
        loss = loss + cfg.sparsity * out["gate_mean"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        acc_tr = (model(tr.graph)["logits"].argmax(-1) == tr.labels).float().mean()
        acc_te = (model(te.graph)["logits"].argmax(-1) == te.labels).float().mean()
    metrics = {"train_acc": float(acc_tr), "test_acc": float(acc_te),
               "train_loss": float(loss), "params": model.n_params(),
               "seconds": round(time.time() - t0, 1)}
    cfg_dict = asdict(cfg)
    if cfg.out:
        os.makedirs(cfg.out, exist_ok=True)
        with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
            json.dump({"config": cfg_dict, "metrics": metrics}, f, indent=2)
    return {"config": cfg_dict, "metrics": metrics}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drawing", default=Config.drawing)
    p.add_argument("--mesh", default=Config.mesh,
                   choices=["hex", "grid", "rings", "chart", "random", "jitter", "none"])
    p.add_argument("--task", default="recon", choices=["recon", "classify"])
    p.add_argument("--dim", type=int, default=Config.dim)
    p.add_argument("--depth", type=int, default=Config.depth)
    p.add_argument("--blocks", type=int, default=Config.blocks)
    p.add_argument("--heads", type=int, default=Config.heads)
    p.add_argument("--harmonics", type=int, default=Config.harmonics)
    p.add_argument("--rbf-bins", type=int, default=Config.rbf_bins)
    p.add_argument("--cell", type=float, default=Config.cell)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--no-edge-gate", action="store_true")
    p.add_argument("--round-patches", action="store_true")
    p.add_argument("--mask-ratio", type=float, default=Config.mask_ratio)
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--threads", type=int, default=Config.threads)
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--out", default="")
    a = p.parse_args()

    cfg = Config(drawing=a.drawing, mesh=a.mesh, task=a.task, dim=a.dim, depth=a.depth,
                 blocks=a.blocks, heads=a.heads, harmonics=a.harmonics,
                 rbf_bins=a.rbf_bins, cell=a.cell, dropout=a.dropout,
                 edge_gate=not a.no_edge_gate, round_patches=a.round_patches,
                 mask_ratio=a.mask_ratio,
                 steps=a.steps, lr=a.lr, seed=a.seed, out=a.out,
                 threads=a.threads)
    res = (run_classify(cfg) if cfg.task == "classify"
           else run_recon(cfg, save_weights=not a.no_save_model))
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()
