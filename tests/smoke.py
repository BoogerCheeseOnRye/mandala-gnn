"""Smoke tests, no pytest needed: python tests/smoke.py"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.data import recon_example
from mandalanet.lattices import build_mesh
from mandalanet.model import MandalaNet
from mandalanet.compute import GraphReasoner, chain_edges, distances, random_edges
from mandalanet.seq import SeqLM
from mandalanet.synth import LAYOUTS, sample

DRAWING = "drawings/star-mandala.jpg"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"ok  {msg}")


def main() -> None:
    for kind in ("hex", "grid", "rings", "chart", "random", "jitter", "none"):
        m = build_mesh(kind, size=64)
        check(m.pos_px.shape[1] == 2 and m.edge_index.shape[0] == 2,
              f"{kind}: {m.pos_px.shape[0]} nodes, {m.edge_index.shape[1]} edges")
        check(bool((m.pos.abs() <= 1.05).all()), f"{kind}: positions normalised")
        if m.edge_index.numel():
            check(int(m.edge_index.min()) >= 0 and int(m.edge_index.max()) < m.pos_px.shape[0],
                  f"{kind}: edge indices in range")
            check(bool((m.edge_index[0] != m.edge_index[1]).all()), f"{kind}: no self loops")

    ex = recon_example(DRAWING, mesh="hex", size=64, seed=0)
    check(not bool((ex.train_mask & ex.val_mask).any()), "train/val masks disjoint")
    check(not bool((ex.train_mask & ex.wedge_mask).any()), "train/wedge masks disjoint")
    check(bool(ex.wedge_mask.any()) and bool(ex.val_mask.any()), "eval sets non-empty")
    known = ~(ex.train_mask | ex.val_mask | ex.wedge_mask)
    check(float(ex.graph.x[~known][:, :-1].abs().max()) == 0.0, "masked cells are zeroed")

    model = MandalaNet(in_dim=ex.patch_dim + 1, out_dim=ex.patch_dim,
                       dim=32, depth=3, blocks=1, heads=4, harmonics=4)
    out = model(ex.graph)
    check(out["node_pred"].shape == ex.target.shape, "reconstruction head shape")
    check(out["depth_used"] >= 2, "pyramid actually coarsened the graph")
    check(0.0 < float(out["gate_mean"]) < 1.0, "edge gates in (0,1)")
    check(abs(float(out["level_weights"].sum()) - 1.0) < 1e-5, "level weights sum to 1")
    check(torch.isfinite(out["node_pred"]).all(), "no NaNs")

    loss = ((out["node_pred"][ex.train_mask] - ex.target[ex.train_mask]) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check(len(grads) > 0 and all(torch.isfinite(g).all() for g in grads), "gradients finite")

    tiny = MandalaNet(in_dim=ex.patch_dim + 1, out_dim=ex.patch_dim, dim=16, depth=1, blocks=1)
    check(tiny(ex.graph)["node_pred"].shape == ex.target.shape, "depth=1 works")

    for pos in ("none", "abs1d", "abs2d", "geom", "geom_rbf"):
        m = SeqLM(vocab=32, dim=32, depth=1, heads=4, pos=pos, dropout=0.0)
        tok = torch.randint(0, 32, (2, 12))
        logits = m(tok[:, :-1], torch.rand(2, 12, 2) * 2 - 1)
        check(logits.shape == (2, 11, 32), f"seq pos={pos}: logits shape")
    for layout in LAYOUTS:
        p, c, img = sample(layout, torch.Generator().manual_seed(0))
        check(p.shape[1] == 16 and c.shape == (p.shape[0], 2),
              f"synth {layout}: {p.shape[0]} nodes, 4x4 patches")
        check(float(img.min()) >= 0 and float(img.max()) <= 1, f"synth {layout}: image range")

    ce = chain_edges(9)
    dist = distances(9, ce, 5)
    check(int(dist[0, 8]) == 5 and int(dist[0, 0]) == 0, "chain: BFS distances")
    req = GraphReasoner(dim=16, classes=6, steps=5)
    out = req(torch.eye(9)[:4], ce)
    check(out.shape == (4, 9, 6), "reasoner: per-node distance logits")
    re = random_edges(12, 12, seed=0)
    a = torch.zeros(12, 12)
    a[re[0], re[1]] = 1
    check(bool(torch.equal(a, a.T)), "random edges: symmetric")
    check(bool((distances(12, re, 12) < 12).all()), "random edges: connected")

    from tools.reach_task import auc, choose_k, encode, pair_sets

    d9 = distances(9, chain_edges(9), 9)
    k9, rate9 = choose_k(d9)
    check(1 <= k9 <= 16 and abs(rate9 - 0.5) <= 0.1,
          f"reach: k chosen near balance (k={k9}, rate={rate9:.2f})")
    pos, neg = pair_sets(d9, 2, torch.arange(9))
    check(pos.shape[1] == 2 and neg.shape[1] == 2, "reach: pair sets are (pair, 2)")
    x = encode(torch.tensor([[0, 3], [4, 1]]), 9)
    check(x.shape == (2, 9, 2) and float(x[0, 0, 0]) == 1 and float(x[0, 3, 1]) == 1,
          "reach: source and target channels")
    check(auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0.0, 0.0, 1.0, 1.0])) == 1.0,
          "reach: AUC on a separable split")
    check(auc(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([0.0, 1.0, 0.0, 1.0])) == 0.5,
          "reach: AUC at chance on ties")

    dead = recon_example(DRAWING, mesh="none", size=64, seed=0)
    mlp = MandalaNet(in_dim=dead.patch_dim + 1, out_dim=dead.patch_dim, dim=16,
                     depth=2, blocks=1)
    o = mlp(dead.graph)
    check(torch.isfinite(o["node_pred"]).all(), "edgeless (MLP) variant runs")
    try:
        from mandalanet.layers import RadialAttention
        from mandalanet.pyg_adapter import HAS_PYG, GeoConv, from_data, to_data

        check(HAS_PYG, "torch_geometric importable (optional)")
        attn = RadialAttention(16, ex.patch_dim + 1, heads=2)
        ef = torch.randn(ex.graph.num_edges, attn.bias[0].in_features)
        pure, _ = attn(ex.graph.x[:, :16], ex.graph, ef)
        conv = GeoConv(attn, ef.shape[1])
        wrapped = conv(to_data(ex.graph).x[:, :16], ex.graph.edge_index, ef)
        check(float((pure - wrapped).abs().max()) < 1e-6, "PyG wrapper matches pure torch")
        check(from_data(to_data(ex.graph)).num_nodes == ex.graph.num_nodes,
              "Graph <-> PyG Data roundtrip")
    except ImportError as err:
        print(f"skip  torch_geometric ({err})")

    print("\nall smoke tests passed")


if __name__ == "__main__":
    main()
