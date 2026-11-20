---
task_categories:
  - graph-ml
tags:
  - graph-neural-networks
  - ablations
  - geometric-deep-learning
  - masked-autoencoder
---

# Mandala-GNN ablations

Every sweep behind the Mandala-GNN release, as the raw result files the project
page and paper read from — not a retyped table.

Each sweep is `{"base": <config>, "table": <axis → variant → metrics>, "runs":
<per-seed runs>}`, so the means, the standard deviations and the individual
seeds are all recoverable. `summary.csv` is the same numbers flattened, if you
just want to sort them.

## Files

| path | what it is |
|---|---|
| `ablations/hex400/` | lattice sweep at 400 steps: hex, hex with near-only edges, ring mandala, rewired, grid, edgeless, jittered |
| `ablations/model/` | harmonics, depth, edge gate |
| `ablations/round/` | the same lattice sweep with disc-shaped patches |
| `ablations/none/`, `ablations/none-round/` | the no-edge controls, both patch shapes |
| `metrics/` | per-run `metrics.json`: `main-hex`, `main-rings`, `mlp-0`, `mlp-1`, `classify` |
| `figures/` | the three drawings rendered as meshes |
| `summary.csv` | one row per (sweep, axis, variant) with mean ± sd |

## The numbers

Held-out 45° sector, hex tessellation drawing, never trained on:

| model | wedge r | wedge MSE |
|---|---|---|
| mean-patch floor | – | 0.0154 |
| plain MLP, no edges, no pyramid | 0.00 | 0.0286 |
| pyramid pooling, no edges | 0.73 | 0.0177 |
| random edges on the same nodes | 0.88 | 0.0033 |
| **honeycomb mesh, full model** | **0.90** | **0.0030** |
| square grid, disc patches | 0.99 | 0.0019 |

What the sweeps show:

1. **Edges are worth ~6× on the missing wedge.** The edgeless control collapses
   to the pyramid-only number.
2. **The angular symmetry prior is load-bearing.** `harmonics=0` drops wedge r
   to 0.24 with a huge seed spread; the learned edge gate is worse still (0.08).
3. **The pyramid is worth 4–5 points of r** over a flat stack of the same width.
4. **The specific lattice is *not* the source of the gain.** Rewiring the
   honeycomb's edges at random on the same 203 node positions scores within 2
   points. The 6-point grid win with square patches disappears with disc patches
   — that was patch tiling, not topology.

Point 4 is the point of publishing the sweep rather than the good run.

## Reproducing

```bash
git clone https://github.com/BoogerCheeseOnRye/mandala-gnn && cd mandala-gnn
python -m venv .venv && .venv/bin/pip install torch numpy pillow
.venv/bin/python -m mandalanet.ablate --axis mesh  --seeds 2 --workers 3 --out runs/abl-hex
.venv/bin/python -m mandalanet.ablate --axis model --seeds 2 --workers 3 --out runs/abl-model
./tools/experiments.sh        # everything, ~45 min on a phone CPU
```

Each run is 400 steps, one CPU thread; the whole suite is a coffee break, which
is why the seeds are cheap and the controls are included.

## Provenance

Single 128×128 drawing per run, generated from a photograph of a paint-marker
sketch. Numbers are from a phone CPU (aarch64, one thread), not a cluster. No
licence is asserted over the source drawings.

- Code: https://github.com/BoogerCheeseOnRye/mandala-gnn
- Weights: https://huggingface.co/BoogerCheeseOnRye/mandala-gnn
- Demo (runs locally, not hosted): https://boogercheeseonrye.github.io/mandala-gnn/
