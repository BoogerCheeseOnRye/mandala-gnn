# Results

All numbers from this machine (ARM phone, 1 CPU thread per run, ~2 min per
400-step run). `wedge_corr` is the Pearson r between predicted and true patch
values on the held-out 45 degree sector that is never optimised;
`wedge_mse` is the same in squared error. Lower MSE / higher r is better.

The do-nothing predictor (predict the mean visible patch at every hidden cell)
scores **wedge MSE 0.0154** on the hex-tessellation drawing and **0.0208** on the
star mandala.

## Main runs

| drawing | mesh | nodes / edges | wedge r | wedge MSE | mean-patch MSE |
|---|---|---|---|---|---|
| hex tessellation | honeycomb | 203 / 2092 | **0.904** | **0.00268** | 0.0154 |
| star mandala | 12 spokes x 5 rings | 61 / 192 | **0.742** | **0.00553** | 0.0208 |

400 steps, 405k parameters. On the hex drawing the held-out sector is
reconstructed at 5.7x lower error than the mean-patch floor, i.e. the model is
drawing the missing wedge, not smearing grey into it. `runs/main-hex/recon.png`
shows `drawing | what the model sees | prediction | composite`.

## Graph classification

Three sketches, four training rotations each, evaluated on four *unseen*
rotations (45/135/225/315): train 1.00, test 1.00, 12 test graphs.
That is a sanity check of the invariant geometry features, not a result: with
three classes and twelve graphs a coin flip would occasionally look fine.

## Ablation 1 - topology (hex drawing, square patches, 400 steps, 2 seeds)

| mesh | nodes | wedge r | wedge MSE | val MSE | gate mean |
|---|---|---|---|---|---|
| hex (honeycomb + star edges) | 203 | 0.896 +- 0.010 | 0.00280 | 0.00184 | 0.285 |
| hex_near (6 neighbours only) | 203 | 0.893 +- 0.005 | 0.00305 | 0.00375 | 0.270 |
| rewired (*same positions*, random edges) | 203 | 0.878 +- 0.001 | 0.00325 | 0.00223 | 0.259 |
| jitter (freehand positions, kNN) | 203 | 0.871 +- 0.020 | 0.00323 | 0.00422 | 0.291 |
| random (uniform positions, kNN) | 203 | 0.882 +- 0.038 | 0.00359 | 0.00362 | 0.273 |
| grid (square lattice) | 196 | **0.960** +- 0.002 | **0.00235** | 0.00522 | 0.321 |
| rings (spoke lattice) | 61 | 0.308 +- 0.080 | 0.00089 | 0.00285 | 0.242 |
| chart (hub and spoke) | 25 | 0.520 +- 0.138 | 0.00065 | 0.00130 | 0.248 |
| none (no edges) | 196 | 0.730 +- 0.002 | 0.01770 | 0.01745 | 0.000 |

Plus the strict control: **no edges and no pyramid** (`depth=1`, a plain MLP)
scores r **0.00**, wedge MSE **0.0286** - it can only emit one constant patch,
because with no edges and no pooling every hidden cell has identical input.

### What this says

* **Message passing is doing real work.** Edges + pyramid: 0.0030 MSE. Pyramid
  pooling without edges: 0.0177. Plain MLP: 0.0286. Each step is a 3-6x
  improvement, and the gate closes to a mean of ~0.28, i.e. the model keeps
  about a quarter of the attention mass it started with.
* **The specific lattice is not.** `rewired` uses the *same 203 node positions*
  as `hex` with random edges and lands within 2 correlation points (0.878 vs
  0.896). The honeycomb's 6-fold wiring buys ~1 point over a random graph at the
  same points. The `rings`/`chart` rows are not a topology statement: they have
  61 and 25 nodes, so they simply represent the drawing with far fewer patches.
* **The grid's win is a tiling artefact.** Square patches tile a square grid
  exactly, which is free information. With disc patches (`--round-patches`,
  same runs) that advantage disappears:

| mesh (disc patches) | wedge r | wedge MSE |
|---|---|---|
| hex | 0.989 +- 0.005 | 0.00198 |
| rewired | 0.987 +- 0.004 | 0.00246 |
| grid | 0.990 +- 0.000 | 0.00193 |
| none | 0.882 +- 0.001 | 0.02165 |

  Both patch shapes are now near the ceiling, so this comparison is blunt, but
  the 6-point grid gap from the square-patch table is gone. Conclusion: the
  ranking there was about how patches cover pixels, not about graph topology.

## Ablation 2 - model knobs (hex drawing, 300 steps, 2 seeds)

| harmonics | wedge r | wedge MSE |
|---|---|---|
| 0 (no angular symmetry bias) | 0.238 +- 0.319 | 0.01283 |
| 2 | 0.870 +- 0.001 | 0.00340 |
| 4 (default) | 0.888 +- 0.014 | 0.00324 |
| 8 | **0.901** +- 0.001 | 0.00265 |

| depth | wedge r | wedge MSE | params |
|---|---|---|---|
| 1 (no pyramid) | 0.844 +- 0.014 | 0.00408 | 148k |
| 2 | 0.877 +- 0.019 | 0.00334 | 277k |
| 3 (default) | 0.888 +- 0.014 | 0.00324 | 405k |
| 4 | 0.881 +- 0.031 | 0.00320 | 534k |

| edge gate | wedge r | wedge MSE |
|---|---|---|
| on | 0.888 +- 0.014 | 0.00324 |
| off | 0.078 +- 0.130 | 0.01398 |

* The symmetry harmonics are the single biggest knob: removing them collapses
  the run (one seed even fails to train), and adding more helps up to 8.
* The pyramid is worth ~4-5 correlation points over a flat stack, with
  diminishing returns after depth 2-3.
* Removing the edge gate destabilises training completely. That is a
  confounded result - the gate also damps attention by ~0.3x, so the collapse
  is "no damping + no adaptivity", not proof that adaptivity alone is
  essential. It is worth re-running with a fixed attention temperature to
  separate the two.

## Caveats

* One drawing, two seeds, 300-400 steps. Differences under ~0.02 r are noise;
  only the big gaps (edges vs none, harmonics 0 vs 8, gate on/off) should be
  trusted.
* The task is low-frequency: each node predicts a 4x4 average of an 8x8 patch,
  so most of the signal is "how much ink is around here". A topology that
  matches the drawing's line structure has little extra to offer. A harder
  task - predicting the line pattern itself, or classifying tessellation
  families - is where the lattice hypothesis should be tested next.
* `wedge_corr` is computed over raw patch values, so reproducing the overall
  ink level contributes. Read it next to `wedge_mse` and the mean-patch floor.
* The classification number is a smoke test (3 classes, 12 test graphs).
