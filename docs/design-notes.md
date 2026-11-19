# Design notes: prompt -> code, line by line

The Grok prompt had four asks. Here is where each one lives, and what had to be
decided to make it run instead of just sound good.

### 1. "hexagonal sketch as a graph, centre nodes connect via radial edges for sparse message passing"

`lattices.hex_mesh` builds a pointy-top honeycomb at a given centre spacing and
links each cell to its six neighbours (`RADIAL`). Because the hex cell centres
of a drawing are *not* the drawing, the graph is only half the story: every node
also owns an 8x8 px window of the photograph, which is what the model actually
predicts.

Sparsity is enforced by construction: 6 neighbours + 6 second-ring star edges
(the long radial lines in the tessellation sketch), never a dense kNN graph.

Controls, so "sparse message passing" can be tested rather than asserted:

* `hex_near` -- the same nodes with the star edges removed.
* `rewired` -- the *same node positions*, same degree, random edges. This is the
  strict topology test: sampling points identical, only the wiring changes.
* `jitter`/`random` -- freehand positions, kNN edges.
* `grid` -- a square lattice of the same density.
* `none` -- every edge deleted, i.e. an MLP with the same trunk.

One confound had to be handled: patches are square windows, so a square grid
tiles the image exactly and gets a free ride. `--round-patches` samples discs
instead, which removes that alignment advantage; results are reported both ways.

### 2. "embed the freehand symmetry directly into attention mechanisms"

`layers.RadialAttention` adds a learned bias to the attention logit:

```
logit_ij = q_i.k_j / sqrt(d) + B(edge_ij)
B(edge_ij) = MLP( RBF(|p_j-p_i|), cos k*dtheta, sin k*dtheta,
                  |p_i|, |p_j|, edge_type )
```

The cos/sin harmonics of the edge direction are the symmetry term: the sketch is
invariant under rotation by 2pi/n, and a bias built from k*dtheta can express
exactly that without hard-coding n. `--harmonics 0` turns it off, which is one of
the ablation axes.

### 3. "hierarchical residual blocks, multi-scale chaos engine, learnable radial biases for dynamic topology adaptation"

Two different things, kept as two different mechanisms:

* **Multi-scale** (`pyramid.ChaosPyramid`): spatial cell pooling halves the grid
  each level (8x8 cells -> 4x4 -> 2x2), radial-attention blocks run at every
  scale, and the way back up is a gated residual injection
  `h_fine += sigmoid(MLP[h_up, h_fine]) * h_up`. Per-level read-outs are mixed
  by a softmax over learned weights, logged as `level_weights`.
* **Topology adaptation** (`layers.RadialAttention.gate`): each edge gets a
  learned gate in (0,1) multiplying its message. Trained with a small L1 term;
  `gate_mean` in the metrics tells you how much of the graph the model kept.
  This is the honest version of "dynamic topology": edges are re-weighted, not
  conjured, and the number is reported.

### 4. "train on a simple task, then iterate with ablation studies"

Task: masked patch reconstruction of the drawing itself (`data.recon_example`).
Three disjoint hidden sets -- training cells, unseen cells, and one 45 degree
sector that is never optimised -- so failing to generalise is visible. A
`mean-patch` predictor is reported next to every number as the floor.

`ablate.py` sweeps the sketch-derived knobs one at a time: mesh family,
harmonics, RBF bins, pyramid depth, edge gate on/off. Two seeds, mean +- sd.

## Decisions worth knowing

* **PyG is optional.** The graph code is plain PyTorch (tensors + `index_add_`),
  because the target machine is an ARM phone and compiled PyG extras are a
  liability. `mandalanet/pyg_adapter.py` exposes the same layer as a
  `MessagePassing` module for PyG pipelines; no math is duplicated.
* **The photos are cropped, not rectified.** `data.CROPS` picks the paper region
  per drawing and autocontrast normalises the lighting. Perspective in the
  hex-tessellation photo is left in: it is a real property of the input, and the
  mesh only samples it.
* **Patches, not pixels.** Predicting 4x4 pooled patches makes a node a
  meaningful unit of a drawing (a cell with some ink in it) and keeps the graph
  small enough to train in seconds.
* **`wedge_mse` is the interesting number.** Held-out random cells are mostly a
  smoothing problem; the held-out sector is where a graph that understands the
  sketch's angular structure should beat a per-position lookup.
