---
library_name: pytorch
tags:
  - graph-neural-networks
  - geometric-deep-learning
  - masked-autoencoder
  - pytorch
  - research-prototype
datasets:
  - BoogerCheeseOnRye/mandala-gnn-ablations
metrics:
  - mse
---

# Mandala-GNN

A 405k-parameter geometric GNN for **masked-patch reconstruction on a hand-drawn
hexagonal lattice**. Its topology — nodes, radial/ring/spoke edges, the pyramid
depth — was read off three paint-marker drawings rather than picked from a
benchmark.

This repo holds the single checkpoint behind the live demo, plus the cold-state
cache and metrics that let a viewer paint the page before torch is loaded.

| | |
|---|---|
| params | 405,342 |
| nodes / edges | 203 / 2,092 |
| checkpoint | 1.7 MB (`model.pt`) |
| training | 400 steps, 88 s, single CPU thread |
| task | predict the 4×4 pooled patch of every hidden node |

## Headline result

One 45° sector of the drawing is held out and never trained on. A
position-memorising model cannot fill it in; a model that carries the sketch's
angular structure across the graph can.

| | wedge r | wedge MSE |
|---|---|---|
| mean-patch floor | – | 0.0154 |
| plain MLP (no edges, no pyramid) | 0.00 | 0.0286 |
| pyramid pooling, **no edges** | 0.73 | 0.0177 |
| **honeycomb mesh, full model** | **0.90** | **0.0030** |

Ablations on the full model: removing the angular harmonics drops wedge r to
0.24; removing the learned edge gate drops it to 0.08. Both the symmetry prior
and the multi-scale pyramid earn their keep.

The negative result, which is part of the release: **rewiring the edges at
random on the same 203 node positions scores within 2 points of the honeycomb.**
The specific hand-drawn lattice is not measurably doing the work at this task
scale — the geometry-aware attention is.

## Files

| file | what it is |
|---|---|
| `model.pt` | `{config, state_dict, metrics}`, 225 tensors, CPU |
| `metrics.json` | config + final metrics + training curve |
| `demo.json` | drawing, mesh, wedge and eval indices — the pre-torch state |
| `recon.png` | `drawing \| visible \| prediction \| composite`, 3× |

## Use it

The checkpoint is only meaningful with the code that built it (same layer names,
same mesh). Clone the repo and point the demo server at it:

```bash
git clone https://github.com/BoogerCheeseOnRye/mandala-gnn
cd mandala-gnn
python -m venv .venv && .venv/bin/pip install torch numpy pillow
.venv/bin/python -m mandalanet.serve --model runs/main-hex/model.pt --port 8000
```

`metrics.json` and `demo.json` must sit next to `model.pt`. From Python:

```python
import urllib.request, json
from huggingface_hub import hf_hub_download

d = hf_hub_download("BoogerCheeseOnRye/mandala-gnn", "model.pt")
for f in ("metrics.json", "demo.json"):
    hf_hub_download("BoogerCheeseOnRye/mandala-gnn", f, local_dir="runs/main-hex")

from mandalanet.serve import Runner      # from the GitHub repo
r = Runner(d, preload=True)
out = r.redraw([])                       # one real forward pass
print(out["stats"], out["heads"])        # latency, MSE, per-head attention share
```

To retrain from the drawing instead (88 s on a phone CPU):

```bash
.venv/bin/python -m mandalanet.train --drawing drawings/hex-tessellation.jpg \
    --mesh hex --steps 400 --out runs/main-hex
```

## Architecture

```
 local patch per node ──► RadialAttention ──► ChaosPyramid ──► patch head
                          content + RBF(edge length)            (4×4 pooled
                          + cos/sin n-fold harmonics             pixel values)
                          + endpoint radii + edge type
                          × learned edge gate
```

* `RadialAttention` — attention logits get a learnable bias from an RBF over edge
  length, cos/sin harmonics of the edge angle, both endpoint radii and the edge
  type; messages are scaled by a learned per-head edge gate. 3 depth levels,
  2 blocks, 4 heads, 64 dim, 4 harmonics, 8 RBF bins.
* `ChaosPyramid` — spatial cell pooling down ×3, gated residual skips back up,
  softmax mixture of per-level mean+max read-outs.

## Live demo

The cockpit runs from the repo — `python -m mandalanet.serve` and open
`localhost:8000`. It is not hosted: Hugging Face only hosts Docker Spaces for
PRO accounts, so the interactive page runs locally (it needs ~1 s to paint and
~0.25 s per forward pass) and the project page carries a recording.

- Project page and paper: https://boogercheeseonrye.github.io/mandala-gnn/
- Code and full ablations: https://github.com/BoogerCheeseOnRye/mandala-gnn

## Scope

A research prototype, not a foundation model. It reconstructs patches of one
128×128 drawing of a hex tessellation, and the class head is a small side task.
It has no text, no generalisation to arbitrary images, and no interface beyond
the demo server. No licence is asserted over the source drawings.
