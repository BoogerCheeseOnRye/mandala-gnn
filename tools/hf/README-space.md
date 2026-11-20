---
title: Mandala-GNN
emoji: 🔷
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Masked-patch reconstruction on a hand-drawn hex lattice
---

# Mandala-GNN — live cockpit

A 405k-parameter geometric GNN whose topology was read off three hand drawings,
doing masked-patch reconstruction on the drawing itself. This Space runs the
real thing: every reply, gate, hop count and head share comes from a forward
pass of `runs/main-hex/model.pt`, not from a recording.

![The cockpit](cockpit.jpg)

## What to do with it

Type anything into the chat and it runs one pass:

- `hide sector 4` or `held-out wedge` — hide one 45° sector of the hex lattice
- `hide 24 random cells` — a random mask
- `clear` — unmask everything
- any other sentence — deterministically mapped to a sector, masked, redrawn;
  the reply reports the measured r, MSE and latency

Tapping the drawing masks the cells under your finger. The right rail shows the
four attention heads: each meter is that head's share of the attention routed in
the pass just run, the canvas colours each edge by its dominant head, and a halo
wave follows the BFS hop distance from the visible cells.

## The number that matters

The model never trains on one 45° sector. Filling it in therefore cannot be done
by memorising positions — the information has to travel across the graph.

| | wedge r | wedge MSE |
|---|---|---|
| mean-patch floor | – | 0.0154 |
| plain MLP, no edges, no pyramid | 0.00 | 0.0286 |
| pyramid pooling, no edges | 0.73 | 0.0177 |
| **honeycomb mesh, full model** | **0.90** | **0.0030** |

Removing the angular symmetry bias drops r to 0.24; removing the edge gate drops
it to 0.08. The pyramid is worth 4–5 points of correlation over a flat stack.

The honest negative: random edges on the same 203 node positions score within 2
points of the honeycomb, so the *specific* hand-drawn lattice is not doing the
measurable work at this scale. Symmetry-aware attention and the pyramid are.

## Links

- Code, ablations, paper: https://github.com/BoogerCheeseOnRye/mandala-gnn
- Project page: https://boogercheeseonrye.github.io/mandala-gnn/
- Weights in a model repo: https://huggingface.co/BoogerCheeseOnRye/mandala-gnn

> This Space is staged but unpublished: Hugging Face now requires PRO to host
> Docker Spaces on cpu-basic, and a static Space cannot run the Python server.

## Run it yourself

```bash
git clone https://github.com/BoogerCheeseOnRye/mandala-gnn
cd mandala-gnn
python -m venv .venv && .venv/bin/pip install torch numpy pillow
.venv/bin/python -m mandalanet.serve --port 8000     # then open localhost:8000
```

Training from the drawing takes 88 s on a phone CPU: `mandalanet.train --drawing
drawings/hex-tessellation.jpg --mesh hex --steps 400 --out runs/main-hex`.

No claim of a new architecture — message passing, RBF edge features, angular
harmonics and U-Net pooling are all standard. The point is that they were chosen
from the sketches, and the sweep then checks whether the sketch topology earns
its keep.
