"""Energy head and test-time refinement -- the first "next step" after the
reconstruction prototype.

The plan, in one line: keep `RadialAttention + ChaosPyramid` exactly as it is,
learn a scalar energy on the final node embeddings,

    E(h) = sum_i phi(h_i)  +  sum_(i,j) in E  psi(h_i, h_j, e_ij)

and at inference treat completion as minimisation: take the normal forward pass
as the initial state, then descend E on the hidden nodes only. If the landscape
means anything, a lower-energy completion should also be a better redraw.

`psi` gets the same geometric edge features the attention uses (RBF over edge
length, cos/sin harmonics of the edge angle, endpoint radii, edge type), so the
energy is as symmetry-aware as the model that produced `h`. Setting
`edge_term=False` gives the node-only control the plan asks for.

Nothing here touches `MandalaNet`: the head is trained against a frozen
checkpoint and stored beside it, so the original demo keeps working untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """phi per node, psi per edge, summed. ~7k parameters at the default size."""

    def __init__(self, dim: int = 64, edge_dim: int = 22, hidden: int = 32,
                 edge_term: bool = True):
        super().__init__()
        self.dim, self.edge_dim, self.edge_term = dim, edge_dim, edge_term
        self.phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.psi = (
            nn.Sequential(
                nn.Linear(2 * dim + edge_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )
            if edge_term else None
        )

    def forward(self, h: torch.Tensor, edge_feat: torch.Tensor,
                edge_index: torch.Tensor) -> dict:
        """h: (N, dim) node embeddings, edge_feat: (E, edge_dim).

        Returns the scalar, and the per-node / per-edge fields, so a caller can
        draw the landscape instead of just quoting a number.
        """
        node = self.phi(h).squeeze(-1)                       # (N,)
        if self.edge_term and edge_feat.numel():
            src, dst = edge_index
            pair = torch.cat([h[src], h[dst], edge_feat], -1)
            edge = self.psi(pair).squeeze(-1)                # (E,)
        else:
            edge = h.new_zeros(edge_index.shape[1])

        total = node.sum() + edge.sum()
        # attribute half of each directed edge to each endpoint: with the graph
        # stored both ways, that counts every undirected edge exactly once
        share = torch.zeros_like(node).index_add_(0, edge_index[1], edge * 0.5)
        return {"total": total, "node": node, "edge": edge,
                "per_node": node + share}


# --------------------------------------------------------------------------- #
# training: frozen backbone, contrastive on states
# --------------------------------------------------------------------------- #
@dataclass
class StateBank:
    """Precomputed paired embeddings -- the backbone never runs inside the
    training loop, so the head trains in seconds.

    `pos` are states that saw most of the drawing, `neg` states that saw much
    less. The head is only ever asked to order those two, never to reproduce a
    particular target vector.
    """
    pos: list[torch.Tensor] = field(default_factory=list)
    neg: list[torch.Tensor] = field(default_factory=list)

    def pairs(self, shuffle: bool = True):
        n = max(len(self.pos), len(self.neg))
        idx = list(range(n))
        if shuffle:
            idx = torch.randperm(n).tolist()
        for i in idx:
            yield self.pos[i % len(self.pos)], self.neg[i % len(self.neg)]


def train_head(head: EnergyHead, bank: StateBank, edge_feat: torch.Tensor,
               edge_index: torch.Tensor, steps: int = 400, lr: float = 3e-3,
               margin: float = 2.0, weight_decay: float = 1e-5,
               seed: int = 0) -> list[float]:
    """low energy on what the model saw, high energy on what it had to guess:

        loss = relu(E_pos - E_neg + margin)

    Only differences matter for descent, so there is no absolute target; the
    margin just stops the two clouds from collapsing onto each other.
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    hist = []
    it = bank.pairs()
    for _ in range(steps):
        try:
            pos, neg = next(it)
        except StopIteration:
            it = bank.pairs()
            pos, neg = next(it)
        e_pos = head(pos, edge_feat, edge_index)["total"]
        e_neg = head(neg, edge_feat, edge_index)["total"]
        loss = torch.relu(e_pos - e_neg + margin)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
    return hist


# --------------------------------------------------------------------------- #
# inference: descend the energy on the hidden nodes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _states(model, g, hidden):
    out = model(g)
    return out["node"].detach(), out


def refine(model, head: EnergyHead, ex, hidden: torch.Tensor, *,
           steps: int = 20, lr: float = 0.05, anchor: float = 1e-2,
           track: bool = False) -> dict:
    """Initialise from one forward pass, then move only the hidden rows of `h`
    downhill. `anchor` pulls the move back toward the initial state, which is
    what stops a shallow head from collapsing every hidden node to one point.

    Returns the refined prediction, the energy curve, and -- when `track` -- the
    wedge correlation after every step, so stability is measured rather than
    asserted.
    """
    from .data import with_mask

    g = with_mask(ex, hidden)
    out = model(g)
    h0 = out["node"].detach()
    feat = model.geom(g)
    mask = hidden.unsqueeze(-1)

    delta = torch.zeros_like(h0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    from .model import masked_corr, masked_mse

    with torch.no_grad():
        e0 = float(head(h0, feat, g.edge_index)["total"])
        pred0 = model.recon_head(h0)
    curve = [e0]
    wedges = ([{"step": 0, "energy": e0,
                "wedge_mse": float(masked_mse(pred0, ex.target, ex.wedge_mask)),
                "wedge_corr": float(masked_corr(pred0, ex.target, ex.wedge_mask))}]
              if track else [])
    best = (e0, h0.clone(), 0)

    for _ in range(max(1, steps)):
        opt.zero_grad()
        h = torch.where(mask, h0 + delta, h0)
        e = head(h, feat, g.edge_index)["total"]
        loss = e + anchor * (delta[hidden] ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            hh = torch.where(mask, h0 + delta, h0)
            energy = float(head(hh, feat, g.edge_index)["total"])
            pred = model.recon_head(hh)
            curve.append(energy)
            if track:
                wedges.append({
                    "step": len(curve) - 1,
                    "energy": energy,
                    "wedge_mse": float(masked_mse(pred, ex.target, ex.wedge_mask)),
                    "wedge_corr": float(masked_corr(pred, ex.target, ex.wedge_mask)),
                })
            if energy < best[0]:
                best = (energy, hh.detach().clone(), len(curve) - 1)

    h_best = best[1]
    pred = model.recon_head(h_best)
    return {
        "pred": pred,
        "node": h_best,
        "energy0": curve[0],
        "energy": curve[-1],
        "energy_best": best[0],
        "best_step": best[2],
        "curve": curve,
        "track": wedges,
        "delta_norm": float(delta[hidden].detach().norm()),
    }


def refine_patches(model, head: EnergyHead, ex, hidden: torch.Tensor, *,
                   steps: int = 10, lr: float = 0.02, anchor: float = 0.01,
                   track: bool = False) -> dict:
    """The other way to descend: optimise the *completion* instead of `h`.

    The state is the predicted patch of every hidden cell, written into the
    drawing so the whole picture reads as visible -- the same kind of state the
    head was trained to score. Each step runs the frozen backbone and moves the
    patches downhill through it, anchored to where they started.

    Expensive (one forward and one backward per step) and, as measured, unsafe:
    the head was trained to *rank* states, not to be optimised, so this loop
    walks into whatever low-energy corner it has.
    """
    from .data import with_mask
    from .graph import Graph

    g = with_mask(ex, hidden)
    feat = model.geom(g)             # geometry only: constant across steps
    with torch.no_grad():
        c0 = model(g)["node_pred"].detach().clone()

    def as_drawing(patches):
        return Graph(torch.cat([patches, torch.zeros_like(patches[:, :1])], -1),
                     ex.mesh.pos, ex.mesh.edge_index, ex.mesh.edge_type)

    with torch.no_grad():
        e0 = float(head(model(as_drawing(c0))["node"], feat, g.edge_index)["total"])

    c = c0[hidden].clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    curve, hist = [e0], []
    tgt = ex.target[hidden]

    def _corr(a, b):
        a = a.reshape(-1) - a.mean()
        b = b.reshape(-1) - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))

    def _mse(a, b):
        return float(((a - b) ** 2).mean())

    if track:
        hist.append({"step": 0, "energy": e0,
                     "corr": _corr(c0[hidden], tgt), "mse": _mse(c0[hidden], tgt)})

    for _ in range(steps):
        opt.zero_grad()
        patches = c0.clone()
        patches[hidden] = c
        out = model(as_drawing(patches))
        e = head(out["node"], feat, g.edge_index)["total"]
        loss = e + anchor * ((c - c0[hidden]) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            patches = c0.clone()
            patches[hidden] = c
            energy = float(head(model(as_drawing(patches))["node"], feat,
                                 g.edge_index)["total"])
            curve.append(energy)
            if track:
                hist.append({"step": len(curve) - 1, "energy": energy,
                             "corr": _corr(c, tgt), "mse": _mse(c, tgt)})

    patches = c0.clone()
    patches[hidden] = c.detach()
    return {"pred": patches, "energy0": e0, "energy": curve[-1],
            "curve": curve, "track": hist}


def energy_of_state(model, head: EnergyHead, ex, hidden: torch.Tensor,
                    state: Optional[torch.Tensor] = None) -> float:
    """Energy of one configuration: the model's own state, or a state passed in
    (e.g. one the backbone produced after seeing more of the drawing)."""
    from .data import with_mask

    g = with_mask(ex, hidden)
    if state is None:
        with torch.no_grad():
            h = model(g)["node"]
    else:
        h = state
    with torch.no_grad():
        return float(head(h, model.geom(g), g.edge_index)["total"])
