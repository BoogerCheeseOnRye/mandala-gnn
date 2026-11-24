"""S: can a scalar energy rerank completions?

The energy head's last surviving job was selection: given K completions, pick
the one closest to the truth. Every earlier measurement trained the head for a
different job -- tell a state that saw most of the drawing from one that saw
little -- and it did that perfectly (24/24 on visibility) while preferring the
*wrong* answer at a fixed mask.

This tool asks the selection question in the two regimes that matter, on all
three drawings, with disjoint training and evaluation masks.

  fixed     K completions of the *same* mask at a *fixed* temperature: add
            sigma * N(0, I) to the hidden rows of the base state, decode with
            the frozen reconstruction head. Every candidate saw the same cells,
            so visibility cannot explain a ranking. This is the only regime in
            which the head has to read *quality* rather than *quantity*.

  context   K completions of the same target mask after seeing different
            amounts: each candidate additionally hides a random q of the
            currently visible cells. Error varies a lot and the head's own
            training signal (saw more vs saw less) applies. A "visibility
            oracle" that simply picks the candidate that saw the most cells is
            scored alongside, because if the head only matches that oracle it
            adds nothing a count could not give you.

  swap      K completions of the same target mask where every candidate sees
            the *same number* of cells but a different random subset. Real
            forward passes, matched information, so counting visible cells is
            exactly at chance by construction and the only way to rank is to
            read how well the completion fits.

  sample    K stochastic forward passes of the same masked drawing with the
            backbone's dropout on: a real sampler, and the case a reranker
            actually faces. Candidates differ only in the model's own noise.

True quality is the masked MSE of the decoded patches, so labels never touch
the head's input. Three heads are compared on identical candidates:

    visibility   the published training signal (saw more vs saw less)
    quality      pairwise hinge on same-family candidates ordered by error
    inverted     the quality head trained on the opposite ordering: a check
                 that the labels are learnable at all

Each head is also scored on its own *training* masks. A head that cannot fit
its training pairs has found no usable signal at all; one that fits them and
fails held out has memorised. The two failures mean different things and the
results file records both.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.data import with_mask  # noqa: E402
from mandalanet.energy import EnergyHead, StateBank, train_head  # noqa: E402
from mandalanet.model import masked_mse  # noqa: E402
from tools.energy_experiment import (BG, BLUE, DIM, GRID, INK,  # noqa: E402
                                     ORANGE, RED, _canvas, _font, GREEN,
                                     load, random_mask, pearson)

DRAWINGS = (("hex", "runs/main-hex"), ("rings", "runs/main-rings"),
            ("chart", "runs/main-chart"))
SIGMAS = (1.0, 2.0)
K = 8
N_TRAIN = 60
N_EVAL = 40
RATIOS = (0.15, 0.35)      # unseen by the visibility bank (0.04-0.11, 0.45-0.75)
CONTEXT_Q = (0.0, 0.30)    # extra hidden fraction of the visible cells
SWAP_FRAC = 0.20           # fraction of visible cells hidden, same for every candidate


# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
@torch.no_grad()
def base_state(model, ex, mask):
    return model(with_mask(ex, mask))["node"].detach()


@torch.no_grad()
def fixed_candidates(model, ex, mask, h0, sigma, k, seed):
    """Same mask, same temperature: only the noise realisation differs."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(k):
        hh = h0.clone()
        hh[mask] = hh[mask] + sigma * torch.randn(hh[mask].shape, generator=g)
        pred = model.recon_head(hh)
        comp = ex.target.clone()
        comp[mask] = pred[mask]
        out.append({"h": hh, "err": float(masked_mse(pred, ex.target, mask)),
                    "seen": int((~mask).sum()), "comp": comp})
    return out


@torch.no_grad()
def context_candidates(model, ex, mask, k, seed):
    """Same target mask, different amounts of visible context."""
    n = ex.graph.num_nodes
    pool = (~mask).nonzero(as_tuple=True)[0]
    qs = [CONTEXT_Q[0] + (CONTEXT_Q[1] - CONTEXT_Q[0]) * i / (k - 1) for i in range(k)]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(k, generator=g).tolist()
    out = []
    for i in range(k):
        q = qs[order[i]]
        extra = random_mask(n, pool, q, seed + 100 + i) & ~mask
        hidden = mask | extra
        h = model(with_mask(ex, hidden))["node"].detach()
        pred = model.recon_head(h)
        comp = ex.target.clone()
        comp[hidden] = pred[hidden]
        out.append({"h": h, "err": float(masked_mse(pred, ex.target, mask)),
                    "seen": int((~hidden).sum()), "comp": comp})
    return out


@torch.no_grad()
def swap_candidates(model, ex, mask, k, seed, frac=SWAP_FRAC):
    """Same target mask, same number of visible cells, different subsets."""
    n = ex.graph.num_nodes
    pool = (~mask).nonzero(as_tuple=True)[0]
    cnt = max(1, int(round(frac * pool.numel())))
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(k):
        idx = pool[torch.randperm(pool.numel(), generator=g)[:cnt]]
        extra = torch.zeros(n, dtype=torch.bool)
        extra[idx] = True
        hidden = mask | extra
        h = model(with_mask(ex, hidden))["node"].detach()
        pred = model.recon_head(h)
        comp = ex.target.clone()
        comp[hidden] = pred[hidden]
        out.append({"h": h, "err": float(masked_mse(pred, ex.target, mask)),
                    "seen": int((~hidden).sum()), "comp": comp})
    return out


@torch.no_grad()
def sample_candidates(model, ex, mask, k, seed):
    """K stochastic forward passes with dropout on -- a real sampler."""
    g = with_mask(ex, mask)
    model.train()
    out = []
    for i in range(k):
        torch.manual_seed(seed + i)
        o = model(g)
        pred = o["node_pred"].detach()
        comp = ex.target.clone()
        comp[mask] = pred[mask]
        out.append({"h": o["node"].detach(),
                    "err": float(masked_mse(pred, ex.target, mask)),
                    "seen": int((~mask).sum()), "comp": comp})
    model.eval()
    return out


def mask_set(n, wedge, count, seed):
    nonwedge = (~wedge).nonzero(as_tuple=True)[0]
    masks = []
    for i in range(count):
        r = RATIOS[0] + (RATIOS[1] - RATIOS[0]) * (i / max(1, count - 1))
        masks.append(random_mask(n, nonwedge, r, seed + i))
    return masks


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
def visibility_head(cfg, model, ex, feat, edges, steps=400):
    n = ex.graph.num_nodes
    wedge = ex.wedge_mask
    nonwedge = (~wedge).nonzero(as_tuple=True)[0]
    bank = StateBank()
    with torch.no_grad():
        for i in range(8):
            m = wedge | random_mask(n, nonwedge, 0.04 + 0.01 * i, 10 + i)
            bank.pos.append(model(with_mask(ex, m))["node"].detach().clone())
        for i, r in enumerate((0.45, 0.55, 0.65, 0.75)):
            for j in range(3):
                m = wedge | random_mask(n, nonwedge, r, 1000 + 10 * i + j)
                bank.neg.append(model(with_mask(ex, m))["node"].detach().clone())
    head = EnergyHead(dim=cfg.dim, edge_dim=model.geom.out_dim, edge_term=True)
    hist = train_head(head, bank, feat, edges, steps=steps)
    return head, sum(hist[-50:]) / 50


def quality_head(cfg, model, feat, edges, samples, steps=800,
                 invert_labels=False, seed=0):
    """Pairwise hinge on candidates of one family ordered by true error.

        loss = relu((E_good - E_bad) / scale + 1)

    `invert_labels` trains the same head on the opposite ordering. It is the
    control that matters: if an inverted head lands clearly below chance, the
    pipeline is demonstrably learning the labels, so a quality head at chance
    means the labels could not be learned -- not that the loop is broken.
    """
    torch.manual_seed(seed)
    head = EnergyHead(dim=cfg.dim, edge_dim=model.geom.out_dim, edge_term=True)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1.5e-4)

    pairs = []
    for states in samples:
        errs = [c["err"] for c in states]
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                good, bad = (i, j) if errs[i] < errs[j] else (j, i)
                if invert_labels:
                    good, bad = bad, good
                pairs.append((states[good]["h"], states[bad]["h"]))

    # Fix the energy scale once, at init, from the natural spread of the head's
    # own outputs. Normalising by the *live* output would let the head satisfy
    # the margin by inflating its magnitudes instead of ordering anything --
    # measured as a blow-up to +-1e6 with the ranking inverted.
    with torch.no_grad():
        ref = torch.stack([head(c["h"], feat, edges)["total"]
                           for s in samples[:12] for c in s])
        scale = float(ref.std()) + 1e-6

    g = torch.Generator().manual_seed(seed + 11)
    hist = []
    for _ in range(steps):
        i = int(torch.randint(0, len(pairs), (1,), generator=g))
        hg, hb = pairs[i]
        eg = head(hg, feat, edges)["total"]
        eb = head(hb, feat, edges)["total"]
        loss = F.relu((eg - eb) / scale + 1.0)   # good must score lower
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
    return head, sum(hist[-50:]) / 50


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _gains(errs, es):
    """How much of the available improvement the pick actually captures.

    Pairwise accuracy says whether the ordering is right; it says nothing about
    whether being right is worth anything. With K candidates whose errors span
    2%, a perfect ranker still buys almost nothing. So report the picked
    candidate's error as a fraction of the average candidate's (lower is
    better, 1.0 = no better than picking at random) and the fraction of the
    best-to-average gap it captures.
    """
    i = min(range(len(es)), key=lambda j: es[j])
    mean = sum(errs) / len(errs)
    best = min(errs)
    rel = errs[i] / mean if mean else 1.0
    gap = mean - best
    gain = (mean - errs[i]) / gap if gap > 1e-12 else 0.0
    return rel, best / mean if mean else 1.0, gain


@torch.no_grad()
def score(head, samples, feat, edges, min_gap=0.01):
    accs, tops, rhos, rels, gains = [], [], [], [], []
    for states in samples:
        errs = [c["err"] for c in states]
        es = [float(head(c["h"], feat, edges)["total"]) for c in states]
        rel, _, gain = _gains(errs, es)
        rels.append(rel)
        gains.append(gain)
        win = tot = 0
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                lo, hi = min(errs[i], errs[j]), max(errs[i], errs[j])
                if hi - lo < min_gap * (hi + 1e-9):
                    continue
                tot += 1
                win += int((es[i] < es[j]) == (errs[i] < errs[j]))
        accs.append(win / tot if tot else 0.5)
        tops.append(int(min(range(len(es)), key=lambda i: es[i])
                        == min(range(len(errs)), key=lambda i: errs[i])))
        rhos.append(pearson([-e for e in es], errs))
    t = torch.tensor(accs)
    return {"pairwise": float(t.mean()), "pairwise_sd": float(t.std()),
            "masks_above_half": int((t > 0.5).sum()), "masks": len(accs),
            "top1": float(sum(tops) / len(tops)),
            "spearman": float(torch.tensor(rhos).mean()),
            "pick_rel": float(torch.tensor(rels).mean()),
            "gain": float(torch.tensor(gains).mean()),
            "per_mask": [round(a, 4) for a in accs]}


@torch.no_grad()
def smooth_score(samples, edges):
    """A one-line image prior: prefer the completion whose neighbours agree.

    Control for the candidate families where the head has no visibility cue. If
    this cannot rank them either, the family has no quality signal for anything
    to read, and a head at chance says nothing about the head.
    """
    accs, tops, rels, gains = [], [], [], []
    for states in samples:
        errs = [c["err"] for c in states]
        es = []
        for c in states:
            comp = c["comp"]
            diff = comp[edges[0]] - comp[edges[1]]
            es.append(-float((diff ** 2).sum()))
        rel, _, gain = _gains(errs, es)
        rels.append(rel)
        gains.append(gain)
        win = tot = 0
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                lo, hi = min(errs[i], errs[j]), max(errs[i], errs[j])
                if hi - lo < 0.01 * (hi + 1e-9):
                    continue
                tot += 1
                win += int((es[i] < es[j]) == (errs[i] < errs[j]))
        accs.append(win / tot if tot else 0.5)
        tops.append(int(min(range(len(es)), key=lambda i: es[i])
                        == min(range(len(errs)), key=lambda i: errs[i])))
    t = torch.tensor(accs)
    return {"pairwise": float(t.mean()), "pairwise_sd": float(t.std()),
            "masks_above_half": int((t > 0.5).sum()), "masks": len(accs),
            "top1": float(sum(tops) / len(tops)),
            "pick_rel": float(torch.tensor(rels).mean()),
            "gain": float(torch.tensor(gains).mean()),
            "per_mask": [round(a, 4) for a in accs]}


@torch.no_grad()
def oracle_score(samples):
    """Pick the candidate that saw the most cells. No head, no learning."""
    accs, tops, rels, gains = [], [], [], []
    for states in samples:
        errs = [c["err"] for c in states]
        es = [-c["seen"] for c in states]
        rel, _, gain = _gains(errs, es)
        rels.append(rel)
        gains.append(gain)
        win = tot = 0
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                lo, hi = min(errs[i], errs[j]), max(errs[i], errs[j])
                if hi - lo < 0.01 * (hi + 1e-9):
                    continue
                tot += 1
                win += int((es[i] < es[j]) == (errs[i] < errs[j]))
        accs.append(win / tot if tot else 0.5)
        tops.append(int(min(range(len(es)), key=lambda i: es[i])
                        == min(range(len(errs)), key=lambda i: errs[i])))
    t = torch.tensor(accs)
    return {"pairwise": float(t.mean()), "pairwise_sd": float(t.std()),
            "masks_above_half": int((t > 0.5).sum()), "masks": len(accs),
            "top1": float(sum(tops) / len(tops)),
            "pick_rel": float(torch.tensor(rels).mean()),
            "gain": float(torch.tensor(gains).mean()),
            "per_mask": [round(a, 4) for a in accs]}


# --------------------------------------------------------------------------- #
# one drawing
# --------------------------------------------------------------------------- #
def run_drawing(name, ckpt, steps, args):
    cfg, ex, model = load(ckpt)
    n = ex.graph.num_nodes
    wedge = ex.wedge_mask
    feat = model.geom(ex.graph)
    edges = ex.graph.edge_index

    train_masks = mask_set(n, wedge, N_TRAIN, seed=2000)
    eval_masks = mask_set(n, wedge, N_EVAL, seed=5000) + [wedge]

    t0 = time.time()
    vis, vis_loss = visibility_head(cfg, model, ex, feat, edges, steps=400)
    print(f"  visibility head ({time.time()-t0:.0f}s, loss {vis_loss:.3f})",
          flush=True)

    out = {"drawing": name, "checkpoint": ckpt, "cells": n,
           "wedge_cells": int(wedge.sum()), "k": K,
           "n_train_masks": N_TRAIN, "n_eval_masks": len(eval_masks),
           "ratio_range": list(RATIOS), "context_q": list(CONTEXT_Q),
           "swap_frac": SWAP_FRAC, "fixed": {}, "context": {}, "swap": {},
           "sample": {}}

    def evaluate(family, samples_train, samples_eval, oracle=True):
        qh, qloss = quality_head(cfg, model, feat, edges, samples_train,
                                 steps=steps, seed=1)
        ih, iloss = quality_head(cfg, model, feat, edges, samples_train,
                                 steps=steps, invert_labels=True, seed=2)
        heads = {"visibility": vis, "quality": qh, "inverted": ih}
        block = {"quality_loss": qloss, "inverted_loss": iloss,
                 "error_spread": float(sum(
                     max(c["err"] for c in s) / max(1e-12, min(c["err"] for c in s)) - 1
                     for s in samples_eval) / len(samples_eval)),
                 "heads": {}}
        for label, head in heads.items():
            held = score(head, samples_eval, feat, edges)
            fit = score(head, samples_train, feat, edges)
            block["heads"][label] = {**held, "pairwise_train": fit["pairwise"]}
        block["oracle"] = oracle_score(samples_eval) if oracle else None
        block["smooth"] = smooth_score(samples_eval, edges)
        block["seen_err_rho"] = float(sum(
            pearson([c["seen"] for c in s], [-c["err"] for c in s])
            for s in samples_eval) / len(samples_eval))
        if block["oracle"]:
            q = block["heads"]["quality"]["per_mask"]
            v = block["heads"]["visibility"]["per_mask"]
            o = block["oracle"]["per_mask"]
            dqo = torch.tensor([a - b for a, b in zip(q, o)])
            dqv = torch.tensor([a - b for a, b in zip(q, v)])
            block["quality_minus_oracle"] = float(dqo.mean())
            block["quality_minus_oracle_sd"] = float(dqo.std())
            block["quality_minus_visibility"] = float(dqv.mean())
            block["quality_minus_visibility_sd"] = float(dqv.std())
        for label, s in block["heads"].items():
            print(f"    {family:7s} {label:11s} held-out {s['pairwise']:.3f} "
                  f"(sd {s['pairwise_sd']:.3f}, {s['masks_above_half']}/"
                  f"{s['masks']})  train {s['pairwise_train']:.3f}  "
                  f"top1 {s['top1']:.3f}  gain {s['gain']:+.2f}  "
                  f"pick {s['pick_rel']:.3f}", flush=True)
        if block["oracle"]:
            print(f"    {family:7s} {'oracle':11s} held-out {block['oracle']['pairwise']:.3f}  "
                  f"top1 {block['oracle']['top1']:.3f}", flush=True)
        sm = block["smooth"]
        print(f"    {family:7s} {'smooth':11s} held-out {sm['pairwise']:.3f} "
              f"(sd {sm['pairwise_sd']:.3f}, {sm['masks_above_half']}/{sm['masks']})  "
              f"top1 {sm['top1']:.3f}  gain {sm['gain']:+.2f}  "
              f"pick {sm['pick_rel']:.3f}", flush=True)
        return block

    for sigma in SIGMAS:
        tr = [fixed_candidates(model, ex, m, base_state(model, ex, m), sigma, K,
                               seed=9000 + 17 * i)
              for i, m in enumerate(train_masks)]
        ev = [fixed_candidates(model, ex, m, base_state(model, ex, m), sigma, K,
                               seed=7000 + 17 * i)
              for i, m in enumerate(eval_masks)]
        print(f"  fixed sigma {sigma}: spread "
              f"{100*sum(max(c['err'] for c in s)/max(1e-12,min(c['err'] for c in s))-1 for s in ev)/len(ev):.0f}%",
              flush=True)
        out["fixed"][str(sigma)] = evaluate(f"fixed{sigma:g}", tr, ev, oracle=False)

    tr = [context_candidates(model, ex, m, K, seed=9000 + 17 * i)
          for i, m in enumerate(train_masks)]
    ev = [context_candidates(model, ex, m, K, seed=7000 + 17 * i)
          for i, m in enumerate(eval_masks)]
    print(f"  context: spread "
          f"{100*sum(max(c['err'] for c in s)/max(1e-12,min(c['err'] for c in s))-1 for s in ev)/len(ev):.0f}%",
          flush=True)
    out["context"] = evaluate("context", tr, ev)

    tr = [swap_candidates(model, ex, m, K, seed=9000 + 17 * i)
          for i, m in enumerate(train_masks)]
    ev = [swap_candidates(model, ex, m, K, seed=7000 + 17 * i)
          for i, m in enumerate(eval_masks)]
    print(f"  swap: spread "
          f"{100*sum(max(c['err'] for c in s)/max(1e-12,min(c['err'] for c in s))-1 for s in ev)/len(ev):.0f}%",
          flush=True)
    out["swap"] = evaluate("swap", tr, ev)

    tr = [sample_candidates(model, ex, m, K, seed=9000 + 17 * i)
          for i, m in enumerate(train_masks)]
    ev = [sample_candidates(model, ex, m, K, seed=7000 + 17 * i)
          for i, m in enumerate(eval_masks)]
    print(f"  sample: spread "
          f"{100*sum(max(c['err'] for c in s)/max(1e-12,min(c['err'] for c in s))-1 for s in ev)/len(ev):.0f}%",
          flush=True)
    out["sample"] = evaluate("sample", tr, ev, oracle=False)

    out["visibility_train_loss"] = vis_loss
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(path, results):
    im, d = _canvas(1180, 1330, "S: can the energy rerank completions?",
                    "pairwise accuracy against true masked error · K = 8 · chance 0.5 · "
                    "training and evaluation masks are disjoint")
    l, r = 100, 1060
    colours = {"visibility": BLUE, "quality": GREEN, "inverted": RED,
               "oracle": ORANGE, "smooth": (170, 200, 120)}
    draws = [x["drawing"] for x in results]

    def panel(top, title, series, spread_of):
        bottom = top + 220

        def py(v):
            return bottom - v * (bottom - top)
        for i in range(6):
            v = i / 5
            d.line([l, py(v), r, py(v)], fill=GRID, width=1)
            d.text((l - 10, py(v)), f"{v:.1f}", fill=(150, 158, 172),
                   anchor="rm", font=_font(12))
        d.line([l, py(0.5), r, py(0.5)], fill=(150, 158, 172), width=1)
        d.text((r + 8, py(0.5)), "chance", fill=(190, 196, 208), anchor="lm",
               font=_font(11))
        d.text((l, top - 22), title, fill=(190, 196, 208), font=_font(13))
        group = (r - l) / len(draws)
        for gi, name in enumerate(draws):
            res = next(x for x in results if x["drawing"] == name)
            cx = l + (gi + 0.5) * group
            w = group / (len(series) + 1.0)
            for si, (label, colour, getter, shade) in enumerate(series):
                v = getter(res, name)
                x = cx + (si - (len(series) - 1) / 2) * w
                c = tuple(int(ch * shade) for ch in colour)
                d.rectangle([x - w * 0.42, py(v), x + w * 0.42, bottom], fill=c)
                d.text((x, py(v) - 4), f"{v:.2f}", fill=INK, anchor="mb",
                       font=_font(10))
            d.text((cx, bottom + 11), name, fill=(200, 206, 218), anchor="ma",
                   font=_font(12))
            d.text((cx, bottom + 24), f"error spread {100*spread_of(res, name):.0f}%",
                   fill=DIM, anchor="ma", font=_font(10))

    def at(family, head, sigma=None):
        if sigma is None:
            return lambda rr, name: rr[family]["heads"][head]["pairwise"]
        return lambda rr, name: rr[family][sigma]["heads"][head]["pairwise"]

    def sm(family):
        return lambda rr, name: rr[family]["smooth"]["pairwise"]

    def orc(family):
        return lambda rr, name: rr[family]["oracle"]["pairwise"]

    panel(140, "1 · same mask, fixed temperature (lighter sigma 1.0, darker 2.0)", [
        ("visibility 1.0", BLUE, at("fixed", "visibility", "1.0"), 0.55),
        ("visibility 2.0", BLUE, lambda rr, n: rr["fixed"]["2.0"]["heads"]["visibility"]["pairwise"], 1.0),
        ("quality 1.0", GREEN, at("fixed", "quality", "1.0"), 0.55),
        ("quality 2.0", GREEN, lambda rr, n: rr["fixed"]["2.0"]["heads"]["quality"]["pairwise"], 1.0),
        ("inverted 1.0", RED, at("fixed", "inverted", "1.0"), 0.55),
        ("inverted 2.0", RED, lambda rr, n: rr["fixed"]["2.0"]["heads"]["inverted"]["pairwise"], 1.0),
    ], lambda rr, n: rr["fixed"]["1.0"]["error_spread"])

    panel(420, "2 · same target mask, different amounts visible", [
        ("visibility", BLUE, at("context", "visibility"), 1.0),
        ("quality", GREEN, at("context", "quality"), 1.0),
        ("inverted", RED, at("context", "inverted"), 1.0),
        ("smooth", (170, 200, 120), sm("context"), 1.0),
        ("oracle", ORANGE, orc("context"), 1.0),
    ], lambda rr, n: rr["context"]["error_spread"])

    panel(700, "3 · same target mask, same number visible, different subsets", [
        ("visibility", BLUE, at("swap", "visibility"), 1.0),
        ("quality", GREEN, at("swap", "quality"), 1.0),
        ("inverted", RED, at("swap", "inverted"), 1.0),
        ("smooth", (170, 200, 120), sm("swap"), 1.0),
        ("oracle", ORANGE, orc("swap"), 1.0),
    ], lambda rr, n: rr["swap"]["error_spread"])

    panel(980, "4 · same input, K stochastic forward passes (dropout sampler)", [
        ("visibility", BLUE, at("sample", "visibility"), 1.0),
        ("quality", GREEN, at("sample", "quality"), 1.0),
        ("inverted", RED, at("sample", "inverted"), 1.0),
        ("smooth", (170, 200, 120), sm("sample"), 1.0),
    ], lambda rr, n: rr["sample"]["error_spread"])

    for i, (label, colour) in enumerate((("visibility-trained", BLUE),
                                         ("quality-trained", GREEN),
                                         ("labels inverted", RED),
                                         ("neighbour smoothness", (170, 200, 120)),
                                         ("counts visible cells", ORANGE))):
        x = 100 + i * 205
        d.rectangle([x, 1288, x + 12, 1300], fill=colour)
        d.text((x + 20, 1294), label, fill=INK, anchor="lm", font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drawings", nargs="+", default=[d for d, _ in DRAWINGS])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="runs/energy-selector")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    results = []
    for name, ckpt in DRAWINGS:
        if name not in a.drawings:
            continue
        print(f"[{name}] {ckpt}", flush=True)
        res = run_drawing(name, ckpt, a.steps, a)
        results.append(res)
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump({"config": {"k": K, "sigmas": list(SIGMAS),
                                  "train_masks": N_TRAIN, "eval_masks": N_EVAL,
                                  "ratios": list(RATIOS), "steps": a.steps,
                                  "context_q": list(CONTEXT_Q)},
                       "results": results}, f, indent=2)

    figure(os.path.join(a.out, "energy-selector.png"), results)
    print()
    for res in results:
        print(f"{res['drawing']:6s} fixed1.0 "
              + " ".join(f"{k} {v['pairwise']:.3f}"
                         for k, v in res["fixed"]["1.0"]["heads"].items())
              + f"  oracle n/a")
        print(f"{res['drawing']:6s} fixed2.0 "
              + " ".join(f"{k} {v['pairwise']:.3f}"
                         for k, v in res["fixed"]["2.0"]["heads"].items()))
        for fam in ("context", "swap", "sample"):
            row = (f"{res['drawing']:6s} {fam:7s} "
                   + " ".join(f"{k} {v['pairwise']:.3f}"
                              for k, v in res[fam]["heads"].items())
                   + f"  smooth {res[fam]['smooth']['pairwise']:.3f}")
            if res[fam]["oracle"]:
                row += f"  oracle {res[fam]['oracle']['pairwise']:.3f}"
            print(row)


if __name__ == "__main__":
    main()
