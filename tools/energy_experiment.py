"""Does an energy head earn its keep? One experiment, end to end.

    .venv/bin/python tools/energy_experiment.py --ckpt runs/main-hex --out runs/energy

The plan being tested (Option A): keep the backbone frozen, learn

    E(h) = sum_i phi(h_i) + sum_(i,j) psi(h_i, h_j, e_ij)

and treat completion as minimisation -- forward pass for the initial state,
then descend E on the hidden nodes.

What this script does:

1. Loads the frozen checkpoint. Nothing is retrained.
2. Builds a bank of states under different masks, keeping the wedge hidden in
   every one of them, and trains the head to rank "saw more" above "saw less".
   No training state ever contains the true wedge, so the wedge test is clean.
3. Asks the three questions that decide whether descent can work at all:
      a. does E prefer the true completion to the model's guess?
      b. does E order states by how much they saw? (its actual objective)
      c. does E order states by completion quality at a fixed mask?
4. Runs both descents -- on h, and on the completion itself -- plus a
   best-of-K sampler that keeps the lowest-energy candidate.
5. Writes results.json and the figures.

The point is not to make the number go up. It is to find out whether it can.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandalanet.data import (IMAGE, patches_to_image, recon_example,  # noqa: E402
                             sector_of, with_mask)
from mandalanet.energy import (EnergyHead, StateBank, refine,  # noqa: E402
                               refine_patches, train_head)
from mandalanet.graph import Graph  # noqa: E402
from mandalanet.model import masked_corr, masked_mse  # noqa: E402
from mandalanet.train import Config, build  # noqa: E402

FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
BG, INK, DIM, GRID = (18, 20, 24), (230, 234, 240), (130, 138, 152), (38, 42, 50)
BLUE, GREEN, ORANGE, RED = ((110, 160, 255), (120, 220, 170),
                            (235, 170, 90), (220, 120, 120))


# --------------------------------------------------------------------------- #
def load(ckpt_dir: str):
    meta = json.load(open(os.path.join(ckpt_dir, "metrics.json")))
    fields = Config.__dataclass_fields__
    cfg = Config(**{k: v for k, v in meta["config"].items() if k in fields})
    cfg.threads = 1
    torch.set_num_threads(1)
    ex = recon_example(cfg.drawing, mesh=cfg.mesh, seed=cfg.seed,
                       circle=cfg.round_patches, n_sectors=cfg.n_sectors,
                       val_sector=cfg.val_sector, mask_ratio=cfg.mask_ratio)
    model = build(cfg, ex.patch_dim + 1, ex.patch_dim)
    ck = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return cfg, ex, model


def random_mask(n: int, pool: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = max(1, int(round(ratio * pool.numel())))
    idx = pool[torch.randperm(pool.numel(), generator=g)[:k]]
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m


def pearson(xs, ys) -> float:
    x = torch.as_tensor(xs, dtype=torch.float32)
    y = torch.as_tensor(ys, dtype=torch.float32)
    if x.numel() < 2:
        return 0.0
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-9))


def compose(pred, ex, hidden) -> torch.Tensor:
    pred_img = patches_to_image(pred, ex.mesh.pos_px, IMAGE)
    mask_img = patches_to_image(hidden.float().unsqueeze(-1), ex.mesh.pos_px, IMAGE)
    return torch.where(mask_img > 0.5, pred_img, ex.img)


def r_mse(pred, ex, mask):
    return (float(masked_corr(pred, ex.target, mask)),
            float(masked_mse(pred, ex.target, mask)))


# --------------------------------------------------------------------------- #
# figures: PIL only, same dark palette as the live cockpit
# --------------------------------------------------------------------------- #
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:                                   # noqa: TRY003
        return ImageFont.load_default()


def _canvas(w: int, h: int, title: str, sub: str):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    d.text((24, 18), title, fill=INK, font=_font(17, True))
    d.text((24, 42), sub, fill=DIM, font=_font(12))
    return im, d


def _axes(d, box, x0, x1, y0, y1, xfmt="{:.3f}", yfmt="{:+.0f}"):
    l, t, r, b = box
    d.rectangle([l, t, r, b], outline=(60, 66, 78), width=1)
    for frac in (0.0, 0.5, 1.0):
        xx = l + frac * (r - l)
        d.line([xx, t, xx, b], fill=GRID, width=1)
        d.text((xx, b + 14), xfmt.format(x0 + frac * (x1 - x0)),
               fill=(150, 158, 172), anchor="ma", font=_font(12))
        yy = b - frac * (b - t)
        d.line([l, yy, r, yy], fill=GRID, width=1)
        d.text((l - 10, yy), yfmt.format(y0 + frac * (y1 - y0)),
               fill=(150, 158, 172), anchor="rm", font=_font(12))


def curve_png(path: str, track: list[dict], baseline: dict, title: str) -> str:
    """Energy and wedge r against descent step: two panels, no matplotlib."""
    im, d = _canvas(1040, 560, title,
                    "backbone frozen · only the 25 hidden wedge cells move")
    pad_l, pad_r, pad_t, pad_b = 74, 26, 104, 56
    panels = [("energy", "energy  (lower is better)", BLUE),
              ("wedge_corr", "held-out wedge r", GREEN)]
    pw = (1040 - pad_l - pad_r - 48) // 2
    for pi, (key, label, colour) in enumerate(panels):
        x0 = pad_l + pi * (pw + 48)
        y0, x1, y1 = pad_t, pad_l + pi * (pw + 48) + pw, 560 - pad_b
        vals = [p[key] for p in track]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        pad = (hi - lo) * 0.12
        lo, hi = lo - pad, hi + pad
        _axes(d, (x0, y0, x1, y1), 0, len(vals) - 1, lo, hi,
              xfmt="{:.0f}", yfmt="{:.4f}" if abs(hi) < 100 else "{:.3g}")
        d.text((x0, y0 - 26), label, fill=(190, 196, 208), font=_font(14))
        pts = [(x0 + (x1 - x0) * (i / max(1, len(vals) - 1)),
                y1 - (v - lo) / (hi - lo) * (y1 - y0)) for i, v in enumerate(vals)]
        d.line(pts, fill=colour, width=2, joint="curve")
        d.ellipse([pts[0][0] - 4, pts[0][1] - 4, pts[0][0] + 4, pts[0][1] + 4],
                  fill=BG, outline=colour, width=2)
        by = y1 - (baseline[key] - lo) / (hi - lo) * (y1 - y0)
        d.line([x0, by, x1, by], fill=RED, width=1)
        d.text((x0 + 8, by - 18), "no refinement", fill=RED, font=_font(12))
        d.text((x1, y1 + 38), "descent steps", fill=DIM, anchor="ra",
               font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def mask_png(path: str, masked: list[dict], corr: float, title: str) -> str:
    """Energy and r against how many cells are hidden: the signal the head did
    learn, plotted next to the one we wanted it to learn."""
    im, d = _canvas(1040, 590, title, "same drawing · more hidden cells to the right")
    panels = [("energy", "energy", BLUE), ("corr", "r on the hidden cells", GREEN)]
    l, r, t, b = 96, 28, 108, 476
    pw = (1040 - l - r - 48) // 2
    xs = [row["hidden"] for row in masked]
    for pi, (key, label, colour) in enumerate(panels):
        x0 = l + pi * (pw + 48)
        x1 = x0 + pw
        ys = [row[key] for row in masked]
        lo, hi = min(ys), max(ys)
        gap = (hi - lo) * 0.15 or 1.0
        lo, hi = lo - gap, hi + gap
        d.rectangle([x0, t, x1, b], outline=(60, 66, 78), width=1)
        for xv in xs:
            xx = x0 + (xv - min(xs)) / (max(xs) - min(xs)) * (x1 - x0)
            d.line([xx, t, xx, b], fill=GRID, width=1)
            d.text((xx, b + 14), f"{xv}", fill=(150, 158, 172), anchor="ma",
                   font=_font(12))
        for frac in (0.0, 0.5, 1.0):
            yy = b - frac * (b - t)
            vy = lo + frac * (hi - lo)
            d.line([x0, yy, x1, yy], fill=GRID, width=1)
            d.text((x0 - 10, yy), f"{vy / 1000:.1f}k" if abs(vy) >= 1000
                   else f"{vy:.3f}", fill=(150, 158, 172), anchor="rm",
                   font=_font(12))
        d.text((x0, t - 26), label, fill=(190, 196, 208), font=_font(14))
        pts = [(x0 + (x - min(xs)) / (max(xs) - min(xs)) * (x1 - x0),
                b - (v - lo) / (hi - lo) * (b - t)) for x, v in zip(xs, ys)]
        d.line(pts, fill=colour, width=2, joint="curve")
        for px, py in pts:
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=colour)
        d.text(((x0 + x1) // 2, b + 40), "cells hidden", fill=DIM,
               anchor="ma", font=_font(12))
    d.text((l, 552), f"energy vs r on those five masks: r = {corr:+.2f} — "
                     f"energy climbs, r stays in one band",
           fill=(190, 196, 208), font=_font(13))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def quality_png(path: str, sweep: list[dict], title: str) -> str:
    """Energy against completion quality, with the mask held fixed.

    x is the correlation of the completion with the truth, y is the energy of
    the state that completion produces. A usable landscape is a falling line:
    better redraw, lower energy. This one rises.
    """
    im, d = _canvas(900, 560, title,
                    "same mask, same support · a linear blend from the guess to the truth")
    W = 900
    l, r, t, b = 104, W - 44, 120, 446
    xs = [row["corr"] for row in sweep]
    ys = [row["energy"] for row in sweep]
    x0, x1 = min(xs) - 0.012, max(xs) + 0.012
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.3, 0.5)
    y0, y1 = lo - pad, hi + pad

    def px(x):
        return l + (x - x0) / (x1 - x0) * (r - l)

    def py(y):
        return b - (y - y0) / (y1 - y0) * (b - t)

    # legend, under the subtitle and above the plot
    d.ellipse([l, 78, l + 9, 87], fill=BLUE)
    d.text((l + 16, 82), "the model's guess", fill=BLUE, anchor="lm", font=_font(13))
    d.text((l + 150, 82), f"r {sweep[0]['corr']:.3f} · E {sweep[0]['energy']:.1f}",
           fill=(150, 158, 172), anchor="lm", font=_font(12))
    d.ellipse([l + 380, 78, l + 389, 87], fill=ORANGE)
    d.text((l + 396, 82), "the true patches", fill=ORANGE, anchor="lm", font=_font(13))
    d.text((l + 520, 82), f"r {sweep[-1]['corr']:.3f} · E {sweep[-1]['energy']:.1f}",
           fill=(150, 158, 172), anchor="lm", font=_font(12))

    d.rectangle([l, t, r, b], outline=(60, 66, 78), width=1)
    for frac in (0.0, 0.5, 1.0):
        xx = l + frac * (r - l)
        d.line([xx, t, xx, b], fill=GRID, width=1)
        d.text((xx, b + 12), f"{x0 + frac * (x1 - x0):.2f}", fill=(150, 158, 172),
               anchor="ma", font=_font(12))
        yy = b - frac * (b - t)
        d.line([l, yy, r, yy], fill=GRID, width=1)
        d.text((l - 10, yy), f"{y0 + frac * (y1 - y0):.1f}", fill=(150, 158, 172),
               anchor="rm", font=_font(12))

    d.line([(px(x), py(y)) for x, y in zip(xs, ys)], fill=BLUE, width=2,
           joint="curve")
    for i, (row, x, y) in enumerate(zip(sweep, xs, ys)):
        colour = ORANGE if i == len(sweep) - 1 else BLUE
        d.ellipse([px(x) - 4, py(y) - 4, px(x) + 4, py(y) + 4], fill=colour)
        if i in (0, len(sweep) // 2, len(sweep) - 1):
            anchor = "mb" if i < len(sweep) - 1 else "rb"
            dx, dy = (6, -10) if i == 0 else ((0, -12) if i < len(sweep) - 1 else (-8, -6))
            d.text((px(x) + dx, py(y) + dy), f"{row['blend']:.0%} truth",
                   fill=(150, 158, 172), anchor=anchor, font=_font(11))

    d.text((l, b + 42), "energy  ->   (whole range: 0.08% of E)",
           fill=DIM, font=_font(12))
    d.text((r, b + 42), "wedge r on the completion  ->", fill=DIM, anchor="ra",
           font=_font(12))
    d.text((l, 514), "better completion  ->  higher energy:", fill=RED, font=_font(13))
    d.text((l + 330, 514), "the head prefers the guess", fill=(190, 196, 208),
           font=_font(13))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def triptych(path: str, panels, labels, title: str, sub: str,
             scale: int = 3, gap: int = 10, pad: int = 24) -> str:
    """Three drawings side by side, captioned, on the same dark ground."""
    from PIL import Image, ImageDraw

    def grey(img):
        arr = (img.detach().squeeze().clamp(0, 1) * 255).byte().numpy()
        return Image.fromarray(arr).convert("L")

    ims = [grey(p) for p in panels]
    w, h = ims[0].size
    tw, th = w * scale, h * scale
    W = pad * 2 + len(ims) * tw + (len(ims) - 1) * gap
    H = 66 + th + 34
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 16), title, fill=INK, font=_font(17, True))
    d.text((pad, 40), sub, fill=DIM, font=_font(12))
    for i, (panel, label) in enumerate(zip(ims, labels)):
        x, y = pad + i * (tw + gap), 62
        im.paste(panel.resize((tw, th), Image.LANCZOS).convert("RGB"), (x, y))
        d.rectangle([x, y, x + tw - 1, y + th - 1], outline=(60, 66, 78), width=1)
        d.text((x, y + th + 9), label, fill=(190, 196, 208), font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main-hex")
    p.add_argument("--out", default="runs/energy")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--descent", type=int, default=25)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg, ex, model = load(a.ckpt)
    n = ex.graph.num_nodes
    wedge = ex.wedge_mask
    feat = model.geom(ex.graph)
    edges = ex.graph.edge_index
    nonwedge = (~wedge).nonzero(as_tuple=True)[0]

    t0 = time.time()

    # ---- 1. the head, trained on states that never see the wedge ---------- #
    bank = StateBank()
    with torch.no_grad():
        for i in range(8):
            m = wedge | random_mask(n, nonwedge, 0.04 + 0.01 * i, 10 + i)
            bank.pos.append(model(with_mask(ex, m))["node"].detach().clone())
        for i, r in enumerate((0.45, 0.55, 0.65, 0.75)):
            for j in range(3):
                m = wedge | random_mask(n, nonwedge, r, 1000 + 10 * i + j)
                bank.neg.append(model(with_mask(ex, m))["node"].detach().clone())

    heads = {}
    for name, edge_term in (("full", True), ("node-only", False)):
        head = EnergyHead(dim=cfg.dim, edge_dim=model.geom.out_dim,
                          edge_term=edge_term)
        hist = train_head(head, bank, feat, edges, steps=a.steps)
        heads[name] = (head, hist)
    head = heads["full"][0]
    print(f"bank {len(bank.pos)}/{len(bank.neg)} · trained in {time.time()-t0:.1f}s")

    def energy_of(h, edge_feature=None):
        with torch.no_grad():
            return float(head(h, feat if edge_feature is None else edge_feature,
                              edges)["total"])

    def drawing(patches):
        return Graph(torch.cat([patches, torch.zeros_like(patches[:, :1])], -1),
                     ex.mesh.pos, ex.mesh.edge_index, ex.mesh.edge_type)

    def energy_drawing(patches):
        with torch.no_grad():
            return energy_of(model(drawing(patches))["node"])

    # ---- 2a. does E prefer the answer? ------------------------------------ #
    with torch.no_grad():
        guess = model(with_mask(ex, wedge))["node_pred"].detach()
    completed = ex.target.clone()
    completed[wedge] = guess[wedge]
    e_guess = energy_drawing(completed)
    e_true = energy_drawing(ex.target)
    prefers_truth = {"pred": e_guess, "true": e_true,
                     "delta": e_true - e_guess,
                     "delta_pct": (e_true - e_guess) / abs(e_guess) * 100}
    print(f"E(guess) {e_guess:.1f}  E(true) {e_true:.1f} "
          f"({prefers_truth['delta_pct']:+.2f}%)")

    # ---- 2b. does E order states by how much they saw? --------------------- #
    gen = torch.Generator().manual_seed(3)
    wins, trials = 0, 24
    for _ in range(trials):
        idx = torch.randperm(n, generator=gen)
        a_mask = torch.zeros(n, dtype=torch.bool)
        b_mask = torch.zeros(n, dtype=torch.bool)
        a_mask[idx[:int(0.2 * n)]] = True
        b_mask[idx[:int(0.5 * n)]] = True
        ea = energy_drawing(ex.target * (~a_mask).unsqueeze(-1)
                            + guess * a_mask.unsqueeze(-1))
        eb = energy_drawing(ex.target * (~b_mask).unsqueeze(-1)
                            + guess * b_mask.unsqueeze(-1))
        wins += int(ea < eb)
    visibility = {"accuracy": wins / trials, "wins": wins, "n": trials}

    # ---- 2c. does E order completions by quality at a fixed mask? ---------- #
    # corrupt only what the model can see, keep the wedge hidden exactly as at
    # test time: a useful energy should rise as the visible input degrades
    noise_sweep = []
    for sigma in (0.0, 0.05, 0.10, 0.20, 0.35):
        g = torch.Generator().manual_seed(4242)
        noisy = ex.target * (1.0 + sigma * torch.randn(ex.target.shape, generator=g))
        feats = torch.cat([noisy * (~wedge).float().unsqueeze(-1),
                           wedge.float().unsqueeze(-1)], dim=-1)
        gg = Graph(feats, ex.mesh.pos, ex.mesh.edge_index, ex.mesh.edge_type)
        with torch.no_grad():
            out = model(gg)
            c, m = r_mse(out["node_pred"], ex, wedge)
        noise_sweep.append({"noise": sigma, "energy": energy_of(out["node"]),
                            "corr": c, "mse": m})
    noise_r = pearson([v["energy"] for v in noise_sweep],
                      [v["corr"] for v in noise_sweep])

    # the mask sweep: energy and r as more cells are hidden
    masked = []
    for k in (0, 20, 40, 60, 80):
        extra = random_mask(n, nonwedge, k / nonwedge.numel(), 500 + k)
        m = wedge | extra
        with torch.no_grad():
            out = model(with_mask(ex, m))
            c, ms = r_mse(out["node_pred"], ex, m)
            e = energy_of(out["node"])
        masked.append({"extra": k, "hidden": int(m.sum()), "energy": e,
                       "corr": c, "mse": ms})
    mask_r = pearson([v["energy"] for v in masked], [v["corr"] for v in masked])

    # a clean ordering test: blend the guess towards the truth and ask whether
    # E falls monotonically. Same mask, same support, only quality changes.
    quality_sweep = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        patches = ex.target.clone()
        patches[wedge] = (1 - t) * guess[wedge] + t * ex.target[wedge]
        with torch.no_grad():
            out = model(drawing(patches))
            c, m = r_mse(patches, ex, wedge)
        quality_sweep.append({"blend": t, "energy": energy_of(out["node"]),
                              "corr": c, "mse": m})

    # ---- 3. h-descent, tuned on val --------------------------------------- #
    val0 = model(with_mask(ex, ex.val_mask))["node_pred"].detach()
    val_base = dict(zip(("corr", "mse"), r_mse(val0, ex, ex.val_mask)))
    tune = []
    for lr in (0.01, 0.05, 0.2):
        for an in (0.0, 1e-2, 1e-1):
            out = refine(model, head, ex, ex.val_mask, steps=a.descent,
                         lr=lr, anchor=an)
            c, m = r_mse(out["pred"], ex, ex.val_mask)
            tune.append({"lr": lr, "anchor": an, "corr": c, "mse": m,
                         "energy_drop": out["energy0"] - out["energy"]})
    best = max(tune, key=lambda r: r["corr"])

    plain = model(with_mask(ex, wedge))["node_pred"].detach()
    base_corr, base_mse = r_mse(plain, ex, wedge)

    runs = {}
    for name, (h, _) in heads.items():
        t = time.time()
        out = refine(model, h, ex, wedge, steps=a.descent, lr=best["lr"],
                     anchor=best["anchor"], track=True)
        ms = (time.time() - t) * 1000
        c, m = r_mse(out["pred"], ex, wedge)
        runs[name] = {"corr": c, "mse": m, "energy0": out["energy0"],
                      "energy": out["energy"], "best_step": out["best_step"],
                      "delta_norm": out["delta_norm"],
                      "ms_per_step": round(ms / a.descent, 2),
                      "track": out["track"], "pred": out["pred"]}

    # ---- 4. patch descent: the version that can change the redraw --------- #
    patch_sweep, patch_preds = [], {}
    for lr in (0.005, 0.01, 0.02, 0.05, 0.15):
        t = time.time()
        out = refine_patches(model, head, ex, wedge, steps=10, lr=lr,
                             anchor=0.01, track=True)
        ms = (time.time() - t) * 1000
        c, m = r_mse(out["pred"], ex, wedge)
        patch_sweep.append({"lr": lr, "corr": c, "mse": m,
                            "energy0": out["energy0"], "energy": out["energy"],
                            "ms": round(ms, 1), "track": out["track"]})
        patch_preds[lr] = out["pred"]
        print(f"patch descent lr={lr}: r {c:.4f} energy "
              f"{out['energy0']:.0f}->{out['energy']:.0f}")

    # ---- 5. best-of-K sampling, with an error bar ------------------------- #
    def select(mask, sigma: float, K: int, seed: int = 0):
        g = with_mask(ex, mask)
        with torch.no_grad():
            out = model(g)
            h = out["node"]
            base_e = energy_of(h)
        gen = torch.Generator().manual_seed(seed)
        best_e, best_pred = base_e, out["node_pred"]
        with torch.no_grad():
            for _ in range(K):
                hh = h.clone()
                hh[mask] = hh[mask] + sigma * torch.randn(hh[mask].shape,
                                                          generator=gen)
                e = float(head(hh, feat, edges)["total"])
                if e < best_e:
                    best_e, best_pred = e, model.recon_head(hh)
        return best_pred, base_e, best_e

    sel_tune = []
    for sigma in (0.05, 0.1, 0.2, 0.4, 0.8):
        pred, e0, e1 = select(ex.val_mask, sigma, 16, seed=11)
        c, m = r_mse(pred, ex, ex.val_mask)
        sel_tune.append({"sigma": sigma, "corr": c, "mse": m})
    sel_sigma = max(sel_tune, key=lambda r: r["corr"])["sigma"]

    reps, picks = [], 0
    for seed in range(8):
        pred, e0, e1 = select(wedge, sel_sigma, 8, seed=100 + seed)
        c, m = r_mse(pred, ex, wedge)
        reps.append({"seed": 100 + seed, "corr": c, "mse": m})
        picks += int(e1 < e0)
    rc = torch.tensor([r["corr"] for r in reps])
    rm = torch.tensor([r["mse"] for r in reps])
    selection = {"sigma": sel_sigma, "K": 8, "tuning": sel_tune,
                 "repeats": len(reps),
                 "corr_mean": float(rc.mean()), "corr_sd": float(rc.std()),
                 "mse_mean": float(rm.mean()), "mse_sd": float(rm.std()),
                 "delta_corr": float(rc.mean()) - base_corr,
                 "delta_mse": float(rm.mean()) - base_mse,
                 "runs": reps, "found_lower": picks}

    # ---- 6. per-node: is it an uncertainty map? --------------------------- #
    per_node = {}
    for name in heads:
        for label, mask in (("wedge", wedge), ("val", ex.val_mask)):
            with torch.no_grad():
                out = model(with_mask(ex, mask))
                e = heads[name][0](out["node"], feat, edges)["per_node"]
                err = ((out["node_pred"] - ex.target) ** 2).mean(-1)
            per_node[f"{name}_{label}"] = pearson(e[mask], err[mask])

    # ---- 7. other masks, so the wedge is the only claim ------------------- #
    probe = []
    for label, mask in (("sector 3", sector_of(ex.mesh.pos, cfg.n_sectors) == 3),
                        ("random 20%", random_mask(n, torch.arange(n), 0.2, 99)),
                        ("random 50%", random_mask(n, torch.arange(n), 0.5, 98))):
        p0 = model(with_mask(ex, mask))["node_pred"].detach()
        c0, m0 = r_mse(p0, ex, mask)
        out = refine(model, head, ex, mask, steps=a.descent, lr=best["lr"],
                     anchor=best["anchor"])
        c1, m1 = r_mse(out["pred"], ex, mask)
        probe.append({"mask": label, "cells": int(mask.sum()),
                      "corr": c0, "corr_refined": c1, "mse": m0,
                      "mse_refined": m1,
                      "energy_drop": out["energy0"] - out["energy"]})

    # ---- 8. figures + results.json ---------------------------------------- #
    track = runs["full"]["track"]
    curve_png(os.path.join(a.out, "energy-curve.png"), track,
              {"energy": runs["full"]["energy0"], "wedge_corr": base_corr},
              "Descending h: the energy falls, the redraw does not improve")
    mask_png(os.path.join(a.out, "energy-mask.png"), masked, mask_r,
             "What the head did learn: more hidden cells, higher energy")
    quality_png(os.path.join(a.out, "energy-quality.png"), quality_sweep,
                "Does the energy prefer the right answer?")
    triptych(os.path.join(a.out, "energy-recon.png"),
             [ex.img, compose(plain, ex, wedge), compose(runs["full"]["pred"], ex, wedge)],
             ["the drawing", "no refinement", "h-descent · 25 steps"],
             "Descending h moves the redraw nowhere",
             "the soft patch is the held-out wedge · only its 25 cells are allowed to move")
    lrs = sorted(patch_preds)
    triptych(os.path.join(a.out, "energy-collapse.png"),
             [compose(plain, ex, wedge), compose(patch_preds[lrs[1]], ex, wedge),
              compose(patch_preds[lrs[-1]], ex, wedge)],
             ["no refinement", "10 steps · lr 0.01", "10 steps · lr 0.15"],
             "Descending the completion: energy falls, the drawing collapses",
             "every step lowers E · the wedge dissolves into noise while the score improves")

    result = {
        "checkpoint": a.ckpt,
        "config": {"head_steps": a.steps, "descent": a.descent,
                   "lr": best["lr"], "anchor": best["anchor"],
                   "tuned_on": "val_mask"},
        "params": {"full": sum(p.numel() for p in heads["full"][0].parameters()),
                   "node_only": sum(p.numel() for p in heads["node-only"][0].parameters()),
                   "backbone": sum(p.numel() for p in model.parameters())},
        "baseline": {"corr": base_corr, "mse": base_mse,
                     "cells": int(wedge.sum())},
        "baseline_val": val_base,
        "prefers_truth": prefers_truth,
        "visibility": visibility,
        "noise_sweep": noise_sweep,
        "quality_sweep": quality_sweep,
        "noise_r": noise_r,
        "masked": masked,
        "mask_r": mask_r,
        "tuning": tune,
        "refined": {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                    for k, v in runs.items()},
        "patch_descent": {"anchor": 0.01, "steps": 10, "sweep": patch_sweep},
        "selection": selection,
        "per_node": per_node,
        "probe": probe,
        "train_loss": {"full": heads["full"][1][-1],
                       "node-only": heads["node-only"][1][-1]},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"full": heads["full"][0].state_dict(),
                "node_only": heads["node-only"][0].state_dict(),
                "config": {"dim": cfg.dim, "edge_dim": model.geom.out_dim,
                           "head_steps": a.steps, "lr": best["lr"],
                           "anchor": best["anchor"]}},
               os.path.join(a.out, "head.pt"))

    print(json.dumps({k: result[k] for k in
                      ("baseline", "prefers_truth", "visibility", "mask_r",
                       "noise_r", "per_node")}, indent=2))
    print("h-descent:", {k: (round(v["corr"], 4), round(v["mse"], 5))
                         for k, v in runs.items()})
    print("selection:", {k: round(v, 4) for k, v in
                         {"corr_mean": selection["corr_mean"],
                          "delta_corr": selection["delta_corr"],
                          "sd": selection["corr_sd"]}.items()})


if __name__ == "__main__":
    main()
