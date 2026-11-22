"""Compare the energy experiment across the three drawings.

    .venv/bin/python tools/energy_compare.py \
        --runs runs/energy runs/energy-rings runs/energy-chart --out runs/energy

Reads one `results.json` per checkpoint, prints the rows the page quotes, and
draws the two panels that decide whether the single-drawing result replicates:

  left  -- energy along the quality axis (guess -> truth), as a percentage of
           each drawing's own energy, because the three heads share no scale;
  right -- the visibility ranking, 24 pairs per drawing, chance at 0.5.

Writes `summary.json` next to the figure so the page has one machine-readable
source for all three drawings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.energy_experiment import (BG, BLUE, DIM, GREEN, GRID, INK,  # noqa: E402
                                     ORANGE, _canvas, _font)

LABELS = {
    "main-hex": ("hex tessellation", BLUE),
    "main-rings": ("star mandala", GREEN),
    "main-chart": ("hub chart", ORANGE),
}


def summary_row(result: dict) -> dict:
    name = os.path.basename(result["checkpoint"].rstrip("/"))
    label, _ = LABELS.get(name, (name, BLUE))
    blend = result["quality_sweep"]
    first, last = blend[0], blend[-1]
    refined = result["refined"]
    patch = result["patch_descent"]["sweep"]
    return {
        "drawing": label,
        "checkpoint": result["checkpoint"],
        "cells": result["baseline"]["cells"],
        "baseline_corr": result["baseline"]["corr"],
        "baseline_mse": result["baseline"]["mse"],
        "truth_delta_pct": result["prefers_truth"]["delta_pct"],
        "blend_end_pct": (last["energy"] - first["energy"]) / abs(first["energy"]) * 100,
        "visibility": result["visibility"]["accuracy"],
        "h_descent_corr": refined["full"]["corr"],
        "h_descent_delta": refined["full"]["corr"] - result["baseline"]["corr"],
        "select_delta": result["selection"]["delta_corr"],
        "select_sd": result["selection"]["corr_sd"],
        "patch_best_corr": patch[0]["corr"],
        "patch_worst_corr": patch[-1]["corr"],
        "per_node_wedge": result["per_node"]["full_wedge"],
        "seconds": result["seconds"],
    }


def replication_png(path: str, rows: list[dict], results: list[dict]) -> str:
    im, d = _canvas(1080, 600, "Does the single-drawing result replicate?",
                    "two questions, three drawings, same protocol")
    # ---- left: energy along the quality axis ----------------------------- #
    l, r, t, b = 96, 500, 118, 470
    d.rectangle([l, t, r, b], outline=(60, 66, 78), width=1)

    series = []
    for label, colour, result in zip((row["drawing"] for row in rows),
                                     (c for _, c in (LABELS[os.path.basename(
                                         rs["checkpoint"].rstrip("/"))]
                                         for rs in results)),
                                     results):
        sweep = result["quality_sweep"]
        base = sweep[0]["energy"]
        series.append((label, colour, [(q["corr"], (q["energy"] - base) / abs(base) * 100)
                                       for q in sweep]))

    xs = [x for _, _, pts in series for x, _ in pts]
    ys = [y for _, _, pts in series for _, y in pts]
    x0, x1 = min(xs) - 0.02, max(xs) + 0.01
    y0, y1 = min(ys) - 0.05, max(ys) + 0.05

    def px(x):
        return l + (x - x0) / (x1 - x0) * (r - l)

    def py(y):
        return b - (y - y0) / (y1 - y0) * (b - t)

    for frac in (0.0, 0.5, 1.0):
        xx = l + frac * (r - l)
        d.line([xx, t, xx, b], fill=GRID, width=1)
        d.text((xx, b + 14), f"{x0 + frac * (x1 - x0):.2f}", fill=(150, 158, 172),
               anchor="ma", font=_font(12))
        yy = b - frac * (b - t)
        d.line([l, yy, r, yy], fill=GRID, width=1)
        d.text((l - 10, yy), f"{y0 + frac * (y1 - y0):+.2f}%", fill=(150, 158, 172),
               anchor="rm", font=_font(12))
    d.line([l, py(0.0), r, py(0.0)], fill=(80, 88, 102), width=1)

    for i, (label, colour, pts) in enumerate(series):
        d.line([(px(x), py(y)) for x, y in pts], fill=colour, width=2, joint="curve")
        for x, y in pts:
            d.ellipse([px(x) - 3, py(y) - 3, px(x) + 3, py(y) + 3], fill=colour)
        d.text((l + 8, t + 12 + 20 * i), f"● {label}", fill=colour, font=_font(13))
    d.text((l, t - 22), "energy along the quality axis (guess → truth)",
           fill=(190, 196, 208), font=_font(13))
    d.text((r, b + 40), "wedge r of the completion  ->", fill=DIM, anchor="ra",
           font=_font(12))

    # ---- right: visibility ranking --------------------------------------- #
    l2, r2, t2, b2 = 620, 1030, 118, 470
    d.rectangle([l2, t2, r2, b2], outline=(60, 66, 78), width=1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = b2 - frac * (b2 - t2)
        d.line([l2, yy, r2, yy], fill=GRID, width=1)
        d.text((l2 - 10, yy), f"{frac:.2f}", fill=(150, 158, 172), anchor="rm",
               font=_font(12))
    chance = b2 - 0.5 * (b2 - t2)
    d.line([l2, chance, r2, chance], fill=(220, 120, 120), width=1)

    chance_label = (l2 + 6, chance - 16)
    bar = (r2 - l2) / len(rows)
    for i, (row, (label, colour, _)) in enumerate(zip(rows, series)):
        cx = l2 + (i + 0.5) * bar
        h = row["visibility"] * (b2 - t2)
        d.rectangle([cx - 26, b2 - h, cx + 26, b2], fill=colour)
        d.text((cx, b2 - h - 18), f"{row['visibility']:.2f}", fill=INK,
               anchor="ma", font=_font(13))
        d.text((cx, b2 + 14), label, fill=(150, 158, 172), anchor="ma",
               font=_font(11))
    d.text(chance_label, "chance", fill=(220, 120, 120), font=_font(11))
    d.text((l2, t2 - 42), "visibility ranking, 24 pairs", fill=(190, 196, 208),
           font=_font(13))

    d.text((96, 556), "Left: better redraw should mean lower energy. The whole vertical axis spans "
           "half a percent,", fill=DIM, font=_font(12))
    d.text((96, 574), "and two of the three drawings move the wrong way — there is no quality axis to descend.",
           fill=DIM, font=_font(12))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+",
                   default=["runs/energy", "runs/energy-rings", "runs/energy-chart"])
    p.add_argument("--out", default="runs/energy")
    a = p.parse_args()

    results = [json.load(open(os.path.join(r, "results.json"))) for r in a.runs]
    rows = [summary_row(r) for r in results]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"drawings": rows}, f, indent=2)
    replication_png(os.path.join(a.out, "energy-replication.png"), rows, results)

    print(f"{'drawing':18s} {'cells':>5s} {'base r':>7s} {'truth':>7s} "
          f"{'vis':>5s} {'Δr descend':>10s} {'Δr select':>9s} {'patch r':>13s}")
    for row in rows:
        print(f"{row['drawing']:18s} {row['cells']:5d} {row['baseline_corr']:7.3f} "
              f"{row['truth_delta_pct']:+6.2f}% {row['visibility']:5.2f} "
              f"{row['h_descent_delta']:+10.4f} {row['select_delta']:+9.4f} "
              f"{row['patch_best_corr']:.3f}→{row['patch_worst_corr']:.3f}")


if __name__ == "__main__":
    main()
