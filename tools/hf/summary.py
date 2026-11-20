"""Flatten the ablation result files into one sortable CSV.

    python tools/hf/summary.py runs/ /tmp/summary.csv

Each `results.json` looks like

    {"base": {...}, "table": {axis: {variant: {metric: {"mean": , "sd": }}}}}

so the CSV is one row per (sweep, axis, variant, metric). Written for the
Hugging Face dataset repo, where the raw JSON is the source of truth and this is
the convenience view.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

FIELDS = ["sweep", "axis", "variant", "metric", "mean", "sd"]


def rows(sweep: str, path: Path):
    with open(path) as f:
        table = (json.load(f) or {}).get("table") or {}
    for axis, variants in table.items():
        for variant, metrics in (variants or {}).items():
            for metric, value in (metrics or {}).items():
                if isinstance(value, dict) and "mean" in value:
                    mean, sd = value.get("mean"), value.get("sd")
                elif isinstance(value, (int, float)):
                    mean, sd = value, None
                else:
                    continue
                yield {
                    "sweep": sweep, "axis": axis, "variant": variant,
                    "metric": metric,
                    "mean": "" if mean is None else round(float(mean), 6),
                    "sd": "" if sd is None else round(float(sd), 6),
                }


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "summary.csv")
    files = sorted(src.rglob("results.json"))
    if not files:
        raise SystemExit(f"no results.json under {src}")
    n = 0
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for path in files:
            for row in rows(path.parent.name, path):
                w.writerow(row)
                n += 1
    print(f"{dst}: {n} rows from {len(files)} sweeps")


if __name__ == "__main__":
    main()
