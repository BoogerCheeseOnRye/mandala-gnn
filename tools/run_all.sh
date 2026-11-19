#!/usr/bin/env bash
# Everything, start to finish, on the phone CPU. ~30 min for the full sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY tests/smoke.py

$PY -m tools.viz --drawing drawings/star-mandala.jpg --mesh rings --out runs/mesh-rings.png
$PY -m tools.viz --drawing drawings/hex-tessellation.jpg --mesh hex --out runs/mesh-hex.png
$PY -m tools.viz --drawing drawings/hub-chart.jpg --mesh chart --out runs/mesh-chart.png

./tools/experiments.sh
