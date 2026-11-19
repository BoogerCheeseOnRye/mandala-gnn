#!/usr/bin/env bash
# Final sweeps, at the step counts the main runs use.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

echo "== A: topology @400, hex drawing =="
$PY -m mandalanet.ablate --axis mesh --drawing drawings/hex-tessellation.jpg \
    --steps 400 --seeds 2 --workers 3 --out runs/abl-hex400

echo "== B: round patches, hex vs grid @400 =="
$PY -m mandalanet.ablate --axis mesh --drawing drawings/hex-tessellation.jpg \
    --steps 400 --seeds 2 --workers 3 --out runs/abl-round --round-patches \
    --only hex grid rewired none

echo "== C: model knobs @300, hex drawing =="
$PY -m mandalanet.ablate --axis harmonics depth edge_gate \
    --drawing drawings/hex-tessellation.jpg --steps 300 --seeds 2 --workers 3 \
    --out runs/abl-model
