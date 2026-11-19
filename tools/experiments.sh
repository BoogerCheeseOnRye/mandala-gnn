#!/usr/bin/env bash
# The full experiment set: main runs, classification, ablations.
# ~45 min on a phone CPU (8 cores); each run is ~2 min at 400 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
W=${W:-3}

echo "== main: hex tessellation on the honeycomb =="
$PY -m mandalanet.train --drawing drawings/hex-tessellation.jpg --mesh hex \
    --steps 400 --out runs/main-hex

echo "== main: star mandala on its spoke lattice =="
$PY -m mandalanet.train --drawing drawings/star-mandala.jpg --mesh rings \
    --steps 400 --out runs/main-rings

echo "== classification: which drawing is this? =="
$PY -m mandalanet.train --task classify --steps 150 --out runs/classify

echo "== A: topology @400, hex drawing =="
$PY -m mandalanet.ablate --axis mesh --drawing drawings/hex-tessellation.jpg \
    --steps 400 --seeds 2 --workers "$W" --out runs/abl-hex400

echo "== B: round patches, hex vs grid @400 =="
$PY -m mandalanet.ablate --axis mesh --drawing drawings/hex-tessellation.jpg \
    --steps 400 --seeds 2 --workers "$W" --out runs/abl-round --round-patches \
    --only hex grid rewired none

echo "== C: model knobs @300, hex drawing =="
$PY -m mandalanet.ablate --axis harmonics depth edge_gate \
    --drawing drawings/hex-tessellation.jpg --steps 300 --seeds 2 --workers "$W" \
    --out runs/abl-model
