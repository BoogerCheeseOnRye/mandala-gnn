#!/usr/bin/env bash
# Remaining sweeps (main runs + classification already done).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
STEPS=${STEPS:-200}
W=${W:-3}

echo "== ablation: topology, on the hex drawing =="
$PY -m mandalanet.ablate --axis mesh --drawing drawings/hex-tessellation.jpg \
    --steps "$STEPS" --seeds 2 --workers "$W" --out runs/abl-hex

echo "== ablation: topology, on the star drawing =="
$PY -m mandalanet.ablate --axis mesh --drawing drawings/star-mandala.jpg \
    --steps "$STEPS" --seeds 2 --workers "$W" --out runs/abl-star

echo "== ablation: model knobs, on the hex drawing =="
$PY -m mandalanet.ablate --axis harmonics rbf_bins depth edge_gate \
    --steps "$STEPS" --seeds 2 --workers "$W" --out runs/abl-model
