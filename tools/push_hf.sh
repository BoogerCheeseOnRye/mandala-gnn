#!/usr/bin/env bash
# Publish the checkpoint, the ablation sweep and the live cockpit to Hugging Face.
#
#   tools/push_hf.sh            # model + dataset (what is published)
#   tools/push_hf.sh model      # just the weights
#   tools/push_hf.sh dataset    # just the sweeps
#   tools/push_hf.sh space      # just the demo Space (needs PRO; see below)
#   tools/push_hf.sh --stage    # assemble the folders, upload nothing
#
# Needs `hf auth login` (or HF_TOKEN). Files are staged from this repo, so one
# source of truth: nothing is hand-maintained on the Hub.
#
# The Space target is Docker, which Hugging Face now only hosts on cpu-basic for
# PRO accounts (`402` otherwise). Static Spaces are still free, but they cannot
# run this Python server, so the cockpit is not hosted there.
set -euo pipefail

USER_NS="${HF_USER:-BoogerCheeseOnRye}"
MODEL_REPO="$USER_NS/mandala-gnn"
DATA_REPO="$USER_NS/mandala-gnn-ablations"
SPACE_REPO="$USER_NS/mandala-gnn-demo"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF="${HF_CLI:-$ROOT/.venv/bin/hf}"
STAGE="${STAGE:-/tmp/opencode/hf-stage}"

what="${1:-all}"
upload=1
[[ "$what" == "--stage" ]] && { upload=0; what=all; }

cd "$ROOT"

# ---------------------------------------------------------------- staging ----
echo "staging in $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{model,dataset,space}

copy_pkg() {  # $1 = dest, then paths relative to the repo
  local dest="$1"; shift
  for p in "$@"; do
    mkdir -p "$dest/$(dirname "$p")"
    if [[ -d "$ROOT/$p" ]]; then
      cp -r "$ROOT/$p" "$dest/$(dirname "$p")/"
    else
      cp "$ROOT/$p" "$dest/$p"
    fi
  done
  find "$dest" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "$dest" -name '*.pyc' -delete 2>/dev/null || true
}

if [[ "$what" == all || "$what" == model ]]; then
  cp "$ROOT/runs/main-hex/model.pt"     "$STAGE/model/"
  cp "$ROOT/runs/main-hex/metrics.json" "$STAGE/model/"
  cp "$ROOT/runs/main-hex/demo.json"    "$STAGE/model/"
  cp "$ROOT/runs/main-hex/recon.png"    "$STAGE/model/"
  cp "$ROOT/tools/hf/README-model.md"   "$STAGE/model/README.md"
fi

if [[ "$what" == all || "$what" == dataset ]]; then
  d="$STAGE/dataset"
  mkdir -p "$d"/{ablations,metrics,figures}
  for r in abl-hex400 abl-model abl-round abl-none abl-none-round; do
    mkdir -p "$d/ablations/${r#abl-}"
    cp "$ROOT/runs/$r/results.json" "$d/ablations/${r#abl-}/"
    [[ -f "$ROOT/runs/$r/report.md" ]] && cp "$ROOT/runs/$r/report.md" "$d/ablations/${r#abl-}/"
  done
  for r in main-hex main-rings mlp-0 mlp-1 classify; do
    mkdir -p "$d/metrics/$r"
    cp "$ROOT/runs/$r/metrics.json" "$d/metrics/$r/"
    [[ -f "$ROOT/runs/$r/recon.png" ]] && cp "$ROOT/runs/$r/recon.png" "$d/metrics/$r/"
  done
  cp "$ROOT"/runs/mesh-*.png "$d/figures/"
  cp "$ROOT/tools/hf/README-dataset.md" "$d/README.md"
  "$ROOT/.venv/bin/python" "$ROOT/tools/hf/summary.py" "$d/ablations" "$d/summary.csv"
fi

if [[ "$what" == all || "$what" == space ]]; then
  s="$STAGE/space"
  copy_pkg "$s" mandalanet web tools/hf/Dockerfile
  mv "$s/tools/hf/Dockerfile" "$s/Dockerfile"
  rm -rf "$s/tools"
  mkdir -p "$s/drawings" "$s/runs/main-hex"
  cp "$ROOT/drawings/hex-tessellation.jpg" "$s/drawings/"
  cp "$ROOT/runs/main-hex/model.pt" "$ROOT/runs/main-hex/metrics.json" \
     "$ROOT/runs/main-hex/demo.json" "$ROOT/runs/main-hex/recon.png" "$s/runs/main-hex/"
  for r in abl-hex400 abl-model abl-round mlp-0 mlp-1; do
    mkdir -p "$s/runs/$r"
    [[ -f "$ROOT/runs/$r/results.json" ]] && cp "$ROOT/runs/$r/results.json" "$s/runs/$r/"
    [[ -f "$ROOT/runs/$r/metrics.json" ]] && cp "$ROOT/runs/$r/metrics.json" "$s/runs/$r/"
  done
  cp "$ROOT/tools/hf/README-space.md" "$s/README.md"
  cp "$ROOT/docs/img/cockpit.jpg"     "$s/cockpit.jpg"
fi

echo "staged:"; du -sh "$STAGE"/* 2>/dev/null

# --------------------------------------------------------------- uploads ----
if [[ "$upload" == 0 ]]; then
  echo "staged only; nothing uploaded"; exit 0
fi

ensure() {  # $1 = repo id, $2 = type, $3 = sdk (space only)
  local args=(repos create "$1" --type "$2" --public)
  [[ -n "${3:-}" ]] && args+=(--sdk "$3")
  "$HF" "${args[@]}" >/dev/null 2>&1 || true   # fine if it already exists
}

push() {  # $1 = local dir, $2 = repo id, $3 = type, $4 = message
  echo "→ $3 $2"
  "$HF" upload "$2" "$1" . --type "$3" --commit-message "$4"
}

rc=0
if [[ "$what" == all || "$what" == model ]]; then
  ensure "$MODEL_REPO" model
  push "$STAGE/model" "$MODEL_REPO" model \
    "Mandala-GNN: 405k-param checkpoint, metrics and cold-state cache" || rc=1
fi
if [[ "$what" == all || "$what" == dataset ]]; then
  ensure "$DATA_REPO" dataset
  push "$STAGE/dataset" "$DATA_REPO" dataset \
    "Mandala-GNN: full ablation sweep, raw results and per-run metrics" || rc=1
fi
if [[ "$what" == space ]]; then
  ensure "$SPACE_REPO" space docker
  push "$STAGE/space" "$SPACE_REPO" space \
    "Mandala-GNN cockpit: the demo server, the checkpoint and real inference" || rc=1
fi

cat <<EOF

model    https://huggingface.co/$MODEL_REPO
dataset  https://huggingface.co/datasets/$DATA_REPO
EOF
[[ "$what" == space ]] &&
  echo "space    https://huggingface.co/spaces/$SPACE_REPO"
exit $rc
