#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to the served model name}"
BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
N_SEEDS="${N_SEEDS:-40}"
CONCURRENCY="${CONCURRENCY:-32}"
OUT="${OUT:-results/pilot.jsonl}"
mkdir -p "$(dirname "$OUT")"
python -m src.experiment \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --model "$MODEL" \
  --n-seeds "$N_SEEDS" \
  --concurrency "$CONCURRENCY" \
  --output "$OUT"
python -m src.analyze "$OUT" --outdir "${ANALYSIS_DIR:-results/analysis}" --bootstrap "${BOOTSTRAP:-5000}"
