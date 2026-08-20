#!/usr/bin/env bash
set -euo pipefail

STEPS=(1000 2000 4000 8000 16000 32000 48000 64000 80000 96000 112000 128000 143000)
OUT=${OUT:-artifacts/checkpoints}
mkdir -p "$OUT"

for STEP in "${STEPS[@]}"; do
  python src/extract_checkpoint.py \
    --step "$STEP" \
    --num-examples "${NUM_EXAMPLES:-1000}" \
    --positions-per-text "${POSITIONS_PER_TEXT:-8}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --output-dir "$OUT"
done

python src/analyze.py --input-dir "$OUT"
