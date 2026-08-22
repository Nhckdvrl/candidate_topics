#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-results/smoke}"
rm -rf "$OUT"

CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py \
  --tasks math500,gsm8k \
  --n-per-task 8 \
  --layers 0,10,24 \
  --include-baseline \
  --batch-size 4 \
  --max-new-tokens 512 \
  --device cuda:0 \
  --output-dir "$OUT"

python scripts/check_integrity.py \
  --results-dir "$OUT" \
  --expect-layers 28 \
  --allow-missing-layers

echo "Smoke test passed. It intentionally does not run inferential statistics."
