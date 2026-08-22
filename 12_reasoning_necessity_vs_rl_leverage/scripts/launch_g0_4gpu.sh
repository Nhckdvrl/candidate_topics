#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-results/g0_qwen3_1p7b_bypass}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
N_PER_TASK="${N_PER_TASK:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-20260822}"

mkdir -p "$OUT"

echo "[1/4] baseline on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py \
  --model "$MODEL" \
  --tasks math500,gsm8k \
  --n-per-task "$N_PER_TASK" \
  --seed "$SEED" \
  --layers none \
  --include-baseline \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device cuda:0 \
  --output-dir "$OUT"

echo "[2/4] full 28-layer bypass sweep, sharded over four GPUs"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_ablation.py \
    --model "$MODEL" \
    --tasks math500,gsm8k \
    --n-per-task "$N_PER_TASK" \
    --seed "$SEED" \
    --layers all \
    --layer-shard-index "$gpu" \
    --layer-shard-count 4 \
    --residual-scale 0.0 \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device cuda:0 \
    --output-dir "$OUT" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one GPU worker failed; rerun the same command to resume." >&2
  exit 1
fi

echo "[3/4] structural/integrity gate"
python scripts/check_integrity.py \
  --results-dir "$OUT" \
  --expect-layers 28

echo "[4/4] locked statistics + report"
python scripts/analyze_relation.py \
  --results-dir "$OUT" \
  --bootstrap 2000 \
  --seed "$SEED"

echo "Done. Read: $OUT/REPORT.md"
