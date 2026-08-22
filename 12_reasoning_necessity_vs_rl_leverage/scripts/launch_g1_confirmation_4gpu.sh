#!/usr/bin/env bash
set -euo pipefail

# Run only if G-0 produced a large interpretable effect.
# Confirmation changes exactly one thing: ablation strength from full bypass
# (scale=0.0) to a milder 50% residual update (scale=0.5). The example ledger,
# model, prompts, task weights, layer set, and statistics remain locked.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-results/g1_qwen3_1p7b_half_residual}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
N_PER_TASK="${N_PER_TASK:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-20260822}"

mkdir -p "$OUT"

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
    --residual-scale 0.5 \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device cuda:0 \
    --output-dir "$OUT" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi

python scripts/check_integrity.py --results-dir "$OUT" --expect-layers 28
python scripts/analyze_relation.py --results-dir "$OUT" --bootstrap 2000 --seed "$SEED"

echo "Done. Read: $OUT/REPORT.md"
