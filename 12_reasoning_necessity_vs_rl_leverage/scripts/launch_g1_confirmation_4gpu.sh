#!/usr/bin/env bash
set -euo pipefail

# Predeclared mild-intervention run. Use it when full deletion is too destructive,
# or after a strong G-0 as a dose-strength confirmation. It changes exactly one
# scientific variable relative to G-0: residual_scale 0.0 -> 0.5.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-results/g1_qwen3_1p7b_half_residual}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
MODEL_REVISION="${MODEL_REVISION:-912d2727784ca0a6f718845aa14d4d9e5f48fe26}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_math_seed}"
N_PER_TASK="${N_PER_TASK:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-20260822}"

# These are scientific, not throughput knobs. Refuse accidental protocol drift.
if [[ "$N_PER_TASK" != "256" || "$MAX_NEW_TOKENS" != "1536" || "$SEED" != "20260822" || "$PROMPT_STYLE" != "qwen_math_seed" ]]; then
  echo "Refusing non-locked Topic-12 protocol: N_PER_TASK=256, MAX_NEW_TOKENS=1536, SEED=20260822, PROMPT_STYLE=qwen_math_seed are frozen." >&2
  exit 2
fi
if [[ ! -d "$MODEL" && "$MODEL_REVISION" != "912d2727784ca0a6f718845aa14d4d9e5f48fe26" ]]; then
  echo "Refusing unpinned remote model revision. For a local staged snapshot, set MODEL to its directory." >&2
  exit 2
fi

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py \
  --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --prompt-style "$PROMPT_STYLE" \
  --tasks math500,gsm8k --n-per-task "$N_PER_TASK" --seed "$SEED" \
  --layers none --include-baseline --residual-scale 0.5 \
  --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --device cuda:0 --output-dir "$OUT"

# Do not spend 28-layer compute if this protocol cannot reproduce the published base.
python scripts/check_integrity.py --results-dir "$OUT" --baseline-only

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_ablation.py \
    --model "$MODEL" --model-revision "$MODEL_REVISION" \
    --prompt-style "$PROMPT_STYLE" \
    --tasks math500,gsm8k --n-per-task "$N_PER_TASK" --seed "$SEED" \
    --layers all --layer-shard-index "$gpu" --layer-shard-count 4 \
    --residual-scale 0.5 --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" --device cuda:0 --output-dir "$OUT" &
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
