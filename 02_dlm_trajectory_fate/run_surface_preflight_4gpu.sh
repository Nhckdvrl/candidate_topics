#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NUM_EXAMPLES=${NUM_EXAMPLES:-200}
GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_IDS <<< "$GPUS"
NUM_SHARDS=${#GPU_IDS[@]}
OUT_ROOT=${OUT_ROOT:-artifacts/preflight_midtruth}
RAW="$OUT_ROOT/raw"

rm -rf "$OUT_ROOT"
mkdir -p "$RAW"

pids=()
for IDX in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$IDX]}
  CUDA_VISIBLE_DEVICES="$GPU" python src/generate_fates.py \
    --num-examples "$NUM_EXAMPLES" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$IDX" \
    --steps 64 \
    --gen-length 128 \
    --block-length 32 \
    --temperature 0 \
    --prompt-style midtruth \
    --surface-only \
    --output-dir "$RAW" \
    > "$RAW/shard_${IDX}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one generation shard failed. Inspect $RAW/shard_*.log" >&2
  exit 1
fi

python src/summarize_surface.py \
  --input-dir "$RAW" \
  --output-dir "$OUT_ROOT" \
  --label-mode strict \
  --min-novel-class "${MIN_NOVEL_CLASS:-10}"
