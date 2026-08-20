#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NUM_EXAMPLES=${NUM_EXAMPLES:-1000}
GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_IDS <<< "$GPUS"
NUM_SHARDS=${#GPU_IDS[@]}
OUT_ROOT=${OUT_ROOT:-artifacts/reference_probing_geometry}
RAW="$OUT_ROOT/raw"
PROBES="$OUT_ROOT/probes"
CAPTURE=(0 1 2 4 8 16 24 32 48 64 80 96 112 120 124 127)

rm -rf "$OUT_ROOT"
mkdir -p "$RAW"

pids=()
for IDX in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$IDX]}
  CUDA_VISIBLE_DEVICES="$GPU" python src/generate_fates.py \
    --num-examples "$NUM_EXAMPLES" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$IDX" \
    --steps 128 \
    --gen-length 512 \
    --block-length 32 \
    --temperature 0.2 \
    --prompt-style probing \
    --capture-steps "${CAPTURE[@]}" \
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

python src/train_probes.py \
  --input-dir "$RAW" \
  --output-dir "$PROBES" \
  --label-mode strict \
  --min-class-count "${MIN_CLASS_COUNT:-30}" \
  --bootstrap "${BOOTSTRAP:-500}"
