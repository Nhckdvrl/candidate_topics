#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${INPUT:?set INPUT}"
: "${OUTDIR:?set OUTDIR}"

N="${N:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"

mkdir -p "$OUTDIR"
pids=()
for gpu in 0 1 2 3; do
  shard_index=$((SHARD_OFFSET + gpu))
  CUDA_VISIBLE_DEVICES="$gpu" python code/run_vllm_generate.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$OUTDIR/shard_${shard_index}.jsonl" \
    --n "$N" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard_index" \
    > "$OUTDIR/shard_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

# Safe for the common single-node case. In a multi-node/global-shard run,
# concatenate all shard_*.jsonl only after moving/copying them into one directory.
cat "$OUTDIR"/shard_*.jsonl > "$OUTDIR/all.jsonl"
echo "Wrote $OUTDIR/all.jsonl"
