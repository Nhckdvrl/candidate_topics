#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/raw
NUM_EXAMPLES=${NUM_EXAMPLES:-1000}
NUM_SHARDS=4

pids=()
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU python src/generate_fates.py \
    --num-examples "$NUM_EXAMPLES" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$GPU" \
    --output-dir artifacts/raw \
    > "artifacts/raw/shard_${GPU}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
python src/train_probes.py --input-dir artifacts/raw --output-dir artifacts/probes
