#!/usr/bin/env bash
set -euo pipefail

# Robustness run closer to the public Time-Is-a-Feature GSM8K evaluation example:
# gen_length=128, diffusion_steps=64, temperature=0, MidTruth boxed-answer prompt.
OUT=artifacts/raw_midtruth
mkdir -p "$OUT"
CAPTURE=(0 1 2 4 8 16 24 32 40 48 56 60 62 63)
pids=()
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU python src/generate_fates.py \
    --num-examples "${NUM_EXAMPLES:-1000}" \
    --num-shards 4 --shard-index "$GPU" \
    --steps 64 --gen-length 128 --block-length 32 --temperature 0 \
    --prompt-style midtruth \
    --capture-steps "${CAPTURE[@]}" \
    --output-dir "$OUT" \
    > "$OUT/shard_${GPU}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python src/train_probes.py --input-dir "$OUT" --output-dir artifacts/probes_midtruth --label-mode strict
