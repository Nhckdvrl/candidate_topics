#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: ./run_checkpoint_sweep.sh results/g0_seed0/main [gpu]}"
GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
for CKPT in $(ls "$RUN_DIR"/checkpoint_*.pt | sort); do
  STEM=$(basename "$CKPT" .pt)
  python -m src.evaluate_g0 \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CKPT" \
    --seed 424242 \
    --states 96 \
    --samples 64 \
    --rollout-episodes 100
  python -m src.analyze_g0 \
    --csv "$RUN_DIR/state_metrics_${STEM}.csv" \
    --eval-json "$RUN_DIR/eval_${STEM}.json" \
    --out "$RUN_DIR/sweep_${STEM}.json" \
    --bootstrap 500
done
