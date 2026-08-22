#!/usr/bin/env bash
set -euo pipefail

# Minimal dependency path: reuse any environment with PyTorch + NumPy.
#
# Examples:
#   bash run_gate.sh smoke 0
#   bash run_gate.sh pilot 0
#   bash run_gate.sh confirm 0,1,2
#   CONDITIONS=uniform,static bash run_gate.sh full 0
#
# With four GPUs, conditions run concurrently. With fewer GPUs they run in waves.
# No torchrun/Ray/NCCL multi-GPU training is used.

PROFILE="${1:-pilot}"
SEEDS="${2:-0}"
GPUS="${GPUS:-0,1,2,3}"
PRECISION="${PRECISION:-fp16}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
OUT="${OUT:-outputs}"
CONDITIONS="${CONDITIONS:-uniform,static,balanced_slow,balanced_fast}"

python self_test.py
python audit_schedule.py --cycles 2
python launch_grid.py \
  --profile "$PROFILE" \
  --seeds "$SEEDS" \
  --gpus "$GPUS" \
  --conditions "$CONDITIONS" \
  --precision "$PRECISION" \
  --lr-schedule "$LR_SCHEDULE" \
  --warmup-steps "$WARMUP_STEPS" \
  --output "$OUT"
python analyze.py --root "$OUT/$PROFILE"
python plot_results.py --root "$OUT/$PROFILE"
