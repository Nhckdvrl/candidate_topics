#!/usr/bin/env bash
# One arm of the matched G1 pair. Single GPU per arm: multi-GPU NCCL on this host's
# Blackwell cards raised an illegal-memory-access at DDP init, and world_size=1 keeps the
# two arms exactly matched (identical ResumableEpochSampler order) without that risk.
set -euo pipefail
ARM="$1"; LAMBDA="$2"; GPU="$3"
T=/home/xiang/candidate_topics/15_predictive_policy_state
LW=/home/xiang/projects/Light-WAM
PY=/home/xiang/miniconda3/envs/lightwam/bin/python
CUDA_VISIBLE_DEVICES="$GPU" $PY -m accelerate.commands.launch \
  --num_processes 1 --mixed_precision bf16 \
  "$T/g1_train.py" \
  --lightwam-root "$LW" \
  --config "$T/g1/matched_config.yaml" \
  --init-checkpoint "$T/g1/init/init.pt" \
  --output-dir "$T/g1/runs/$ARM" \
  --lambda-video "$LAMBDA"
