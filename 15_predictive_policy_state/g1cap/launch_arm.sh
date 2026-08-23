#!/usr/bin/env bash
# One arm of the CAPACITY-RESTORED matched pair.
#
# Difference from g1/: backbone LoRA is restored (identically in both arms) and receives
# ACTION-loss gradient only, via exact gradient routing in the trainer. This gives the action
# path non-shared capacity, so the earlier negative cannot be explained by future and action
# competing for the same 2.37M shared adapter bottleneck. The WAM adapters remain the ONLY
# module through which future supervision can reach the deployed policy representation.
#
# The treatment is still exactly one thing: lambda_video = 1.0 vs 0.0.
set -euo pipefail
ARM="$1"; LAMBDA="$2"; GPU="$3"
T=/home/xiang/candidate_topics/15_predictive_policy_state
LW=/home/xiang/projects/Light-WAM
PY=/home/xiang/miniconda3/envs/lightwam/bin/python
export LIGHTWAM_LORA_ACTION_ONLY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="$GPU" $PY -m accelerate.commands.launch \
  --num_processes 1 --mixed_precision bf16 \
  "$T/g1_train.py" \
  --lightwam-root "$LW" \
  --config "$T/g1cap/matched_config.yaml" \
  --init-checkpoint "$T/g1cap/init/init.pt" \
  --output-dir "$T/g1cap/runs/$ARM" \
  --lambda-video "$LAMBDA"
