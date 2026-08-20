#!/usr/bin/env bash
set -euo pipefail

# Assumes the official reasoning_forks forward SFT run has already been produced.
# Their Graph Branching setup has 6400 examples, batch size 32 and save_steps=200.
# We intentionally use the SAME checkpoints as the official pass@k script: epochs 1,2,4,8,16.
RUN=${RUN:-external/reasoning_forks/runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}

declare -A CKPTS=(
  [e01]=200 [e02]=400 [e04]=800 [e08]=1600 [e16]=3200
)

for TAG in e01 e02 e04 e08 e16; do
  STEP=${CKPTS[$TAG]}
  CKPT="$RUN/checkpoint-$STEP"
  [[ -d "$CKPT" ]] || { echo "Missing $CKPT" >&2; exit 1; }
  python src/extract_branch_states.py \
    --forks artifacts/forks.jsonl \
    --model "$CKPT" \
    --tag "$TAG" \
    --output-dir artifacts/states
done

python src/train_pairwise_probe.py --input-dir artifacts/states --output artifacts/branch_probe_metrics.csv
