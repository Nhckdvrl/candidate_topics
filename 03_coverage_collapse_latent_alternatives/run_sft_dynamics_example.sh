#!/usr/bin/env bash
set -euo pipefail

# Assumes the official reasoning_forks SFT run has already been produced.
# Their graph setup has 6400 examples, batch size 32, save_steps=200 => one checkpoint per epoch.
RUN=${RUN:-external/reasoning_forks/runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}

# Sparse preregistered trajectory: epoch -> optimizer step.
declare -A CKPTS=(
  [e01]=200 [e02]=400 [e04]=800 [e06]=1200 [e08]=1600
  [e10]=2000 [e12]=2400 [e14]=2800 [e16]=3200
)

for TAG in e01 e02 e04 e06 e08 e10 e12 e14 e16; do
  STEP=${CKPTS[$TAG]}
  python src/extract_branch_states.py \
    --forks artifacts/forks.jsonl \
    --model "$RUN/checkpoint-$STEP" \
    --tag "$TAG" \
    --output-dir artifacts/states
done

python src/train_pairwise_probe.py --input-dir artifacts/states --output artifacts/branch_probe_metrics.csv
