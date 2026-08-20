#!/usr/bin/env bash
set -euo pipefail

[[ -f artifacts/forks.jsonl ]] || ./prepare_upstream.sh

# G0: before spending compute on a checkpoint trajectory, establish that branch viability
# is measurable in the base model at the exact first decision point.
python src/extract_branch_states.py \
  --forks artifacts/forks.jsonl \
  --model "${MODEL:-unsloth/Qwen2.5-0.5B}" \
  --tag base \
  --output-dir artifacts/states

python src/train_pairwise_probe.py \
  --input-dir artifacts/states \
  --output artifacts/branch_probe_metrics.csv
