#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
N="${N:-64}"
SEED="${SEED:-20260823}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-8192}"
STEPS="${STEPS:-8}"
UPSTREAM_SUMMARY="${UPSTREAM_SUMMARY:-}"

python -m pytest -q tests/test_g0_helpers.py

if [[ -z "$UPSTREAM_SUMMARY" ]]; then
  echo "ERROR: set UPSTREAM_SUMMARY to the official long-context-code-understanding fsyn_output_prediction summary.json after reproducing the seed on the same model." >&2
  exit 2
fi

python g0_upstream_contract.py --summary "$UPSTREAM_SUMMARY" --out artifacts/g0_upstream_contract.json
python g0_position_dissociation.py \
  --model "$MODEL" \
  --n "$N" \
  --seed "$SEED" \
  --context-tokens "$CONTEXT_TOKENS" \
  --steps "$STEPS" \
  --outdir artifacts/g0
