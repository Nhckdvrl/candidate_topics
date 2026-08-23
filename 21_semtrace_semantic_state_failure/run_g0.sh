#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
N="${N:-64}"
SEED="${SEED:-20260823}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-8192}"
STEPS="${STEPS:-8}"
python -m pytest -q tests/test_g0_helpers.py
python g0_position_dissociation.py --model "$MODEL" --n "$N" --seed "$SEED" --context-tokens "$CONTEXT_TOKENS" --steps "$STEPS" --outdir artifacts/g0
