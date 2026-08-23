#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-8B}"
N_PAIRS="${N_PAIRS:-512}"
SEED="${SEED:-20260823}"
python -m pytest -q tests/test_g0_helpers.py
python g0_pair_locality.py --dataset zhui711/MedEinst --split test --outdir artifacts/g0_pair_locality
python g0_bias_trap_screen.py --dataset zhui711/MedEinst --split test --model "$MODEL" --n-pairs "$N_PAIRS" --seed "$SEED" --outdir artifacts/g0_behavior
