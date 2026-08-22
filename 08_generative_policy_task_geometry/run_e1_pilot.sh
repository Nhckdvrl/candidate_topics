#!/usr/bin/env bash
# E1 discovery pilot on one GPU. Preflight must pass before any collection runs.
# Run from inside 08_generative_policy_task_geometry/ (the folder name starts with a
# digit, so it cannot be a Python package; `src` is the package root).
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/home/xiang/venvs/topic08/bin/python}"
GPU="${GPU:-0}"
OUT="${OUT:-results/pusht_e1_pilot}"
ROLLOUTS="${ROLLOUTS:-8}"
SAMPLES="${SAMPLES:-64}"
MAX_STEPS="${MAX_STEPS:-200}"
SEED_BASE="${SEED_BASE:-100000}"
CALIB="${CALIB:-4}"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy
mkdir -p "$OUT"

echo "== preflight =="
"$PY" -m src.pusht.preflight --out "$OUT/preflight.json"

echo "== collect =="
"$PY" -m src.pusht.collect_e1 \
  --out "$OUT" --rollouts "$ROLLOUTS" --samples "$SAMPLES" \
  --max-steps "$MAX_STEPS" --seed-base "$SEED_BASE" --calib-rollouts "$CALIB"

echo "== analyse (descriptive, no verdict) =="
"$PY" -m src.pusht.analyze_e1 --csv "$OUT/probe_states.csv" --out "$OUT/e1_report.json"
