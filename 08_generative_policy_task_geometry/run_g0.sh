#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-0}"
BASE_TASKS="${BASE_TASKS:-300}"
MODES="${MODES:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
STATES="${STATES:-256}"
SAMPLES="${SAMPLES:-128}"
ROLLOUTS="${ROLLOUTS:-200}"
RUN="results/g0_seed${SEED}"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$RUN"

python -m src.train_g0 \
  --out "$RUN/main" \
  --seed "$SEED" \
  --base-tasks "$BASE_TASKS" \
  --modes-per-task "$MODES" \
  --null-gain 1.0 \
  --train-steps "$TRAIN_STEPS"

MAIN_CKPT=$(ls "$RUN/main"/checkpoint_*.pt | sort | tail -1)
python -m src.evaluate_g0 \
  --run-dir "$RUN/main" \
  --checkpoint "$MAIN_CKPT" \
  --seed "$((SEED + 10000))" \
  --states "$STATES" \
  --samples "$SAMPLES" \
  --rollout-episodes "$ROLLOUTS"

MAIN_STEM=$(basename "$MAIN_CKPT" .pt)
python -m src.analyze_g0 \
  --csv "$RUN/main/state_metrics_${MAIN_STEM}.csv" \
  --eval-json "$RUN/main/eval_${MAIN_STEM}.json" \
  --out "$RUN/main/G0_GATE.json"

python -m src.train_g0 \
  --out "$RUN/no_null_control" \
  --seed "$SEED" \
  --base-tasks "$BASE_TASKS" \
  --modes-per-task "$MODES" \
  --null-gain 0.0 \
  --train-steps "$TRAIN_STEPS"

CTRL_CKPT=$(ls "$RUN/no_null_control"/checkpoint_*.pt | sort | tail -1)
python -m src.evaluate_g0 \
  --run-dir "$RUN/no_null_control" \
  --checkpoint "$CTRL_CKPT" \
  --seed "$((SEED + 10000))" \
  --states "$STATES" \
  --samples "$SAMPLES" \
  --rollout-episodes "$ROLLOUTS"

CTRL_STEM=$(basename "$CTRL_CKPT" .pt)
python -m src.analyze_g0 \
  --csv "$RUN/no_null_control/state_metrics_${CTRL_STEM}.csv" \
  --eval-json "$RUN/no_null_control/eval_${CTRL_STEM}.json" \
  --out "$RUN/no_null_control/G0_GATE.json"

echo
cat "$RUN/main/G0_GATE.json"
echo
echo "--- no-null control ---"
cat "$RUN/no_null_control/G0_GATE.json"
