#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-external/reasoning_forks/runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}
RUN_ID=${RUN_ID:-$(cat artifacts/behavior/latest_run.txt 2>/dev/null || true)}
GPU_CSV=${GPUS:-0,1,2}
[[ -n "$RUN_ID" ]] || { echo "Set RUN_ID or run behavior preflight first" >&2; exit 1; }
GATE="artifacts/behavior/$RUN_ID/gate.json"
[[ -f "$GATE" ]] || { echo "Missing behavior gate: $GATE" >&2; exit 1; }

read -r STATUS REF_TAG LATE_TAG < <(python - "$GATE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d['status'], d['reference_tag'], d['late_tag'])
PY
)
if [[ "$STATUS" != "continue_to_latent" ]]; then
  echo "Behavior gate says $STATUS; refusing expensive hidden-state run." >&2
  cat "$GATE" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
if (( ${#GPU_IDS[@]} < 3 )); then
  echo "Need at least 3 GPU ids for parallel reference/late/control extraction (e.g. GPUS=0,1,2)." >&2
  exit 1
fi

declare -A CKPTS=( [e01]=200 [e02]=400 [e04]=800 [e08]=1600 [e16]=3200 )
REF_STEP=${CKPTS[$REF_TAG]:-}; LATE_STEP=${CKPTS[$LATE_TAG]:-}
[[ -n "$REF_STEP" && -n "$LATE_STEP" ]] || { echo "Unsupported tags $REF_TAG/$LATE_TAG" >&2; exit 1; }
REF_MODEL="$RUN_REL/checkpoint-$REF_STEP"
LATE_MODEL="$RUN_REL/checkpoint-$LATE_STEP"
[[ -d "$REF_MODEL" && -d "$LATE_MODEL" ]] || { echo "Missing SFT checkpoints under $RUN_REL" >&2; exit 1; }

STATE_DIR="artifacts/states/$RUN_ID"
mkdir -p "$STATE_DIR"
pids=()
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python src/extract_branch_states.py \
  --forks artifacts/forks.jsonl --model "$REF_MODEL" --tag "$REF_TAG" --output-dir "$STATE_DIR" & pids+=("$!")
CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" python src/extract_branch_states.py \
  --forks artifacts/forks.jsonl --model "$LATE_MODEL" --tag "$LATE_TAG" --output-dir "$STATE_DIR" & pids+=("$!")
CUDA_VISIBLE_DEVICES="${GPU_IDS[2]}" python src/extract_branch_states.py \
  --forks artifacts/forks.jsonl --model "$REF_MODEL" --tag "$REF_TAG" --mask-target --output-dir "$STATE_DIR" & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

python src/train_pairwise_probe.py \
  --input-dir "$STATE_DIR" \
  --reference-tag "$REF_TAG" \
  --late-tag "$LATE_TAG" \
  --control-tag "${REF_TAG}_target_blind" \
  --output "artifacts/behavior/$RUN_ID/latent_gate_metrics.csv" \
  --predictions-output "artifacts/behavior/$RUN_ID/latent_gate_predictions.csv"

echo "Latent gate: artifacts/behavior/$RUN_ID/latent_gate_metrics.json"
