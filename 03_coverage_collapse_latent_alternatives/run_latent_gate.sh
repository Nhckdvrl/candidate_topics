#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/topic03_paper_exact/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1}
RUN_ID=${RUN_ID:-$(cat artifacts/behavior/latest_run.txt 2>/dev/null || true)}
GPU_CSV=${GPUS:-0,1,2,3}

[[ -n "$RUN_ID" ]] || { echo "Set RUN_ID or run behavior preflight first" >&2; exit 1; }
GATE="artifacts/behavior/$RUN_ID/gate.json"
EXCLUDE_IDS="artifacts/behavior/$RUN_ID/first_branch_per_problem.csv"
[[ -f "$GATE" ]] || { echo "Missing behavior gate: $GATE" >&2; exit 1; }
[[ -f "$EXCLUDE_IDS" ]] || { echo "Missing behavior problem manifest: $EXCLUDE_IDS" >&2; exit 1; }

read -r STATUS REF_TAG LATE_TAG < <(python - "$GATE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d["status"], d["reference_tag"], d["late_tag"])
PY
)
if [[ "$STATUS" != "continue_to_latent" ]]; then
  echo "Behavior gate says $STATUS; refusing hidden-state run." >&2
  cat "$GATE" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "Need at least one GPU id." >&2
  exit 1
fi

declare -A CKPTS=( [e01]=200 [e02]=400 [e04]=800 [e08]=1600 [e16]=3200 )
REF_STEP=${CKPTS[$REF_TAG]:-}
LATE_STEP=${CKPTS[$LATE_TAG]:-}
[[ -n "$REF_STEP" && -n "$LATE_STEP" ]] || { echo "Unsupported tags $REF_TAG/$LATE_TAG" >&2; exit 1; }

RUN_ROOT="$UPSTREAM/$RUN_REL"
REF_MODEL="$RUN_ROOT/checkpoint-$REF_STEP"
LATE_MODEL="$RUN_ROOT/checkpoint-$LATE_STEP"
[[ -d "$REF_MODEL" && -d "$LATE_MODEL" ]] || { echo "Missing SFT checkpoints under $RUN_ROOT" >&2; exit 1; }

STATE_DIR="artifacts/states/$RUN_ID"
mkdir -p "$STATE_DIR"

# Five cheap 0.5B forward-state jobs. With four local GPUs, the first four run in
# parallel and target-blind runs immediately after. No cross-node communication is needed.
JOB_TAGS=("$REF_TAG" "$REF_TAG" "$LATE_TAG" "$LATE_TAG" "$REF_TAG")
JOB_CONDS=("original" "target_flip" "original" "target_flip" "target_blind")
JOB_MODELS=("$REF_MODEL" "$REF_MODEL" "$LATE_MODEL" "$LATE_MODEL" "$REF_MODEL")

idx=0
while (( idx < ${#JOB_TAGS[@]} )); do
  pids=()
  batch=0
  while (( batch < ${#GPU_IDS[@]} && idx < ${#JOB_TAGS[@]} )); do
    GPU=${GPU_IDS[$batch]}
    TAG=${JOB_TAGS[$idx]}
    COND=${JOB_CONDS[$idx]}
    MODEL=${JOB_MODELS[$idx]}
    CUDA_VISIBLE_DEVICES="$GPU" python src/extract_branch_states.py \
      --forks artifacts/forks.jsonl \
      --exclude-problem-ids "$EXCLUDE_IDS" \
      --model "$MODEL" \
      --tag "$TAG" \
      --condition "$COND" \
      --output-dir "$STATE_DIR" &
    pids+=("$!")
    idx=$((idx + 1))
    batch=$((batch + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

python src/train_pairwise_probe.py \
  --input-dir "$STATE_DIR" \
  --reference-tag "$REF_TAG" \
  --late-tag "$LATE_TAG" \
  --output "artifacts/behavior/$RUN_ID/latent_gate_metrics.csv" \
  --predictions-output "artifacts/behavior/$RUN_ID/latent_gate_predictions.csv"

echo "Latent decision: artifacts/behavior/$RUN_ID/latent_gate_metrics.json"
