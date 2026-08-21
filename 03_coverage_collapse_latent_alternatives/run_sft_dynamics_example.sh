#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/topic03_paper_exact/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1}
RUN_ROOT="$UPSTREAM/$RUN_REL"
RUN_ID=${RUN_ID:-$(cat artifacts/behavior/latest_run.txt 2>/dev/null || true)}
[[ -n "$RUN_ID" ]] || { echo "Run G0 first or set RUN_ID" >&2; exit 1; }

REF_TAG=${REF_TAG:-$(python - "artifacts/behavior/$RUN_ID/gate.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["reference_tag"])
PY
)}
EXCLUDE_IDS="artifacts/behavior/$RUN_ID/first_branch_per_problem.csv"
OUT="artifacts/full_states/$RUN_ID"
mkdir -p "$OUT"

declare -A CKPTS=( [e01]=200 [e02]=400 [e04]=800 [e08]=1600 [e16]=3200 )
for TAG in e01 e02 e04 e08 e16; do
  CKPT="$RUN_ROOT/checkpoint-${CKPTS[$TAG]}"
  [[ -d "$CKPT" ]] || { echo "Missing $CKPT" >&2; exit 1; }
  for COND in original target_flip; do
    python src/extract_branch_states.py \
      --forks artifacts/forks.jsonl \
      --exclude-problem-ids "$EXCLUDE_IDS" \
      --model "$CKPT" \
      --tag "$TAG" \
      --condition "$COND" \
      --output-dir "$OUT"
  done
done

REF_CKPT="$RUN_ROOT/checkpoint-${CKPTS[$REF_TAG]}"
python src/extract_branch_states.py \
  --forks artifacts/forks.jsonl \
  --exclude-problem-ids "$EXCLUDE_IDS" \
  --model "$REF_CKPT" \
  --tag "$REF_TAG" \
  --condition target_blind \
  --output-dir "$OUT"

python src/train_pairwise_probe.py \
  --input-dir "$OUT" \
  --reference-tag "$REF_TAG" \
  --late-tag e16 \
  --output "artifacts/behavior/$RUN_ID/full_trajectory_endpoint_metrics.csv" \
  --predictions-output "artifacts/behavior/$RUN_ID/full_trajectory_endpoint_predictions.csv"

echo "All intermediate original/target-flip states are saved in $OUT for post-G0 trajectory analysis."
