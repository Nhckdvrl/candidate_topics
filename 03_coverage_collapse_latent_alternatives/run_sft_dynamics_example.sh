#!/usr/bin/env bash
set -euo pipefail
# Full hidden trajectory is intentionally post-G0. Do not run this to decide whether
# the topic is alive; ./run_g0.sh compares only the behavior-selected reference and e16.
RUN=${RUN:-external/reasoning_forks/runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}
RUN_ID=${RUN_ID:-$(cat artifacts/behavior/latest_run.txt 2>/dev/null || true)}
[[ -n "$RUN_ID" ]] || { echo "Run G0 first or set RUN_ID" >&2; exit 1; }
REF_TAG=${REF_TAG:-$(python - "artifacts/behavior/$RUN_ID/gate.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['reference_tag'])
PY
)}
OUT="artifacts/full_states/$RUN_ID"
mkdir -p "$OUT"
declare -A CKPTS=( [e01]=200 [e02]=400 [e04]=800 [e08]=1600 [e16]=3200 )
for TAG in e01 e02 e04 e08 e16; do
  CKPT="$RUN/checkpoint-${CKPTS[$TAG]}"
  [[ -d "$CKPT" ]] || { echo "Missing $CKPT" >&2; exit 1; }
  python src/extract_branch_states.py --forks artifacts/forks.jsonl --model "$CKPT" --tag "$TAG" --output-dir "$OUT"
done
python src/train_pairwise_probe.py --input-dir "$OUT" --reference-tag "$REF_TAG" --late-tag e16 \
  --output "artifacts/behavior/$RUN_ID/full_trajectory_endpoint_metrics.csv" \
  --predictions-output "artifacts/behavior/$RUN_ID/full_trajectory_endpoint_predictions.csv"
echo "All intermediate states are saved in $OUT for follow-up trajectory analysis."
