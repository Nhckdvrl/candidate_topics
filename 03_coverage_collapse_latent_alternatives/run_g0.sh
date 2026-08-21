#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/topic03_paper_exact/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1}
GPU_CSV=${GPUS:-0,1,2,3}
export UPSTREAM RUN_REL

[[ -f artifacts/forks.jsonl ]] || bash ./prepare_upstream.sh

if [[ ! -d "$UPSTREAM/$RUN_REL/checkpoint-3200" ]]; then
  TRAIN_GPU=${TRAIN_GPU:-${GPU_CSV%%,*}}
  echo "Paper-exact SFT trajectory missing; training on GPU $TRAIN_GPU."
  TRAIN_GPU="$TRAIN_GPU" bash ./run_train_paper_exact.sh
fi

if [[ "${RUN_STATE_PREFLIGHT:-0}" == "1" ]]; then
  bash ./run_state_preflight.sh
fi

bash ./run_behavior_preflight.sh

RUN_ID=$(cat artifacts/behavior/latest_run.txt)
printf '%s\n' "$RUN_ID" > artifacts/behavior/latest_g0_run.txt
STATUS=$(python - "artifacts/behavior/$RUN_ID/gate.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
if [[ "$STATUS" != "continue_to_latent" ]]; then
  echo "G0-A stopped the mechanism before probing. See artifacts/behavior/$RUN_ID/gate.json"
  exit 2
fi

bash ./run_latent_gate.sh

python - "artifacts/behavior/$RUN_ID/latent_gate_metrics.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("\n=== Topic03 G0 final decision ===")
print(d["status"])
for r in d.get("reasons", []):
    print("-", r)
PY
