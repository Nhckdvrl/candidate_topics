#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/topic03_paper_exact/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1}
TRAIN_GPU=${TRAIN_GPU:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

[[ -d "$UPSTREAM/.git" ]] || { echo "Missing upstream checkout; run ./prepare_upstream.sh first." >&2; exit 1; }
[[ -f "$UPSTREAM/datasets/arithchain_2_10/train_sft_forward.parquet" ]] || {
  echo "Missing generated ArithChain data; run ./prepare_upstream.sh first." >&2
  exit 1
}

if [[ -d "$UPSTREAM/$RUN_REL/checkpoint-3200" ]]; then
  echo "Paper-exact checkpoint-3200 already exists: $UPSTREAM/$RUN_REL"
  exit 0
fi

# Important audit finding:
# - Nguyen et al. Appendix A.2 reports lr=2e-5 for Graph Branching.
# - pinned upstream run_sft.sh currently hard-codes 1e-5 for qwen2.5_0.5b.
# We call sft.py directly so a failed 1e-5 run cannot create a false scientific kill.
pushd "$UPSTREAM" >/dev/null
mkdir -p "$RUN_REL"
RESUME_ARGS=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python src/training/sft.py \
  --model_name unsloth/Qwen2.5-0.5B \
  --data_path datasets/arithchain_2_10/train_sft_forward.parquet \
  --prompt_key question \
  --response_key solution \
  --chat_template_path src/alpaca_template.jira \
  --batch_size 32 \
  --grad_accum 1 \
  --warmup_ratio 0.1 \
  --num_train_epochs 16 \
  --learning_rate 2e-5 \
  --save_steps 200 \
  --output_dir "$RUN_REL" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$RUN_REL/train.log"
popd >/dev/null

[[ -d "$UPSTREAM/$RUN_REL/checkpoint-3200" ]] || {
  echo "Training finished without checkpoint-3200; inspect $UPSTREAM/$RUN_REL/train.log" >&2
  exit 2
}
echo "Paper-exact SFT trajectory ready: $UPSTREAM/$RUN_REL"
