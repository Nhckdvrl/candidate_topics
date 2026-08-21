#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/topic03_paper_exact/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1}
RUN_ROOT="$UPSTREAM/$RUN_REL"
LIMIT=${LIMIT:-200}
GPU_CSV=${GPUS:-0,1,2,3}
OUT=${OUT:-artifacts/preflight_states}
TAGS=(e01 e02 e04 e16)
declare -A CKPTS=( [e01]=200 [e02]=400 [e04]=800 [e16]=3200 )
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
(( ${#GPU_IDS[@]} > 0 )) || { echo "No GPUs specified" >&2; exit 1; }

mkdir -p "$OUT"
pids=()
for i in "${!TAGS[@]}"; do
  TAG=${TAGS[$i]}
  CKPT="$RUN_ROOT/checkpoint-${CKPTS[$TAG]}"
  [[ -d "$CKPT" ]] || { echo "Missing $CKPT" >&2; exit 1; }
  GPU=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
  CUDA_VISIBLE_DEVICES="$GPU" python src/extract_branch_states.py \
    --forks artifacts/forks.jsonl \
    --model "$CKPT" \
    --tag "$TAG" \
    --condition original \
    --limit "$LIMIT" \
    --output-dir "$OUT" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python src/summarize_state_preflight.py \
  --input-dir "$OUT" \
  --tags e01,e02,e04,e16 \
  --output artifacts/state_preflight.csv
