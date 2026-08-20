#!/usr/bin/env bash
set -euo pipefail

# Forward-only reproduction of the reasoning_forks Graph Branching pass@k trajectory.
# The upstream prepare_sampling_synthetic.sh also schedules reverse checkpoints; this
# wrapper deliberately avoids requiring an unrelated reverse SFT run.

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}
BASE_REL=${BASE_REL:-inference_runs/candidate_topic_forward/qwen2.5-0.5b}
NUM_SAMPLES=${NUM_SAMPLES:-64}
WORKERS=${WORKERS:-16}
GPU_CSV=${GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
STEPS=(200 400 800 1600 3200)

[[ -d "$UPSTREAM" ]] || { echo "Missing upstream repo: $UPSTREAM; run ./prepare_upstream.sh first" >&2; exit 1; }

pushd "$UPSTREAM" >/dev/null
mkdir -p "$BASE_REL"

# Build exactly the five forward configs used by the official synthetic pass@k script.
RUN_DIRS=()
for STEP in "${STEPS[@]}"; do
  CKPT="$RUN_REL/checkpoint-$STEP"
  [[ -d "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
  EPOCH=$((STEP / 200))
  SAMPLE_DIR="$BASE_REL/sft_forward_ep${EPOCH}"
  mkdir -p "$SAMPLE_DIR"
  cat > "$SAMPLE_DIR/sampler_config.yaml" <<YAML
sampler:
  class: VLLMSampler
  model_name: ${CKPT}
  temperature: 1.0
  top_p: 0.95
  top_k: -1
  max_tokens: 512
  trust_remote_code: true
YAML
  RUN_DIRS+=("$SAMPLE_DIR")
done

# Reuse the upstream prompt builder and exact Alpaca chat template.
python src/inference/build_prompts.py \
  --dataset_name arithchain_2_10 \
  --save_dir "$BASE_REL" \
  --tokenizer_path "$RUN_REL/checkpoint-200" \
  --chat_template_path src/alpaca_template.jira

# Batch the five vLLM jobs across configurable local GPU IDs (default 0,1,2,3).
pids=()
for i in "${!RUN_DIRS[@]}"; do
  GPU=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
  REL_DIR=${RUN_DIRS[$i]#"$BASE_REL/"}
  CUDA_VISIBLE_DEVICES="$GPU" python src/inference/run_sampling.py \
    --job_dir "$BASE_REL" \
    --split arithchain_2_10 \
    --sampler_config_dir "$REL_DIR" \
    --gpu_id "$GPU" \
    --gpu_memory_utilization 0.8 \
    --n "$NUM_SAMPLES" &
  pids+=("$!")
  # Avoid loading two models onto the same GPU at once when jobs > GPUs.
  if (( (i + 1) % ${#GPU_IDS[@]} == 0 )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

python src/math_eval/evaluate_pass_k.py \
  --dirs "${RUN_DIRS[@]}" \
  --dataset math \
  --split arithchain_2_10 \
  --workers "$WORKERS"

popd >/dev/null
