#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
RUN_REL=${RUN_REL:-runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}
NUM_PROBLEMS=${NUM_PROBLEMS:-200}
NUM_SAMPLES=${NUM_SAMPLES:-16}
WORKERS=${WORKERS:-8}
GPU_CSV=${GPUS:-0,1,2,3}
RUN_ID=${RUN_ID:-g0_$(date +%Y%m%d_%H%M%S)}
TAGS_CSV=${TAGS:-e01,e02,e04,e16}
SPLIT=arithchain_2_10_g0
BASE_REL="inference_runs/topic03/${RUN_ID}"
OUT_DIR="artifacts/behavior/${RUN_ID}"

[[ -d "$UPSTREAM/.git" ]] || { echo "Missing upstream repo; run ./prepare_upstream.sh" >&2; exit 1; }
[[ -f artifacts/forks.jsonl ]] || { echo "Missing artifacts/forks.jsonl; run ./prepare_upstream.sh" >&2; exit 1; }
[[ ! -e "$UPSTREAM/$BASE_REL" ]] || { echo "Run already exists: $UPSTREAM/$BASE_REL. Use a fresh RUN_ID." >&2; exit 1; }

IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
IFS=',' read -r -a TAGS_ARR <<< "$TAGS_CSV"
if (( ${#GPU_IDS[@]} == 0 )); then echo "No GPUs specified" >&2; exit 1; fi

declare -A CKPTS=( [e01]=200 [e02]=400 [e04]=800 [e08]=1600 [e16]=3200 )

pushd "$UPSTREAM" >/dev/null
mkdir -p "$BASE_REL"
python src/inference/build_prompts.py \
  --dataset_name arithchain_2_10 \
  --save_dir "$BASE_REL" \
  --tokenizer_path "$RUN_REL/checkpoint-200" \
  --chat_template_path src/alpaca_template.jira

python - "$BASE_REL/arithchain_2_10.prompts.csv" "$BASE_REL/$SPLIT.prompts.csv" "$NUM_PROBLEMS" <<'PY'
import sys
import pandas as pd
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src).head(n)
if len(df) != n:
    raise SystemExit(f"requested {n} prompts, found {len(df)}")
df.to_csv(dst, index=False)
print(f"wrote {len(df)} preflight prompts -> {dst}")
PY

RUN_DIRS=()
pids=()
for i in "${!TAGS_ARR[@]}"; do
  TAG=${TAGS_ARR[$i]}
  STEP=${CKPTS[$TAG]:-}
  [[ -n "$STEP" ]] || { echo "Unknown tag: $TAG" >&2; exit 1; }
  CKPT="$RUN_REL/checkpoint-$STEP"
  [[ -d "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
  DIR="$BASE_REL/$TAG"
  mkdir -p "$DIR"
  cat > "$DIR/sampler_config.yaml" <<YAML
sampler:
  class: VLLMSampler
  model_name: ${CKPT}
  temperature: 1.0
  top_p: 0.95
  top_k: -1
  max_tokens: 512
  trust_remote_code: true
YAML
  RUN_DIRS+=("$DIR")
  GPU=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
  CUDA_VISIBLE_DEVICES="$GPU" python src/inference/run_sampling.py \
    --job_dir "$BASE_REL" \
    --split "$SPLIT" \
    --sampler_config_dir "$TAG" \
    --gpu_id "$GPU" \
    --gpu_memory_utilization 0.75 \
    --n "$NUM_SAMPLES" &
  pids+=("$!")
  if (( (i + 1) % ${#GPU_IDS[@]} == 0 )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

python src/math_eval/evaluate_pass_k.py \
  --dirs "${RUN_DIRS[@]}" \
  --dataset math \
  --split "$SPLIT" \
  --cutoff "$NUM_SAMPLES" \
  --workers "$WORKERS"
popd >/dev/null

mkdir -p "$OUT_DIR"
python src/analyze_sampled_branches.py \
  --forks artifacts/forks.jsonl \
  --run-root "$UPSTREAM/$BASE_REL" \
  --split "$SPLIT" \
  --late-tag e16 \
  --output-dir "$OUT_DIR"
printf '%s\n' "$RUN_ID" > artifacts/behavior/latest_run.txt

echo "Behavior gate: $OUT_DIR/gate.json"
