#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/configs/g0.json}"
RUN_DIR="${RUN_DIR:-$HERE/runs/g0}"
RUN_TESTS="${RUN_TESTS:-1}"

# Frozen scientific defaults. Infrastructure knobs (GPU ids, batch size) may be
# overridden without changing the scientific design.
eval "$(python - "$CONFIG" <<'PY'
import json, shlex, sys
c=json.load(open(sys.argv[1]))
keys = [
    'model','revision','num_pairs','min_eligible_pairs','anchor_min','anchor_max',
    'batch_size','dtype','max_intervention_tokens','protocol_probe_pairs',
    'protocol_probe_seed','bootstrap','permutations','seed'
]
for key in keys:
    print(f"CFG_{key.upper()}={shlex.quote(str(c[key]))}")
PY
)"

MODEL="${MODEL:-$CFG_MODEL}"
REVISION="${REVISION:-$CFG_REVISION}"
NUM_PAIRS="${NUM_PAIRS:-$CFG_NUM_PAIRS}"
MIN_PAIRS="${MIN_PAIRS:-$CFG_MIN_ELIGIBLE_PAIRS}"
ANCHOR_MIN="${ANCHOR_MIN:-$CFG_ANCHOR_MIN}"
ANCHOR_MAX="${ANCHOR_MAX:-$CFG_ANCHOR_MAX}"
BATCH_SIZE="${BATCH_SIZE:-$CFG_BATCH_SIZE}"
DTYPE="${DTYPE:-$CFG_DTYPE}"
MAX_INTERVENTION_TOKENS="${MAX_INTERVENTION_TOKENS:-$CFG_MAX_INTERVENTION_TOKENS}"
PROBE_PAIRS="${PROBE_PAIRS:-$CFG_PROTOCOL_PROBE_PAIRS}"
PROBE_SEED="${PROBE_SEED:-$CFG_PROTOCOL_PROBE_SEED}"
BOOTSTRAP="${BOOTSTRAP:-$CFG_BOOTSTRAP}"
PERMUTATIONS="${PERMUTATIONS:-$CFG_PERMUTATIONS}"
SEED="${SEED:-$CFG_SEED}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"

if [[ "$RUN_TESTS" == "1" ]]; then
  python -m unittest discover -s "$HERE/tests" -v
fi

# Fail before loading 8B weights if the shared Topic-10 environment is missing a
# dependency or if the design cannot survive tokenizer-level minimality.
python - <<'PY'
import torch, transformers, numpy
print(f"runtime deps: torch={torch.__version__} transformers={transformers.__version__} numpy={numpy.__version__}")
PY

python "$HERE/build_design.py" \
  --out "$RUN_DIR/design.jsonl" \
  --num-pairs "$NUM_PAIRS" \
  --seed "$SEED" \
  --anchor-min "$ANCHOR_MIN" \
  --anchor-max "$ANCHOR_MAX"

score_common=(
  --input "$RUN_DIR/design.jsonl"
  --model "$MODEL"
  --revision "$REVISION"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --min-pairs "$MIN_PAIRS"
  --max-intervention-tokens "$MAX_INTERVENTION_TOKENS"
  --protocol-probe-pairs "$PROBE_PAIRS"
  --protocol-probe-seed "$PROBE_SEED"
)

# Tokenizer-only audit: no model weights are loaded. This catches multi-token
# anchors/result spans and insufficient eligible pairs cheaply.
python "$HERE/score_llada.py" "${score_common[@]}" --audit-only

rm -f "$RUN_DIR"/scores.part*.jsonl "$RUN_DIR/scores.jsonl" \
      "$RUN_DIR/protocol_probe.jsonl" "$RUN_DIR/runtime.json"

resolve_gpu_id() {
  local logical="$1"
  if [[ -z "$GPU_IDS" ]]; then
    echo "$logical"
    return
  fi
  IFS=',' read -ra ids <<< "$GPU_IDS"
  if (( logical >= ${#ids[@]} )); then
    echo "GPU_IDS has only ${#ids[@]} entries but NUM_GPUS=$NUM_GPUS" >&2
    exit 2
  fi
  echo "${ids[$logical]}"
}

if [[ "$NUM_GPUS" -eq 1 ]]; then
  gpu_id="$(resolve_gpu_id 0)"
  CUDA_VISIBLE_DEVICES="$gpu_id" python "$HERE/score_llada.py" \
    "${score_common[@]}" \
    --output "$RUN_DIR/scores.jsonl" \
    --device cuda \
    --protocol-probe-output "$RUN_DIR/protocol_probe.jsonl" \
    --runtime-output "$RUN_DIR/runtime.json"
else
  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    gpu_id="$(resolve_gpu_id "$i")"
    extra=()
    if [[ "$i" -eq 0 ]]; then
      extra+=(--protocol-probe-output "$RUN_DIR/protocol_probe.jsonl")
      extra+=(--runtime-output "$RUN_DIR/runtime.json")
    fi
    CUDA_VISIBLE_DEVICES="$gpu_id" python "$HERE/score_llada.py" \
      "${score_common[@]}" \
      --output "$RUN_DIR/scores.part${i}.jsonl" \
      --device cuda \
      --shard-id "$i" \
      --num-shards "$NUM_GPUS" \
      "${extra[@]}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one scoring worker failed; refusing to analyze partial output." >&2
    exit 1
  fi
  cat "$RUN_DIR"/scores.part*.jsonl > "$RUN_DIR/scores.jsonl"
fi

if [[ ! -s "$RUN_DIR/protocol_probe.jsonl" ]]; then
  echo "Missing protocol_probe.jsonl; refusing to interpret factorial scores." >&2
  exit 1
fi

python "$HERE/analyze.py" \
  --input "$RUN_DIR/scores.jsonl" \
  --protocol-probe "$RUN_DIR/protocol_probe.jsonl" \
  --out-json "$RUN_DIR/summary.json" \
  --out-md "$RUN_DIR/summary.md" \
  --bootstrap "$BOOTSTRAP" \
  --permutations "$PERMUTATIONS" \
  --seed "$SEED"

cat "$RUN_DIR/summary.md"
