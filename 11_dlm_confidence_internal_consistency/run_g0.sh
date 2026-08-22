#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/configs/g0.json}"
RUN_DIR="${RUN_DIR:-$HERE/runs/g0}"

# One frozen source of defaults. Environment variables may override only for
# infrastructure (e.g. GPU count/batch size) or a predeclared confirmation run.
eval "$(python - "$CONFIG" <<'PY'
import json, shlex, sys
c=json.load(open(sys.argv[1]))
for key in ['model','num_pairs','min_eligible_pairs','anchor_min','anchor_max','batch_size','dtype','max_intervention_tokens','bootstrap','seed']:
    print(f"CFG_{key.upper()}={shlex.quote(str(c[key]))}")
PY
)"

MODEL="${MODEL:-$CFG_MODEL}"
NUM_PAIRS="${NUM_PAIRS:-$CFG_NUM_PAIRS}"
MIN_PAIRS="${MIN_PAIRS:-$CFG_MIN_ELIGIBLE_PAIRS}"
ANCHOR_MIN="${ANCHOR_MIN:-$CFG_ANCHOR_MIN}"
ANCHOR_MAX="${ANCHOR_MAX:-$CFG_ANCHOR_MAX}"
BATCH_SIZE="${BATCH_SIZE:-$CFG_BATCH_SIZE}"
DTYPE="${DTYPE:-$CFG_DTYPE}"
MAX_INTERVENTION_TOKENS="${MAX_INTERVENTION_TOKENS:-$CFG_MAX_INTERVENTION_TOKENS}"
BOOTSTRAP="${BOOTSTRAP:-$CFG_BOOTSTRAP}"
SEED="${SEED:-$CFG_SEED}"
NUM_GPUS="${NUM_GPUS:-1}"

mkdir -p "$RUN_DIR"

python "$HERE/build_design.py" \
  --out "$RUN_DIR/design.jsonl" \
  --num-pairs "$NUM_PAIRS" \
  --seed "$SEED" \
  --anchor-min "$ANCHOR_MIN" \
  --anchor-max "$ANCHOR_MAX"

score_common=(
  --input "$RUN_DIR/design.jsonl"
  --model "$MODEL"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --min-pairs "$MIN_PAIRS"
  --max-intervention-tokens "$MAX_INTERVENTION_TOKENS"
)

if [[ "$NUM_GPUS" -le 1 ]]; then
  python "$HERE/score_llada.py" \
    "${score_common[@]}" \
    --output "$RUN_DIR/scores.jsonl"
else
  rm -f "$RUN_DIR"/scores.part*.jsonl "$RUN_DIR/scores.jsonl"
  pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python "$HERE/score_llada.py" \
      "${score_common[@]}" \
      --output "$RUN_DIR/scores.part${i}.jsonl" \
      --device cuda \
      --shard-id "$i" \
      --num-shards "$NUM_GPUS" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  cat "$RUN_DIR"/scores.part*.jsonl > "$RUN_DIR/scores.jsonl"
fi

python "$HERE/analyze.py" \
  --input "$RUN_DIR/scores.jsonl" \
  --out-json "$RUN_DIR/summary.json" \
  --out-md "$RUN_DIR/summary.md" \
  --bootstrap "$BOOTSTRAP" \
  --seed "$SEED"

cat "$RUN_DIR/summary.md"
