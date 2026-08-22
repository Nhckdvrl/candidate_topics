#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/configs/g0.json}"
RUN_DIR="${RUN_DIR:-$HERE/runs/g0}"
RUN_TESTS="${RUN_TESTS:-1}"

eval "$(python - "$CONFIG" <<'PY'
import json,shlex,sys
c=json.load(open(sys.argv[1]))
for k in ['model','revision','num_pairs','min_eligible_pairs','anchor_min','anchor_max','batch_size','dtype','max_intervention_tokens','protocol_probe_pairs','protocol_probe_seed','min_arithmetic_probe_gap','min_alias_probe_gap','min_primary_effect','bootstrap','permutations','seed']:
    print(f"CFG_{k.upper()}={shlex.quote(str(c[k]))}")
PY
)"

# Scientific knobs are locked to CONFIG. Only infrastructure knobs may vary.
MODEL="$CFG_MODEL"
REVISION="$CFG_REVISION"
NUM_PAIRS="$CFG_NUM_PAIRS"
MIN_PAIRS="$CFG_MIN_ELIGIBLE_PAIRS"
ANCHOR_MIN="$CFG_ANCHOR_MIN"
ANCHOR_MAX="$CFG_ANCHOR_MAX"
BATCH_SIZE="${BATCH_SIZE:-$CFG_BATCH_SIZE}"
DTYPE="$CFG_DTYPE"
MAX_INTERVENTION_TOKENS="$CFG_MAX_INTERVENTION_TOKENS"
PROBE_PAIRS="$CFG_PROTOCOL_PROBE_PAIRS"
PROBE_SEED="$CFG_PROTOCOL_PROBE_SEED"
MIN_ARITHMETIC_PROBE_GAP="$CFG_MIN_ARITHMETIC_PROBE_GAP"
MIN_ALIAS_PROBE_GAP="$CFG_MIN_ALIAS_PROBE_GAP"
MIN_PRIMARY_EFFECT="$CFG_MIN_PRIMARY_EFFECT"
BOOTSTRAP="$CFG_BOOTSTRAP"
PERMUTATIONS="$CFG_PERMUTATIONS"
SEED="$CFG_SEED"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"

[[ "$NUM_GPUS" -ge 1 ]] || { echo "NUM_GPUS must be >=1" >&2; exit 2; }
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/locked_config.json"
git -C "$HERE" rev-parse HEAD > "$RUN_DIR/repo_commit.txt" 2>/dev/null || true

if [[ "$RUN_TESTS" == "1" ]]; then
  python -m unittest discover -s "$HERE/tests" -v
fi

python - <<'PY'
import torch,transformers,numpy
print(f"runtime deps: torch={torch.__version__} transformers={transformers.__version__} numpy={numpy.__version__}")
PY

python "$HERE/build_design.py" --out "$RUN_DIR/design.jsonl" --num-pairs "$NUM_PAIRS" --seed "$SEED" --anchor-min "$ANCHOR_MIN" --anchor-max "$ANCHOR_MAX"

score_common=(
  --input "$RUN_DIR/design.jsonl" --model "$MODEL" --revision "$REVISION"
  --batch-size "$BATCH_SIZE" --dtype "$DTYPE" --min-pairs "$MIN_PAIRS"
  --max-intervention-tokens "$MAX_INTERVENTION_TOKENS"
  --protocol-probe-pairs "$PROBE_PAIRS" --protocol-probe-seed "$PROBE_SEED"
)

# Tokenizer/design audit before loading 8B weights.
python "$HERE/score_llada.py" "${score_common[@]}" --audit-only

rm -f "$RUN_DIR"/scores.part*.jsonl "$RUN_DIR/scores.jsonl" "$RUN_DIR/protocol_probe.jsonl" "$RUN_DIR/runtime.json"

resolve_gpu_id() {
  local logical="$1"
  if [[ -z "$GPU_IDS" ]]; then echo "$logical"; return; fi
  IFS=',' read -ra ids <<< "$GPU_IDS"
  (( logical < ${#ids[@]} )) || { echo "GPU_IDS too short" >&2; exit 2; }
  echo "${ids[$logical]}"
}

if [[ "$NUM_GPUS" -eq 1 ]]; then
  gpu_id="$(resolve_gpu_id 0)"
  CUDA_VISIBLE_DEVICES="$gpu_id" python "$HERE/score_llada.py" "${score_common[@]}" --output "$RUN_DIR/scores.jsonl" --device cuda --protocol-probe-output "$RUN_DIR/protocol_probe.jsonl" --runtime-output "$RUN_DIR/runtime.json"
else
  pids=()
  for ((i=0;i<NUM_GPUS;i++)); do
    gpu_id="$(resolve_gpu_id "$i")"; extra=()
    if [[ "$i" -eq 0 ]]; then extra+=(--protocol-probe-output "$RUN_DIR/protocol_probe.jsonl"); extra+=(--runtime-output "$RUN_DIR/runtime.json"); fi
    CUDA_VISIBLE_DEVICES="$gpu_id" python "$HERE/score_llada.py" "${score_common[@]}" --output "$RUN_DIR/scores.part${i}.jsonl" --device cuda --shard-id "$i" --num-shards "$NUM_GPUS" "${extra[@]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || { echo "scoring worker failed; refusing partial analysis" >&2; exit 1; }
  cat "$RUN_DIR"/scores.part*.jsonl > "$RUN_DIR/scores.jsonl"
fi

[[ -s "$RUN_DIR/protocol_probe.jsonl" ]] || { echo "missing protocol probes" >&2; exit 1; }
[[ -s "$RUN_DIR/scores.jsonl" ]] || { echo "missing factorial scores" >&2; exit 1; }

python "$HERE/analyze.py" --input "$RUN_DIR/scores.jsonl" --protocol-probe "$RUN_DIR/protocol_probe.jsonl" --out-json "$RUN_DIR/summary.json" --out-md "$RUN_DIR/summary.md" --bootstrap "$BOOTSTRAP" --permutations "$PERMUTATIONS" --seed "$SEED" --min-arithmetic-probe-gap "$MIN_ARITHMETIC_PROBE_GAP" --min-alias-probe-gap "$MIN_ALIAS_PROBE_GAP" --min-primary-effect "$MIN_PRIMARY_EFFECT"

cat "$RUN_DIR/summary.md"
