#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_IDS <<< "$GPUS"
NUM_SHARDS=${#GPU_IDS[@]}
BOOTSTRAP=${BOOTSTRAP:-2000}
FULL_N=${FULL_N:-1205}

run_sharded() {
  local family=$1 dataset=$2 offset=$3 count=$4 out=$5 surface=$6
  local hidden_indices capture=(0 4 16 63)
  if [[ "$family" == "llada" ]]; then hidden_indices="25 28"; else hidden_indices="22 25"; fi
  rm -rf "$out"; mkdir -p "$out"
  pids=()
  for IDX in "${!GPU_IDS[@]}"; do
    GPU=${GPU_IDS[$IDX]}
    extra=()
    [[ "$surface" == "1" ]] && extra+=(--surface-only)
    CUDA_VISIBLE_DEVICES="$GPU" python src/stage2_generate.py \
      --model-family "$family" --dataset "$dataset" --offset "$offset" \
      --num-examples "$count" --num-shards "$NUM_SHARDS" --shard-index "$IDX" \
      --steps 64 --gen-length 128 --block-length 32 --temperature 0 \
      --capture-steps "${capture[@]}" --hidden-indices $hidden_indices \
      --output-dir "$out" "${extra[@]}" \
      > "$out/shard_${IDX}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
  if (( status != 0 )); then
    echo "Generation failed; inspect $out/shard_*.log" >&2
    exit 1
  fi
}

check_support() {
  local gate_json=$1 label=$2
  python - "$gate_json" "$label" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); label=sys.argv[2]
s=json.loads(p.read_text())
print(label, 'support:', s['status'])
for t in s['tasks']:
    print(' ', t['task'], 'pos=', t['positive'], 'neg=', t['negative'], 'ok=', t['support_ok'])
if s['status'] == 'STOP_LOW_LOCKED_SUPPORT':
    raise SystemExit(f'{label}: both locked tasks have insufficient full-dataset support; stop.')
PY
}

check_confirmation() {
  local result_json=$1 label=$2
  python - "$result_json" "$label" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); label=sys.argv[2]
s=json.loads(p.read_text())
print(label, 'confirmation:', s['status'])
if s['status'] not in {'CONFIRM_BOTH','CONFIRM_ONE'}:
    raise SystemExit(f'{label}: locked confirmation did not survive; stop before the next model.')
PY
}

# G1-A: untouched GSM8K tail. This remains a directional audit only.
run_sharded llada gsm8k 1000 319 artifacts/g1a_gsm8k_holdout/raw 0
python src/stage2_confirm.py \
  --input-dir artifacts/g1a_gsm8k_holdout/raw \
  --output-dir artifacts/g1a_gsm8k_holdout/confirm \
  --model-family llada --mode audit --min-class-count 8 --bootstrap "$BOOTSTRAP"

# G1-B: decisive independent-data confirmation on ALL GSM1K examples.
# Protocol revision: no 200-example stopping gate. We generate the 1,205 locked
# trajectories once, capture only 4 steps x 2 layers, inspect surface support
# first, and fit hidden probes only if at least one locked task has >=25/25 support.
run_sharded llada gsm1k 0 "$FULL_N" artifacts/g1b_gsm1k_confirm/raw 0
python src/stage2_surface_gate.py \
  --input-dir artifacts/g1b_gsm1k_confirm/raw \
  --output-dir artifacts/g1b_gsm1k_confirm/support \
  --min-positive 25 --min-negative 25
check_support artifacts/g1b_gsm1k_confirm/support/locked_surface_gate.json G1-B
python src/stage2_confirm.py \
  --input-dir artifacts/g1b_gsm1k_confirm/raw \
  --output-dir artifacts/g1b_gsm1k_confirm/confirm \
  --model-family llada --mode confirm --min-class-count 25 --bootstrap "$BOOTSTRAP"
check_confirmation artifacts/g1b_gsm1k_confirm/confirm/locked_confirmation.json G1-B

# G1-C: only after same-model independent-data confirmation survives.
# Dream likewise uses the full GSM1K support count, not a noisy 200-example gate.
run_sharded dream gsm1k 0 "$FULL_N" artifacts/g1c_dream_gsm1k/raw 0
python src/stage2_surface_gate.py \
  --input-dir artifacts/g1c_dream_gsm1k/raw \
  --output-dir artifacts/g1c_dream_gsm1k/support \
  --min-positive 25 --min-negative 25
check_support artifacts/g1c_dream_gsm1k/support/locked_surface_gate.json G1-C
python src/stage2_confirm.py \
  --input-dir artifacts/g1c_dream_gsm1k/raw \
  --output-dir artifacts/g1c_dream_gsm1k/confirm \
  --model-family dream --mode model_replication --min-class-count 25 --bootstrap "$BOOTSTRAP"
