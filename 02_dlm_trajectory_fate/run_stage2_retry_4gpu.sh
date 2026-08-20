#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Resume after the historical G1-A / 200-example GSM1K preflight run.
# This intentionally does NOT rerun G1-A. See STAGE2_PROTOCOL_REVISION.md.

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_IDS <<< "$GPUS"
NUM_SHARDS=${#GPU_IDS[@]}
BOOTSTRAP=${BOOTSTRAP:-2000}
FULL_N=${FULL_N:-1205}

run_sharded() {
  local family=$1 dataset=$2 offset=$3 count=$4 out=$5
  local hidden_indices capture=(0 4 16 63)
  if [[ "$family" == "llada" ]]; then hidden_indices="25 28"; else hidden_indices="22 25"; fi
  rm -rf "$out"; mkdir -p "$out"
  pids=()
  for IDX in "${!GPU_IDS[@]}"; do
    GPU=${GPU_IDS[$IDX]}
    CUDA_VISIBLE_DEVICES="$GPU" python src/stage2_generate.py \
      --model-family "$family" --dataset "$dataset" --offset "$offset" \
      --num-examples "$count" --num-shards "$NUM_SHARDS" --shard-index "$IDX" \
      --steps 64 --gen-length 128 --block-length 32 --temperature 0 \
      --capture-steps "${capture[@]}" --hidden-indices $hidden_indices \
      --output-dir "$out" > "$out/shard_${IDX}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
  if (( status != 0 )); then
    echo "Generation failed; inspect $out/shard_*.log" >&2
    exit 1
  fi
}

support_gate() {
  local raw=$1 out=$2 label=$3
  python src/stage2_surface_gate.py \
    --input-dir "$raw" --output-dir "$out" \
    --min-positive 25 --min-negative 25
  python - "$out/locked_surface_gate.json" "$label" <<'PY'
import json, sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text()); label=sys.argv[2]
print(label, 'support:', s['status'])
for t in s['tasks']:
    print(' ', t['task'], 'pos=', t['positive'], 'neg=', t['negative'], 'ok=', t['support_ok'])
if s['status'] == 'STOP_LOW_LOCKED_SUPPORT':
    raise SystemExit(f'{label}: full-dataset locked events are too sparse; stop without fitting probes.')
PY
}

# Decisive same-model / new-data confirmation.
run_sharded llada gsm1k 0 "$FULL_N" artifacts/g1b_gsm1k_confirm/raw
support_gate artifacts/g1b_gsm1k_confirm/raw artifacts/g1b_gsm1k_confirm/support G1-B
python src/stage2_confirm.py \
  --input-dir artifacts/g1b_gsm1k_confirm/raw \
  --output-dir artifacts/g1b_gsm1k_confirm/confirm \
  --model-family llada --mode confirm --min-class-count 25 --bootstrap "$BOOTSTRAP"
python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('artifacts/g1b_gsm1k_confirm/confirm/locked_confirmation.json').read_text())
print('G1-B:', s['status'])
if s['status'] not in {'CONFIRM_BOTH','CONFIRM_ONE'}:
    raise SystemExit('G1-B did not confirm a locked hypothesis. Treat as the decisive negative; do not run Dream.')
PY

# Cross-model replication only if G1-B survives.
run_sharded dream gsm1k 0 "$FULL_N" artifacts/g1c_dream_gsm1k/raw
support_gate artifacts/g1c_dream_gsm1k/raw artifacts/g1c_dream_gsm1k/support G1-C
python src/stage2_confirm.py \
  --input-dir artifacts/g1c_dream_gsm1k/raw \
  --output-dir artifacts/g1c_dream_gsm1k/confirm \
  --model-family dream --mode model_replication --min-class-count 25 --bootstrap "$BOOTSTRAP"
