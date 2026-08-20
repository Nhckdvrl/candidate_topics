#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_IDS <<< "$GPUS"
NUM_SHARDS=${#GPU_IDS[@]}
BOOTSTRAP=${BOOTSTRAP:-2000}

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

# G1-A: untouched in-distribution selection audit (G0 used GSM8K ids 0..999).
run_sharded llada gsm8k 1000 319 artifacts/g1a_gsm8k_holdout/raw 0
python src/stage2_confirm.py \
  --input-dir artifacts/g1a_gsm8k_holdout/raw \
  --output-dir artifacts/g1a_gsm8k_holdout/confirm \
  --model-family llada --mode audit --min-class-count 8 --bootstrap "$BOOTSTRAP"

# G1-B: decisive independent-data confirmation on distribution-matched GSM1K.
run_sharded llada gsm1k 0 200 artifacts/g1b_gsm1k_preflight/raw 1
python src/stage2_surface_gate.py \
  --input-dir artifacts/g1b_gsm1k_preflight/raw \
  --output-dir artifacts/g1b_gsm1k_preflight \
  --min-positive 6 --min-negative 20
python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('artifacts/g1b_gsm1k_preflight/locked_surface_gate.json').read_text())
if s['status'] == 'STOP_LOW_LOCKED_SUPPORT':
    raise SystemExit('GSM1K locked transient events are too rare; stop before full hidden confirmation.')
PY

run_sharded llada gsm1k 0 1205 artifacts/g1b_gsm1k_confirm/raw 0
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
    raise SystemExit('Same-model independent-data confirmation failed; do not spend Dream replication yet.')
PY

# G1-C: cross-model replication. Dream uses official deterministic maskgit_plus.
run_sharded dream gsm1k 0 200 artifacts/g1c_dream_preflight/raw 1
python src/stage2_surface_gate.py \
  --input-dir artifacts/g1c_dream_preflight/raw \
  --output-dir artifacts/g1c_dream_preflight \
  --min-positive 6 --min-negative 20
python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('artifacts/g1c_dream_preflight/locked_surface_gate.json').read_text())
if s['status'] == 'STOP_LOW_LOCKED_SUPPORT':
    raise SystemExit('Dream deterministic geometry has too little locked transient support; record as limited generality.')
PY

run_sharded dream gsm1k 0 1205 artifacts/g1c_dream_gsm1k/raw 0
python src/stage2_confirm.py \
  --input-dir artifacts/g1c_dream_gsm1k/raw \
  --output-dir artifacts/g1c_dream_gsm1k/confirm \
  --model-family dream --mode model_replication --min-class-count 25 --bootstrap "$BOOTSTRAP"
