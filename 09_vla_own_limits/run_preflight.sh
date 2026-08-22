#!/usr/bin/env bash
# P0 technical gate: prove the inference stack before any scientific number is collected.
#
#   bash run_preflight.sh 2k 3k 9k
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="${OPENPI:-/home/xiang/projects/openpi_t09}"
CKPT_ROOT="${CKPT_ROOT:-/home/xiang/projects/t09_ckpts}"
CLIENT_PY="${CLIENT_PY:-/home/xiang/venvs/t09_client/bin/python}"
SERVER_PY="${SERVER_PY:-$OPENPI/.venv/bin/python}"
RESULTS="${RESULTS:-$HERE/results}"
BASE_PORT="${BASE_PORT:-8200}"
N_GPUS="${N_GPUS:-4}"

CKPTS=("$@")
[ ${#CKPTS[@]} -ge 2 ] || { echo "give at least two checkpoints" >&2; exit 1; }

LOGS="$RESULTS/logs"
mkdir -p "$RESULTS" "$LOGS"
cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

PORTS=()
for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$i]}"
  port=$((BASE_PORT + i))
  PORTS+=("$port")
  ( cd "$HERE" && CUDA_VISIBLE_DEVICES=$((i % N_GPUS)) "$SERVER_PY" -m src.openpi_instrumented_server \
    --config pi05_libero --checkpoint-dir "$CKPT_ROOT/pi05_pt_${ckpt}" \
    --port "$port" --device cuda:0 ) >"$LOGS/preflight_server_${ckpt}.log" 2>&1 &
  echo "server ${ckpt} -> port ${port}"
done

for p in "${PORTS[@]}"; do ( cd "$HERE" && "$CLIENT_PY" -m src.wait_for_server --port "$p" ); done

# 1. per-checkpoint identity: state hash, RNG control, feature capture
for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$i]}"
  echo "=== preflight ${ckpt} ==="
  ( cd "$HERE" && MUJOCO_GL=egl "$CLIENT_PY" -m src.preflight \
      --port "${PORTS[$i]}" --out "$RESULTS/preflight_${ckpt}.json" )
done

# 2. the checkpoints must not be the same model
echo "=== checkpoint distinctness ==="
( cd "$HERE" && MUJOCO_GL=egl "$CLIENT_PY" -m src.check_checkpoints_differ \
    --ports "${PORTS[@]}" --names "${CKPTS[@]}" \
    --out "$RESULTS/checkpoints_differ.json" )

echo "PREFLIGHT_OK"
