#!/usr/bin/env bash
# Launch the G0 behavior panel across the idle GPUs of one node.
#
# Rollouts are embarrassingly parallel over (checkpoint, task shard), so this starts
# several independent policy servers and one collector client per server. There is no
# cross-node or distributed anything -- each stream is a self-contained process pair.
#
# Usage:
#   PHASE=discovery bash run_g0_fleet.sh 2k 3k 9k
#   PHASE=confirmation bash run_g0_fleet.sh 2k 9k
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="${OPENPI:-/home/xiang/projects/openpi_t09}"
CKPT_ROOT="${CKPT_ROOT:-/home/xiang/projects/t09_ckpts}"
CLIENT_PY="${CLIENT_PY:-/home/xiang/venvs/t09_client/bin/python}"
SERVER_PY="${SERVER_PY:-$OPENPI/.venv/bin/python}"
RESULTS="${RESULTS:-$HERE/results}"
LOGS="$RESULTS/logs"
PHASE="${PHASE:-discovery}"
N_GPUS="${N_GPUS:-4}"
SHARDS_PER_CKPT="${SHARDS_PER_CKPT:-4}"   # task shards, also = servers per checkpoint
BASE_PORT="${BASE_PORT:-8100}"

case "$PHASE" in
  discovery)    INITS="0-14";  SEEDS="110000-110007" ;;
  confirmation) INITS="15-29"; SEEDS="210000-210007" ;;
  *) echo "PHASE must be discovery or confirmation" >&2; exit 1 ;;
esac

CKPTS=("$@")
[ ${#CKPTS[@]} -gt 0 ] || { echo "give at least one checkpoint name" >&2; exit 1; }

# Ten LIBERO-10 tasks split into SHARDS_PER_CKPT contiguous groups.
shard_tasks() {  # $1 = shard index, $2 = n shards
  local i=$1 n=$2 out=""
  for t in $(seq 0 9); do [ $((t % n)) -eq "$i" ] && out="${out}${out:+,}${t}"; done
  echo "$out"
}

mkdir -p "$RESULTS" "$LOGS"
PIDS=()
idx=0
for ckpt in "${CKPTS[@]}"; do
  for s in $(seq 0 $((SHARDS_PER_CKPT - 1))); do
    port=$((BASE_PORT + idx))
    gpu=$((idx % N_GPUS))
    tasks="$(shard_tasks "$s" "$SHARDS_PER_CKPT")"
    tag="${PHASE}_${ckpt}_s${s}"

    CUDA_VISIBLE_DEVICES="$gpu" "$SERVER_PY" -m src.openpi_instrumented_server \
      --config pi05_libero --checkpoint-dir "$CKPT_ROOT/pt_${ckpt}" \
      --port "$port" --device cuda:0 \
      >"$LOGS/server_${tag}.log" 2>&1 &
    PIDS+=($!)
    echo "server $tag -> gpu $gpu port $port (tasks $tasks)"
    idx=$((idx + 1))
  done
done

echo "waiting for servers to load checkpoints..."
for p in $(seq 0 $((idx - 1))); do
  port=$((BASE_PORT + p))
  for _ in $(seq 1 180); do
    "$CLIENT_PY" - "$port" <<'PY' && break || sleep 10
import socket, sys
s = socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
  done
done

idx=0
for ckpt in "${CKPTS[@]}"; do
  for s in $(seq 0 $((SHARDS_PER_CKPT - 1))); do
    port=$((BASE_PORT + idx))
    tasks="$(shard_tasks "$s" "$SHARDS_PER_CKPT")"
    tag="${PHASE}_${ckpt}_s${s}"
    ( cd "$HERE" && MUJOCO_GL=egl "$CLIENT_PY" -m src.collect_behavior \
        --port "$port" --checkpoint "$ckpt" \
        --suite libero_10 --task-ids "$tasks" \
        --init-indices "$INITS" --policy-seeds "$SEEDS" --resume \
        --out "$RESULTS/g0_${tag}.csv" ) >"$LOGS/client_${tag}.log" 2>&1 &
    PIDS+=($!)
    echo "client $tag -> port $port"
    idx=$((idx + 1))
  done
done

echo "fleet up: ${#PIDS[@]} processes. logs in $LOGS"
wait
