#!/usr/bin/env bash
# Launch the G0 behavior panel across one node's idle GPUs.
#
# Rollouts are embarrassingly parallel over (checkpoint, task shard), so this starts
# independent policy servers and collector clients. Nothing is distributed and nothing
# crosses a node boundary.
#
# Several clients share one server on purpose: inference is serial inside a server, so
# while one client is stepping MuJoCo on the CPU another can be occupying the GPU.
#
#   PHASE=discovery    bash run_g0_fleet.sh 2k 3k 9k
#   PHASE=confirmation bash run_g0_fleet.sh 2k 9k
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI="${OPENPI:-/home/xiang/projects/openpi_t09}"
# Prefer this node's local NVMe stage; fall back to the NFS copy.
if [ -z "${CKPT_ROOT:-}" ] && [ -d /tmp/t09_ckpts ]; then CKPT_ROOT=/tmp/t09_ckpts; fi
CKPT_ROOT="${CKPT_ROOT:-/home/xiang/projects/t09_ckpts}"
CLIENT_PY="${CLIENT_PY:-/home/xiang/venvs/t09_client/bin/python}"
SERVER_PY="${SERVER_PY:-$OPENPI/.venv/bin/python}"
RESULTS="${RESULTS:-$HERE/results}"
PHASE="${PHASE:-discovery}"
N_GPUS="${N_GPUS:-4}"
SERVERS_PER_CKPT="${SERVERS_PER_CKPT:-2}"
CLIENTS_PER_SERVER="${CLIENTS_PER_SERVER:-2}"
BASE_PORT="${BASE_PORT:-8100}"
TASKS_TOTAL="${TASKS_TOTAL:-10}"

case "$PHASE" in
  discovery)    INITS="0-14";  SEEDS="110000-110007" ;;
  confirmation) INITS="15-29"; SEEDS="210000-210007" ;;
  *) echo "PHASE must be discovery or confirmation" >&2; exit 1 ;;
esac

CKPTS=("$@")
[ ${#CKPTS[@]} -gt 0 ] || { echo "usage: PHASE=... bash run_g0_fleet.sh <ckpt>..." >&2; exit 1; }

LOGS="$RESULTS/logs"
mkdir -p "$RESULTS" "$LOGS"

SHARDS=$((SERVERS_PER_CKPT * CLIENTS_PER_SERVER))
# Round-robin the LIBERO tasks over shards so no shard gets only long tasks.
shard_tasks() {  # $1 = shard index, $2 = n shards
  local out=""
  for ((t = 0; t < TASKS_TOTAL; t++)); do
    (( t % $2 == $1 )) && out="${out}${out:+,}${t}"
  done
  echo "$out"
}

# openpi imports JAX even on the PyTorch path, where it is only used for jax.tree.map on
# host arrays. Left on GPU it would preallocate device memory that PyTorch then cannot use,
# which matters as soon as several servers share a card.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cleanup() { echo "shutting down fleet"; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# ---- servers -------------------------------------------------------------------------
srv=0
declare -a SERVER_PORT
for ckpt in "${CKPTS[@]}"; do
  for ((r = 0; r < SERVERS_PER_CKPT; r++)); do
    port=$((BASE_PORT + srv))
    gpu=$((srv % N_GPUS))
    SERVER_PORT[$srv]=$port
    ( cd "$HERE" && CUDA_VISIBLE_DEVICES="$gpu" "$SERVER_PY" -m src.openpi_instrumented_server \
      --config pi05_libero --checkpoint-dir "$CKPT_ROOT/pi05_pt_${ckpt}" \
      --port "$port" --device cuda:0 ) \
      >"$LOGS/server_${PHASE}_${ckpt}_r${r}.log" 2>&1 &
    echo "server ${ckpt} r${r} -> gpu ${gpu} port ${port}"
    srv=$((srv + 1))
  done
done

echo "waiting for ${srv} servers to load their checkpoints..."
for ((i = 0; i < srv; i++)); do
  ( cd "$HERE" && "$CLIENT_PY" -m src.wait_for_server --port "${SERVER_PORT[$i]}" )
done

# ---- collectors ----------------------------------------------------------------------
srv=0
for ckpt in "${CKPTS[@]}"; do
  for ((r = 0; r < SERVERS_PER_CKPT; r++)); do
    port=${SERVER_PORT[$srv]}
    for ((c = 0; c < CLIENTS_PER_SERVER; c++)); do
      shard=$((r * CLIENTS_PER_SERVER + c))
      tasks="$(shard_tasks "$shard" "$SHARDS")"
      [ -n "$tasks" ] || continue
      tag="${PHASE}_${ckpt}_s${shard}"
      ( cd "$HERE" && MUJOCO_GL=egl "$CLIENT_PY" -m src.collect_behavior \
          --port "$port" --checkpoint "$ckpt" \
          --suite libero_10 --task-ids "$tasks" \
          --init-indices "$INITS" --policy-seeds "$SEEDS" --resume \
          --out "$RESULTS/g0_${tag}.csv" ) >"$LOGS/client_${tag}.log" 2>&1 &
      echo "client ${tag} -> port ${port} (tasks ${tasks})"
    done
    srv=$((srv + 1))
  done
done

echo "fleet up. logs in $LOGS"
wait
