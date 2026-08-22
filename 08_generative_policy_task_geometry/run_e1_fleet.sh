#!/usr/bin/env bash
# Fan E1 collection out over free GPUs on several nodes.
#
# There is no distributed training here and nothing to synchronise: each shard is an
# independent set of episode seeds. Shards must use disjoint --seed-base blocks so that
# rollout ids stay unique when the CSVs are concatenated (analyze_e1.py refuses
# duplicates).
#
# Usage:
#   bash run_e1_fleet.sh confirm            # launch
#   bash run_e1_fleet.sh confirm status     # check
set -euo pipefail

cd "$(dirname "$0")"
TAG="${1:-confirm}"
MODE="${2:-launch}"

REPO=/home/xiang/candidate_topics/08_generative_policy_task_geometry
PY=/home/xiang/venvs/topic08/bin/python
ROLLOUTS="${ROLLOUTS:-12}"
SAMPLES="${SAMPLES:-64}"
MAX_STEPS="${MAX_STEPS:-200}"

# node:gpu pairs -- edit after checking `nvidia-smi` on each host.
SHARDS=(
  "fvcrc21:0" "fvcrc21:1" "fvcrc21:2" "fvcrc21:3"
  "fvcrc10:0" "fvcrc10:1" "fvcrc10:2" "fvcrc10:3"
  "fvcrc13:0" "fvcrc13:2" "fvcrc13:3"
  "fvcrc12:0" "fvcrc12:1"
  "fvcrc20:2" "fvcrc20:3"
)

BASE="${BASE:-300000}"   # confirmation seed block, disjoint from the pilot's 100000
STRIDE=1000

if [[ "$MODE" == "status" ]]; then
  for i in "${!SHARDS[@]}"; do
    d="$REPO/results/pusht_e1_${TAG}/shard${i}"
    n=$( [[ -f "$d/probe_states.csv" ]] && echo $(( $(wc -l < "$d/probe_states.csv") - 1 )) || echo 0 )
    echo "shard$i ${SHARDS[$i]} probes=$n $( [[ -f "$d/meta.json" ]] && echo DONE || echo running )"
  done
  exit 0
fi

for i in "${!SHARDS[@]}"; do
  host="${SHARDS[$i]%%:*}"
  gpu="${SHARDS[$i]##*:}"
  out="results/pusht_e1_${TAG}/shard${i}"
  seed_base=$(( BASE + i * STRIDE ))
  mkdir -p "$REPO/$out"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" \
    "cd $REPO && CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
     nohup $PY -m src.pusht.collect_e1 --out $out --rollouts $ROLLOUTS \
       --samples $SAMPLES --max-steps $MAX_STEPS --seed-base $seed_base \
       --calib-rollouts 3 > $out/collect.log 2>&1 &" &
  echo "launched shard$i on $host gpu$gpu seed_base=$seed_base -> $out"
done
wait
echo "all shards launched; poll with: bash run_e1_fleet.sh $TAG status"
