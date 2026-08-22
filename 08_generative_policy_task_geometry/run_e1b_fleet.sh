#!/usr/bin/env bash
# Fan the E1b branching experiment out over free GPUs on several nodes.
#
# Nothing is synchronised: each shard runs its own disjoint block of episode seeds and
# writes its own CSV. Shard seed blocks must not overlap, because the analysis refuses
# duplicate (rollout, probe) keys.
#
#   bash run_e1b_fleet.sh <tag> [launch|status] 
set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:-disc}"
MODE="${2:-launch}"
REPO=/home/xiang/candidate_topics/08_generative_policy_task_geometry
PY=/home/xiang/venvs/topic08/bin/python
CKPT="${CKPT:-/home/xiang/shared_assets/pusht_diffusion_policy/ckpt/diffusion_pusht}"

ROLLOUTS="${ROLLOUTS:-8}"
BRANCHES="${BRANCHES:-32}"
EXTRA="${EXTRA:-88}"
PROBE_EVERY="${PROBE_EVERY:-3}"
SAMPLES="${SAMPLES:-256}"
BASE="${BASE:-200000}"
STRIDE=1000

SHARDS=(
  "fvcrc21:0" "fvcrc21:2" "fvcrc21:3"
  "fvcrc10:0" "fvcrc10:1" "fvcrc10:2" "fvcrc10:3"
  "fvcrc13:0" "fvcrc13:2" "fvcrc13:3"
  "fvcrc12:0" "fvcrc12:1"
  "fvcrc20:2" "fvcrc20:3"
)

if [[ "$MODE" == "status" ]]; then
  tot=0
  for i in "${!SHARDS[@]}"; do
    d="$REPO/results/pusht_e1b_${TAG}/shard${i}"
    n=0; [[ -f "$d/branch_states.csv" ]] && n=$(( $(wc -l < "$d/branch_states.csv") - 1 ))
    tot=$(( tot + n ))
    printf "shard%-3s %-12s states=%-5s %s\n" "$i" "${SHARDS[$i]}" "$n" \
      "$( [[ -f "$d/meta.json" ]] && echo DONE || echo running )"
  done
  echo "total branch states: $tot"
  exit 0
fi

for i in "${!SHARDS[@]}"; do
  host="${SHARDS[$i]%%:*}"; gpu="${SHARDS[$i]##*:}"
  out="results/pusht_e1b_${TAG}/shard${i}"
  seed_base=$(( BASE + i * STRIDE ))
  mkdir -p "$REPO/$out"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" \
    "cd $REPO && CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
     nohup $PY -m src.pusht.collect_e1b --pretrained $CKPT --out $out \
       --rollouts $ROLLOUTS --branches $BRANCHES --extra-steps $EXTRA \
       --probe-every $PROBE_EVERY --samples $SAMPLES --seed-base $seed_base \
       --calib-rollouts 2 > $out/collect.log 2>&1 &" &
  echo "shard$i -> $host gpu$gpu seed_base=$seed_base"
done
wait
echo "launched; poll: bash run_e1b_fleet.sh $TAG status"
