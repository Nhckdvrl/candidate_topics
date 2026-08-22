#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-pilot}"
if [[ $# -ge 2 ]]; then
  SEEDS="$2"
elif [[ "$PROFILE" == "full" ]]; then
  SEEDS="0,1,2,3,4"
elif [[ "$PROFILE" == "paper_anchor" ]]; then
  SEEDS="0,1,2"
else
  SEEDS="0"
fi

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then RESUME_FLAG="--resume"; fi

if [[ "$PROFILE" != "paper_anchor" ]]; then
  python audit_schedule.py --profile "$PROFILE" --seeds "$SEEDS"
fi
python launch_grid.py --profile "$PROFILE" --seeds "$SEEDS" $RESUME_FLAG
