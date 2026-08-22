#!/usr/bin/env bash
set -euo pipefail
PROFILE="${1:-pilot}"
SEEDS="${2:-0}"
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then RESUME_FLAG="--resume"; fi
python audit_schedule.py --profile "$PROFILE"
python launch_grid.py --profile "$PROFILE" --seeds "$SEEDS" $RESUME_FLAG
