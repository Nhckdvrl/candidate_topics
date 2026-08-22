#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-full}"
SEEDS="${2:-0,1,2,3,4}"
# h=1 is already Fast; h=P is already Slow. Only intermediate persistence
# lengths belong here, and this script is conditional on a replicated G0 effect.
HVALS="${HVALS:-32,256,2048}"
IFS=',' read -ra SS <<< "$SEEDS"
IFS=',' read -ra HS <<< "$HVALS"

for s in "${SS[@]}"; do
  python train.py --mode warmup --profile "$PROFILE" --seed "$s" --resume
  for h in "${HS[@]}"; do
    python train.py --mode arm --condition persistence --persistence-h "$h" --profile "$PROFILE" --seed "$s" --resume
  done
done
