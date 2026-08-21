#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${BASE_URL:=http://localhost:8000}"
: "${MODEL:=local-model}"
: "${API_KEY:=EMPTY}"
: "${PAIRS:=50}"
: "${CONCURRENCY:=32}"
: "${TEMPERATURE:=0.7}"
python -m src.runner \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model "$MODEL" \
  --pairs "$PAIRS" --concurrency "$CONCURRENCY" --temperature "$TEMPERATURE" \
  --output results/pilot.jsonl
python -m src.analyze results/pilot.jsonl --out results/pilot_summary.json
