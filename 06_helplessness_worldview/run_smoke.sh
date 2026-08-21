#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests
python -m src.runner --mock --preflight --pairs 2 --concurrency 2 --output results/smoke.jsonl
python -m src.analyze results/smoke.jsonl --bootstrap 200 --out results/smoke_summary.json
