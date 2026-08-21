#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/download_data.py
python -m pytest -q
python -m memory_interference.preflight --config configs/smoke.yaml
python -m memory_interference.runner --config configs/smoke.yaml
python -m memory_interference.analyze outputs/smoke --bootstrap 200
