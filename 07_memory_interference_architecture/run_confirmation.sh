#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/download_data.py
python -m pytest -q
python -m memory_interference.preflight --config configs/confirm.yaml
python -m memory_interference.runner --config configs/confirm.yaml "$@"
python -m memory_interference.analyze outputs/architecture_pi_ri_confirmation --bootstrap 5000
python -m memory_interference.decide outputs/architecture_pi_ri_confirmation --bootstrap 10000
