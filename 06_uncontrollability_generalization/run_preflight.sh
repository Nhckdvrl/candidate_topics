#!/usr/bin/env bash
set -euo pipefail
python -m src.audit_environment --episodes "${EPISODES:-2000}" --steps "${STEPS:-60}"
pytest -q tests
