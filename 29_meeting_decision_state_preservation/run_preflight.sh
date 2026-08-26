#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?usage: run_preflight.sh /path/to/processed/ami}
python audit_ami.py --root "$ROOT" --out ami_audit.json
python -m pytest -q tests
