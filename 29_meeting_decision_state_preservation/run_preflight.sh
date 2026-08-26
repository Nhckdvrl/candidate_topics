#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?usage: run_preflight.sh /path/to/processed/ami}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR}
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_ami.py" --root "$ROOT" --out "$OUT_DIR/ami_audit.json"
"$PYTHON_BIN" -m pytest -q "$SCRIPT_DIR/tests"
