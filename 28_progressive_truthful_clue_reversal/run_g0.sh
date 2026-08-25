#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONHASHSEED=0
export HF_DATASETS_TRUST_REMOTE_CODE=0

python -m unittest discover -s tests -v
python g0_progressive_reversal.py --out-dir artifacts/g0 "$@"
