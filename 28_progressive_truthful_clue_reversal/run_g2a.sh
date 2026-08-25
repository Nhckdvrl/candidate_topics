#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=0

python -m unittest discover -s tests -v
python g2a_destructive_conjunction.py --out-dir artifacts/g2a "$@"
