#!/usr/bin/env bash
set -euo pipefail
LEX=${1:?usage: run_preflight.sh lexdemod.csv simplification_pairs.csv [original_col] [simplified_col]}
PAIRS=${2:?}
ORIG=${3:-original}
SIMP=${4:-simplified}
python audit_lexdemod.py --csv "$LEX" --out lexdemod_audit.json
python audit_simplification_pairs.py --input "$PAIRS" --original-col "$ORIG" --simplified-col "$SIMP" --out simplification_audit.json
python -m pytest -q tests
