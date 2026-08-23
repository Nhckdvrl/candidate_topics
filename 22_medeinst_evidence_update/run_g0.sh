#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-14B}"
N_PAIRS="${N_PAIRS:-256}"
SEED="${SEED:-20260823}"

python -m pytest -q tests/test_g0_helpers.py

python g0_pair_locality.py \
  --dataset zhui711/MedEinst \
  --split test \
  --outdir artifacts/g0_pair_locality

python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/g0_pair_locality/summary.json')
s=json.loads(p.read_text())
if s['verdict']!='PAIR_STRUCTURE_OK':
    raise SystemExit('STOP: pair locality gate failed; do not proceed to mechanism screening')
PY

python g0_bias_trap_screen.py \
  --dataset zhui711/MedEinst \
  --split test \
  --model "$MODEL" \
  --n-pairs "$N_PAIRS" \
  --seed "$SEED" \
  --mode cot \
  --outdir artifacts/g0_behavior_cot

python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('artifacts/g0_behavior_cot/summary.json').read_text())
if s['verdict']!='SEED_PHENOMENON_REPRODUCED':
    raise SystemExit('STOP: seed-faithful zero-shot CoT Bias Trap phenomenon did not reproduce')
PY

python g0_bias_trap_screen.py \
  --dataset zhui711/MedEinst \
  --split test \
  --model "$MODEL" \
  --n-pairs "$N_PAIRS" \
  --seed "$SEED" \
  --mode direct \
  --outdir artifacts/g0_behavior_direct
