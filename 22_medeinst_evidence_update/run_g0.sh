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
s=json.loads(Path('artifacts/g0_pair_locality/summary.json').read_text())
if s['verdict']!='PAIR_STRUCTURE_OK':
    raise SystemExit('STOP: pair locality gate failed; aligned intervention route is not clean enough')
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
v=s['verdict']
if v=='MEASUREMENT_RUNTIME_FAILURE':
    raise SystemExit('STOP: G0b measurement/runtime failure; do NOT interpret as scientific negative. Inspect invalid_reason_counts and thinking_diagnostics.')
if v!='SEED_PHENOMENON_REPRODUCED':
    raise SystemExit('STOP: repaired G0b completed with healthy measurement but the frozen seed phenomenon gate did not reproduce')
PY

python g0_bias_trap_screen.py \
  --dataset zhui711/MedEinst \
  --split test \
  --model "$MODEL" \
  --n-pairs "$N_PAIRS" \
  --seed "$SEED" \
  --mode direct \
  --outdir artifacts/g0_behavior_direct

python - <<'PY'
import json
from pathlib import Path
cot=json.loads(Path('artifacts/g0_behavior_cot/summary.json').read_text())
direct=json.loads(Path('artifacts/g0_behavior_direct/summary.json').read_text())
if cot['sample_case_ids'] != direct['sample_case_ids']:
    raise SystemExit('STOP: CoT and direct modes did not evaluate the identical fixed pair set')
if direct['verdict']=='DIRECT_MODE_MEASUREMENT_FAILURE':
    raise SystemExit('STOP: direct-mode measurement failure; do not interpret as a weak mechanism object')
if direct['verdict']!='DIRECT_MODE_MECHANISM_OBJECT_READY':
    raise SystemExit('STOP: published phenomenon may be real, but the fixed-position direct mechanism object is too weak')
print('ALL TOPIC22 G0 GATES PASSED')
PY
