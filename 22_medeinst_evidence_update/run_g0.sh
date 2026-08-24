#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-14B}"
N_PAIRS="${N_PAIRS:-256}"
SEED="${SEED:-20260823}"
V2_COT_RECORDS="${V2_COT_RECORDS:-artifacts/g0_behavior_cot/records.jsonl}"

python -m pytest -q tests/test_g0_helpers.py tests/test_g0_v3_canonicalizer.py

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

# G0b-v3 is deliberately a SCORING-ONLY repair of the already-generated v2 CoT outputs.
# Do not silently regenerate CoT after seeing the v2 result; preserve the frozen model outputs.
if [[ ! -f "$V2_COT_RECORDS" ]]; then
  echo "STOP: frozen v2 CoT records not found at $V2_COT_RECORDS" >&2
  echo "G0b-v3 must recanonicalize the original v2 records; do not regenerate them implicitly." >&2
  exit 2
fi

python g0_recanonicalize_v3.py \
  --input-records "$V2_COT_RECORDS" \
  --dataset zhui711/MedEinst \
  --split test \
  --model "$MODEL" \
  --mode cot \
  --outdir artifacts/g0_behavior_cot_v3

python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('artifacts/g0_behavior_cot_v3/summary.json').read_text())
v=s['verdict']
if v in {'CANONICALIZER_PREFLIGHT_FAILURE','MEASUREMENT_CANONICALIZATION_FAILURE'}:
    raise SystemExit('STOP: G0b-v3 closed-label measurement is still unhealthy; do not interpret as a scientific negative.')
if v!='SEED_PHENOMENON_REPRODUCED':
    raise SystemExit('STOP: G0b-v3 measurement is healthy but the frozen MedEinst seed phenomenon did not reproduce.')
PY

# Only after CoT reproduction is measurement-healthy do we test the mechanism-tractable direct regime.
# Direct generation remains the same fixed 256 pairs/model/seed; v3 canonicalization is applied afterwards.
python g0_bias_trap_screen.py \
  --dataset zhui711/MedEinst \
  --split test \
  --model "$MODEL" \
  --n-pairs "$N_PAIRS" \
  --seed "$SEED" \
  --mode direct \
  --outdir artifacts/g0_behavior_direct_raw_v3

python g0_recanonicalize_v3.py \
  --input-records artifacts/g0_behavior_direct_raw_v3/records.jsonl \
  --dataset zhui711/MedEinst \
  --split test \
  --model "$MODEL" \
  --mode direct \
  --outdir artifacts/g0_behavior_direct_v3

python - <<'PY'
import json
from pathlib import Path
cot=json.loads(Path('artifacts/g0_behavior_cot_v3/summary.json').read_text())
direct=json.loads(Path('artifacts/g0_behavior_direct_v3/summary.json').read_text())
if cot['sample_case_ids'] != direct['sample_case_ids']:
    raise SystemExit('STOP: CoT and direct modes did not evaluate the identical fixed pair set')
if direct['verdict'] in {'CANONICALIZER_PREFLIGHT_FAILURE','DIRECT_MODE_CANONICALIZATION_FAILURE'}:
    raise SystemExit('STOP: direct-mode closed-label measurement is unhealthy; do not interpret as a weak mechanism object')
if direct['verdict']!='DIRECT_MODE_MECHANISM_OBJECT_READY':
    raise SystemExit('STOP: MedEinst may reproduce under CoT, but the fixed-position direct mechanism object is too weak')
print('ALL TOPIC22 G0 GATES PASSED UNDER V3 MEASUREMENT')
PY
