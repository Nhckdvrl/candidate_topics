#!/usr/bin/env bash
set -euo pipefail

TOPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TOPIC_DIR"

: "${UPSTREAM_REPO:?Set UPSTREAM_REPO=/path/to/weakest-link-effect}"
API_URL="${API_URL:-http://localhost:8000/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_THREADS="${NUM_THREADS:-12}"
BANK_FILE="${BANK_FILE:-$UPSTREAM_REPO/dataset/processed_musique_bank/musique_bank_18docs.jsonl}"

export UPSTREAM_REPO API_URL PYTHON_BIN BANK_FILE

# Pure helper tests do not require a GPU, HF download, or upstream import.
"$PYTHON_BIN" -m unittest discover -s tests -v

# Mandatory official seed reproduction. This exits non-zero if the qualitative
# seed relation does not reproduce on complete support.
bash "$TOPIC_DIR/run_seed_receipt.sh"

receipt_verdict="$($PYTHON_BIN - "$TOPIC_DIR/artifacts/receipt/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f)["verdict"])
PY
)"
if [[ "$receipt_verdict" != "SEED_RELATION_REPRODUCED" ]]; then
  echo "STOP: receipt verdict is $receipt_verdict" >&2
  exit 3
fi

echo "== Topic 25 frozen matched G0 =="
"$PYTHON_BIN" "$TOPIC_DIR/g0_atomic_vs_composed.py" \
  --upstream-repo "$UPSTREAM_REPO" \
  --bank-file "$BANK_FILE" \
  --api-url "$API_URL" \
  --output-dir "$TOPIC_DIR/artifacts/g0" \
  --num-threads "$NUM_THREADS"

g0_verdict="$($PYTHON_BIN - "$TOPIC_DIR/artifacts/g0/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f)["verdict"])
PY
)"

echo "Topic 25 G0 verdict: $g0_verdict"
case "$g0_verdict" in
  GO_MATCHED_BOUNDARY)
    echo "GO: matched computation-selective thinking benefit passed all frozen gates."
    ;;
  STOP_MATCHED_BOUNDARY)
    echo "STOP: do not subset/prompt/model-shop. Archive the first boundary hypothesis." >&2
    ;;
  INCOMPLETE_ENGINEERING)
    echo "INCOMPLETE: fix only the concrete runtime/missing-call defect and resume exact missing cells." >&2
    exit 4
    ;;
  *)
    echo "ERROR: unknown G0 verdict: $g0_verdict" >&2
    exit 5
    ;;
esac
