#!/usr/bin/env bash
set -euo pipefail

PINNED_UPSTREAM_COMMIT="9b01abaad354208a6a8fb26c58eb5c330036fb94"
TOPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${UPSTREAM_REPO:?Set UPSTREAM_REPO=/path/to/weakest-link-effect}"
API_URL="${API_URL:-http://localhost:8000/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_THREADS="${NUM_THREADS:-15}"
FORCE="${FORCE:-0}"

UPSTREAM_REPO="$(cd "$UPSTREAM_REPO" && pwd)"
BANK_FILE="${BANK_FILE:-$UPSTREAM_REPO/dataset/processed_musique_bank/musique_bank_18docs.jsonl}"
RECEIPT_ROOT="${RECEIPT_ROOT:-$TOPIC_DIR/artifacts/receipt}"

actual_commit="$(git -C "$UPSTREAM_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$PINNED_UPSTREAM_COMMIT" ]]; then
  echo "ERROR: upstream commit mismatch" >&2
  echo "  expected: $PINNED_UPSTREAM_COMMIT" >&2
  echo "  actual:   $actual_commit" >&2
  exit 2
fi

if [[ ! -f "$BANK_FILE" ]]; then
  echo "ERROR: bank file not found: $BANK_FILE" >&2
  echo "Use the upstream v1.0.0 release bank or regenerate it with the pinned preprocessing contract." >&2
  exit 2
fi

# We only require an OpenAI-compatible models endpoint here. The actual request
# contract still names Qwen/Qwen3-8B and will fail if the server cannot serve it.
if ! curl -fsS "${API_URL%/}/models" >/dev/null; then
  echo "ERROR: no OpenAI-compatible server at ${API_URL%/}/models" >&2
  echo "Start vLLM with Qwen/Qwen3-8B and max-model-len >= 32768." >&2
  exit 2
fi

mkdir -p "$RECEIPT_ROOT/gold_only" "$RECEIPT_ROOT/spread"
export PYTHONPATH="$UPSTREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"

force_args=()
if [[ "$FORCE" == "1" ]]; then
  force_args+=(--force-rewrite)
fi

common=(
  --bank-file "$BANK_FILE"
  --model Qwen3-8B
  --api-url "$API_URL"
  --temperature 0.0
  --top-p 1.0
  --seed 42
  --num-threads "$NUM_THREADS"
)

echo "== Receipt R1: Qwen3-8B gold-only / no-think =="
"$PYTHON_BIN" -m src.infer.entity.run_ablation \
  --mode gold_only \
  "${common[@]}" \
  --results-dir "$RECEIPT_ROOT/gold_only" \
  --prompt-id 0 \
  --max-tokens 3000 \
  "${force_args[@]}"

echo "== Receipt R2: Qwen3-8B gold-only / think =="
"$PYTHON_BIN" -m src.infer.entity.run_ablation \
  --mode gold_only \
  "${common[@]}" \
  --results-dir "$RECEIPT_ROOT/gold_only" \
  --prompt-id 0 \
  --max-tokens 10000 \
  --enable-thinking \
  "${force_args[@]}"

echo "== Receipt R3: Qwen3-8B 18-doc Spread NA / no-think =="
"$PYTHON_BIN" -m src.infer.entity.run_inference \
  --experiment spread \
  "${common[@]}" \
  --results-dir "$RECEIPT_ROOT/spread" \
  --groups beginning,midsection,tail \
  --distances 1,2,3,4,5 \
  --modes na \
  --prompt-id 22 \
  --max-tokens 3000 \
  "${force_args[@]}"

echo "== Receipt R4: Qwen3-8B 18-doc Spread NA / think =="
"$PYTHON_BIN" -m src.infer.entity.run_inference \
  --experiment spread \
  "${common[@]}" \
  --results-dir "$RECEIPT_ROOT/spread" \
  --groups beginning,midsection,tail \
  --distances 1,2,3,4,5 \
  --modes na \
  --prompt-id 22 \
  --max-tokens 10000 \
  --enable-thinking \
  "${force_args[@]}"

echo "== Receipt check =="
"$PYTHON_BIN" "$TOPIC_DIR/seed_receipt.py" \
  --bank-file "$BANK_FILE" \
  --results-root "$RECEIPT_ROOT" \
  --output "$RECEIPT_ROOT/summary.json"

verdict="$($PYTHON_BIN - "$RECEIPT_ROOT/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f)["verdict"])
PY
)"

echo "Receipt verdict: $verdict"
if [[ "$verdict" != "SEED_RELATION_REPRODUCED" ]]; then
  echo "STOP: the mandatory seed relation did not reproduce. Do not run Topic 25 G0." >&2
  exit 3
fi
