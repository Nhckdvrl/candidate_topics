#!/usr/bin/env bash
set -euo pipefail

# PRE-REGISTRATION REPRODUCTION RECEIPT ONLY.
# This wrapper intentionally calls the official ACL 2026 artifact scripts.
# It does NOT implement our proposed source-transfer experiment and does NOT
# decide a new scientific threshold.

EXPECTED_REPO="JaSchuste/llm-source-preference"
EXPECTED_COMMIT="87dd466f10a76ea1cadc21a552d423d2d60c0cce"
MODEL="${MODEL:-google/gemma-3-4b-it}"
SEED="${SEED:-42}"
RESULTS_DIR="${RESULTS_DIR:-results_receipt}"
DATA_DIR="${DATA_DIR:-data}"

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "STOP: official repository must be pinned to $EXPECTED_COMMIT" >&2
  echo "current: $(git rev-parse HEAD)" >&2
  exit 2
fi

if [[ "$MODEL" != "google/gemma-3-4b-it" ]]; then
  echo "STOP: receipt model is frozen to google/gemma-3-4b-it" >&2
  exit 2
fi

if [[ "$SEED" != "42" ]]; then
  echo "STOP: receipt seed is frozen to 42" >&2
  exit 2
fi

# Dataset download/decryption is part of the artifact receipt.
# The official loader requires LLM_SP_KEY; do not silently substitute new data.
if [[ ! -f "$DATA_DIR/neoqa_entities_counterfactual.jsonl" ]]; then
  if [[ -z "${LLM_SP_KEY:-}" ]]; then
    echo "STOP: official data missing and LLM_SP_KEY is unset." >&2
    echo "Set the passphrase documented by the official artifact, then rerun." >&2
    exit 3
  fi
  python - <<'PY'
from helpers.data_loader import download_dataset
download_dataset()
PY
fi

mkdir -p "$RESULTS_DIR"

run_exp() {
  local script="$1"
  echo "============================================================"
  echo "RUN official experiment: $script"
  echo "model=$MODEL seed=$SEED data=$DATA_DIR results=$RESULTS_DIR"
  echo "============================================================"
  python "experiments/${script}.py" \
    --model "$MODEL" \
    --seed "$SEED" \
    --data-dir "$DATA_DIR" \
    --results-dir "$RESULTS_DIR"
}

# The official evaluator requires the unattributed control first.
run_exp control

# Original source-preference cells.
run_exp no_source_vs_government
run_exp government_vs_social_media

# Repetition cells used by the ACL paper.
run_exp no_source_vs_government_repeated
run_exp government_vs_social_media_repeated

# Majority-vs-repetition sanity cells from Section 5.
run_exp government_vs_social_media_1tm
run_exp government_vs_social_media_2tm

cat <<EOF

RECEIPT RUN COMPLETE.

Do NOT register a numbered topic yet.
Inspect the official aggregate JSON files under:
  $RESULTS_DIR/agg/

Record the observed Gemma-3-4B cells and compare them to the FINAL ACL 2026
paper, not the older arXiv abstract. Final-paper reference values include:

  government vs no-source teacher preference      ~29.6
  government/no-source repetition gap             ~45.9
  government vs social-media teacher preference   ~3.4
  government/social-media repetition gap          ~33.2

Also verify the Section-5 qualitative ordering:
  1-table majority gap  <<  2-table majority / same-source repetition gap.

This wrapper deliberately sets no new GO threshold. The receipt decision must
be documented against the exact final-paper cell and official evaluator output
before the novel source-transfer G0 is implemented.
EOF
