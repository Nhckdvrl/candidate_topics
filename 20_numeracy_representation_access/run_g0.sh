#!/usr/bin/env bash
set -euo pipefail

# Registered Topic 20 entrypoint.
# The scientific implementation is kept in advisor_topic_search/g0 so the
# search-stage preregistration and registered candidate cannot silently diverge.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${1:-}}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-${ROOT}/20_numeracy_representation_access/results/g0_qwen3_8b}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: DATA_ROOT=/path/to/Numeracy-Probing/data bash 20_numeracy_representation_access/run_g0.sh"
  echo "   or: bash 20_numeracy_representation_access/run_g0.sh /path/to/Numeracy-Probing/data"
  exit 2
fi

python "${ROOT}/advisor_topic_search/g0/numeracy_same_prompt_g0.py" \
  --data-root "${DATA_ROOT}" \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}"
