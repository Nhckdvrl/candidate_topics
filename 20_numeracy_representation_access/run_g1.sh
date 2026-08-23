#!/usr/bin/env bash
set -euo pipefail

# Frozen Topic 20 G1 runner.
# Generates seed-0 and fresh-seed data from the exact upstream repository,
# preserves the fresh test set, then runs the preregistered rank-reflection G1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-${1:-}}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${ROOT}/20_numeracy_representation_access/artifacts/g1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PINNED_UPSTREAM="9e1be04b69965662886c79d543936389c5407d27"
FRESH_SEED="20260824"

if [[ -z "${UPSTREAM_ROOT}" ]]; then
  echo "Usage: UPSTREAM_ROOT=/path/to/Numeracy-Probing bash 20_numeracy_representation_access/run_g1.sh"
  echo "   or: bash 20_numeracy_representation_access/run_g1.sh /path/to/Numeracy-Probing"
  exit 2
fi

if [[ ! -f "${UPSTREAM_ROOT}/src/construct_data.py" ]]; then
  echo "ERROR: missing upstream src/construct_data.py under ${UPSTREAM_ROOT}" >&2
  exit 3
fi

ACTUAL_UPSTREAM="$(git -C "${UPSTREAM_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_UPSTREAM}" != "${PINNED_UPSTREAM}" ]]; then
  echo "ERROR: upstream revision mismatch" >&2
  echo " expected: ${PINNED_UPSTREAM}" >&2
  echo " actual:   ${ACTUAL_UPSTREAM}" >&2
  exit 4
fi

mkdir -p "${OUT_DIR}"
TMP="$(mktemp -d -t topic20_g1.XXXXXX)"
trap 'rm -rf "${TMP}"' EXIT
SEED0_ROOT="${TMP}/seed0"
FRESH_ROOT="${TMP}/seed${FRESH_SEED}"
mkdir -p "${SEED0_ROOT}" "${FRESH_ROOT}"

"${PYTHON_BIN}" "${UPSTREAM_ROOT}/src/construct_data.py" \
  --data_type int-sci --output_dir "${SEED0_ROOT}" --seed 0
"${PYTHON_BIN}" "${UPSTREAM_ROOT}/src/construct_data.py" \
  --data_type int-sci --output_dir "${FRESH_ROOT}" --seed "${FRESH_SEED}"

# Preserve the exact fresh confirmation set before temporary data are removed.
cp "${FRESH_ROOT}/int_sci_compare/test.jsonl" \
   "${OUT_DIR}/fresh_seed${FRESH_SEED}_test.jsonl"
sha256sum "${OUT_DIR}/fresh_seed${FRESH_SEED}_test.jsonl" \
  > "${OUT_DIR}/fresh_seed${FRESH_SEED}_test.sha256"

"${PYTHON_BIN}" "${ROOT}/20_numeracy_representation_access/g1_rank_reflection.py" \
  --seed0-data-root "${SEED0_ROOT}" \
  --fresh-data-root "${FRESH_ROOT}" \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}"

echo "G1 outputs: ${OUT_DIR}"
