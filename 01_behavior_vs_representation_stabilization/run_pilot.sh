#!/usr/bin/env bash
set -euo pipefail

# GATE=behavior        -> G0-A only (recommended first run)
# GATE=representation  -> G0-B only (requires behavior artifacts for joint plot)
# GATE=all             -> both
GATE=${GATE:-behavior}
NUM_EXAMPLES=${NUM_EXAMPLES:-1000}
POSITIONS_PER_TEXT=${POSITIONS_PER_TEXT:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
LAYER=${LAYER:-middle}
CORPUS=${CORPUS:-artifacts/corpus/pile_chunks_seed42.jsonl}
OUT=${OUT:-artifacts/checkpoints}
ANALYSIS=${ANALYSIS:-artifacts/analysis}

PAIR_STARTS=(2000 5000 10000 20000 50000 100000 142000)
HORIZON=1000
STEPS=()
for START in "${PAIR_STARTS[@]}"; do
  STEPS+=("$START" "$((START + HORIZON))")
done
# Deduplicate while preserving numeric order.
mapfile -t STEPS < <(printf '%s\n' "${STEPS[@]}" | sort -n -u)

mkdir -p "$OUT" "$ANALYSIS" "$(dirname "$CORPUS")"

if [[ ! -f "$CORPUS" ]]; then
  python src/prepare_corpus.py \
    --num-examples "$NUM_EXAMPLES" \
    --output "$CORPUS"
fi

run_behavior() {
  echo "[G0-A] fixed-horizon behavior reproduction"
  for STEP in "${STEPS[@]}"; do
    python src/extract_checkpoint.py \
      --step "$STEP" \
      --corpus "$CORPUS" \
      --mode behavior \
      --batch-size "$BATCH_SIZE" \
      --output-dir "$OUT"
  done
  python src/analyze.py \
    --phase behavior \
    --input-dir "$OUT" \
    --output-dir "$ANALYSIS" \
    --pair-starts "${PAIR_STARTS[@]}" \
    --horizon "$HORIZON"
}

run_representation() {
  echo "[G0-B] one-layer representation screen"
  for STEP in "${STEPS[@]}"; do
    python src/extract_checkpoint.py \
      --step "$STEP" \
      --corpus "$CORPUS" \
      --mode representation \
      --layer "$LAYER" \
      --positions-per-text "$POSITIONS_PER_TEXT" \
      --batch-size "$BATCH_SIZE" \
      --output-dir "$OUT"
  done
  python src/analyze.py \
    --phase representation \
    --input-dir "$OUT" \
    --output-dir "$ANALYSIS" \
    --pair-starts "${PAIR_STARTS[@]}" \
    --horizon "$HORIZON"
}

case "$GATE" in
  behavior) run_behavior ;;
  representation) run_representation ;;
  all) run_behavior; run_representation ;;
  *) echo "Unknown GATE=$GATE (expected behavior|representation|all)" >&2; exit 2 ;;
esac
