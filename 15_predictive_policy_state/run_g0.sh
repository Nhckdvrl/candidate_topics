#!/usr/bin/env bash
set -euo pipefail

: "${LIGHTWAM_ROOT:?Set LIGHTWAM_ROOT=/path/to/Light-WAM}"
: "${CKPT:?Set CKPT=/path/to/released/checkpoint.pt}"
: "${DATASET_DIR:?Set DATASET_DIR=/path/to/matching/libero_lerobot_dataset}"
: "${LATENT_CACHE_DIR:?Set LATENT_CACHE_DIR=/path/to/matching/lightwam latent cache}"
: "${TEXT_CACHE_DIR:?Set TEXT_CACHE_DIR=/path/to/text_embeds_cache/libero}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/g0}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

EXTRA=()
if [[ -n "${TRAINING_CONFIG:-}" ]]; then
  EXTRA+=(--training-config "${TRAINING_CONFIG}")
fi
if [[ -n "${DATASET_STATS:-}" ]]; then
  EXTRA+=(--dataset-stats "${DATASET_STATS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python "${SCRIPT_DIR}/g0_lightwam.py" \
  --lightwam-root "${LIGHTWAM_ROOT}" \
  --checkpoint "${CKPT}" \
  --dataset-dir "${DATASET_DIR}" \
  --latent-cache-dir "${LATENT_CACHE_DIR}" \
  --text-cache-dir "${TEXT_CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  "${EXTRA[@]}" \
  "$@"
