#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
if [[ ! -d "$UPSTREAM/.git" ]]; then
  mkdir -p "$(dirname "$UPSTREAM")"
  git clone https://github.com/NNHieu/reasoning_forks.git "$UPSTREAM"
fi

# Generate the official 6400 SFT / 1600 RLVR / 1000 test Graph Branching split.
(cd "$UPSTREAM" && python src/data_generation/gen_arithchain.py)
python src/prepare_forks.py \
  --input "$UPSTREAM/datasets/arithchain_2_10/test.parquet" \
  --output artifacts/forks.jsonl \
  --limit 1000
