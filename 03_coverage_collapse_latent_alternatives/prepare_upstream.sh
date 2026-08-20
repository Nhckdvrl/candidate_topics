#!/usr/bin/env bash
set -euo pipefail

UPSTREAM=${UPSTREAM:-external/reasoning_forks}
# Pin the audited upstream snapshot. Floating main would silently change the generator,
# prompts or training/eval code underneath the G0 experiment.
UPSTREAM_COMMIT=${UPSTREAM_COMMIT:-64bf9e3e86231bc6b52f2974ca285ad8aa8fc181}
if [[ ! -d "$UPSTREAM/.git" ]]; then
  mkdir -p "$(dirname "$UPSTREAM")"
  git clone https://github.com/NNHieu/reasoning_forks.git "$UPSTREAM"
fi
if [[ -n "$(git -C "$UPSTREAM" status --porcelain)" ]]; then
  echo "Refusing to change dirty upstream checkout: $UPSTREAM" >&2
  exit 1
fi
git -C "$UPSTREAM" fetch origin "$UPSTREAM_COMMIT" --quiet
git -C "$UPSTREAM" checkout --detach "$UPSTREAM_COMMIT" --quiet
ACTUAL=$(git -C "$UPSTREAM" rev-parse HEAD)
[[ "$ACTUAL" == "$UPSTREAM_COMMIT" ]] || { echo "Upstream pin failed: $ACTUAL" >&2; exit 1; }

# Generate the exact 6400 SFT / 1600 RLVR / 1000 test Graph Branching split.
(cd "$UPSTREAM" && python src/data_generation/gen_arithchain.py)
python src/prepare_forks.py \
  --input "$UPSTREAM/datasets/arithchain_2_10/test.parquet" \
  --output artifacts/forks.jsonl \
  --limit 1000
