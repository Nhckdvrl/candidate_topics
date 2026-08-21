#!/usr/bin/env bash
set -euo pipefail
for D in 01_behavior_vs_representation_stabilization 02_dlm_trajectory_fate 03_coverage_collapse_latent_alternatives 06_helplessness_worldview; do
  echo "== $D =="
  (cd "$D" && python -m pytest -q tests)
done
