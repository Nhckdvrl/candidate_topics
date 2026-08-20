#!/usr/bin/env bash
set -euo pipefail
for D in 01_behavior_vs_representation_stabilization 02_dlm_trajectory_fate 03_coverage_collapse_latent_alternatives; do
  echo "== $D =="
  (cd "$D" && pytest -q tests)
done
