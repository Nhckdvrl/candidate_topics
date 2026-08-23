#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for D in 01_behavior_vs_representation_stabilization 02_dlm_trajectory_fate 03_coverage_collapse_latent_alternatives 06_helplessness_worldview 07_memory_interference_architecture 08_generative_policy_task_geometry 09_vla_own_limits 10_dlm_generation_order_invariance 16_citation_transmutation 17_shortcut_method_fidelity 18_negative_behavioral_adaptation; do
  echo "== $D =="
  (cd "$D" && "$REPO_ROOT/.venv/bin/python" -m pytest -q tests)
done
