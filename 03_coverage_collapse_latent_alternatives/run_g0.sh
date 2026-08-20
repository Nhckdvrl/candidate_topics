#!/usr/bin/env bash
set -euo pipefail

[[ -f artifacts/forks.jsonl ]] || ./prepare_upstream.sh
RUN=${RUN:-external/reasoning_forks/runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1}
if [[ ! -d "$RUN/checkpoint-3200" ]]; then
  cat >&2 <<EOF
Missing official forward-SFT checkpoints.
Run once:
  (cd external/reasoning_forks && ./run_sft.sh arithchain_2_10_forward qwen2.5_0.5b 16)
Then rerun ./run_g0.sh.
EOF
  exit 1
fi

# Optional teacher-forced debugging preflight. Disabled by default because the sampled
# behavior gate is the actual premise test. Enable only when diagnosing a failed run.
if [[ "${RUN_STATE_PREFLIGHT:-0}" == "1" ]]; then
  ./run_state_preflight.sh
fi

# Scientific gate 1: reproduce coverage shrinkage + first-fork polarization cheaply.
./run_behavior_preflight.sh
# Scientific gate 2: only if gate 1 passes, test whether latent branch viability survives
# specifically on late examples whose normal output readout commits to the wrong branch.
./run_latent_gate.sh
