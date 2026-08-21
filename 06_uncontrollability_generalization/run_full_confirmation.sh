#!/usr/bin/env bash
set -euo pipefail
# Locked confirmation: 250 base seeds x four cells = 1,000 independent subject histories.
export N_SEEDS="${N_SEEDS:-250}"
export OUT="${OUT:-results/confirmation.jsonl}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-results/confirmation_analysis}"
./run_pilot.sh
