#!/usr/bin/env bash
set -euo pipefail
# Full confirmation only. The 200x16 preflight should be run first via ./run_g0.sh.
NUM_PROBLEMS=${NUM_PROBLEMS:-1000} \
NUM_SAMPLES=${NUM_SAMPLES:-64} \
TAGS=${TAGS:-e01,e02,e04,e08,e16} \
RUN_ID=${RUN_ID:-full_$(date +%Y%m%d_%H%M%S)} \
./run_behavior_preflight.sh
