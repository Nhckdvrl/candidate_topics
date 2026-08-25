#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-data/merged_scope_benchmark.jsonl}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
mkdir -p results
python g0_temporal_scope.py prepare --data "$DATA" --panel results/g0_panel.jsonl --report results/g0_preflight.json
python g0_temporal_scope.py run --panel results/g0_panel.jsonl --out results/g0_raw.jsonl --model "$MODEL"
python g0_temporal_scope.py summarize --raw results/g0_raw.jsonl --out results/g0_summary.json
