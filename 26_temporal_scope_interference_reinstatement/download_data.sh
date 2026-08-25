#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
huggingface-cli download yashkumaratri/ChronoScope merged_scope_benchmark.jsonl --repo-type dataset --local-dir data
