#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "run_midtruth_geometry_4gpu.sh is kept as a compatibility alias; running the primary deterministic G0." >&2
exec ./run_pilot_4gpu.sh
