#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PREFLIGHT_EXAMPLES=${PREFLIGHT_EXAMPLES:-200}
FULL_EXAMPLES=${FULL_EXAMPLES:-1000}

NUM_EXAMPLES="$PREFLIGHT_EXAMPLES" ./run_surface_preflight_4gpu.sh

python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/preflight_midtruth/surface_summary.json")
summary = json.loads(p.read_text())
print("preflight:", summary["status"])
if summary["status"] != "GO_HIDDEN_G0":
    raise SystemExit(
        "Preflight found too little final-controlled oscillation support. "
        "Do not spend the full hidden-state run yet; inspect the surface census "
        "or switch geometry."
    )
PY

NUM_EXAMPLES="$FULL_EXAMPLES" ./run_pilot_4gpu.sh
