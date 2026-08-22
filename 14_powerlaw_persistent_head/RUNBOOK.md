# Topic 14 server runbook

This runbook is intentionally conservative: audit first, use cheap gates, and only spend near-paper compute after the seed prerequisite is established.

## 0. Enter the folder

```bash
cd candidate_topics/14_powerlaw_persistent_head
```

## 1. Reuse the existing environment

Do **not** create a new environment if the node already has working PyTorch + NumPy.

Check:

```bash
python - <<'PY'
import numpy, torch
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

Only if imports are missing:

```bash
pip install -r requirements.txt
```

`matplotlib` is optional.

## 2. Mandatory zero-GPU audit

```bash
python self_test.py
python audit_schedule.py --cycles 2 | tee schedule_audit.txt
```

Expected structural pattern:

- `SELF_TEST_OK`;
- `occupancy_is_exactly_balanced = true` for slow and fast;
- realized per-skill counts all equal;
- slow lag-1 log-frequency correlation much larger than fast;
- slow head runs much longer than fast.

Do not proceed if this fails.

## 3. Smoke test

Four GPUs:

```bash
bash run_gate.sh smoke 0
```

One GPU:

```bash
GPUS=0 bash run_gate.sh smoke 0
```

The report must say:

```text
SMOKE_ONLY_DO_NOT_INTERPRET
```

Smoke verifies execution only.

## 4. Primary cheap pilot

Recommended on a 4-GPU node:

```bash
bash run_gate.sh pilot 0
```

This runs:

- GPU 0: uniform
- GPU 1: static power law
- GPU 2: balanced slow
- GPU 3: balanced fast

Each job is an independent process; there is no cross-GPU synchronization or cross-node traffic.

Outputs:

```text
outputs/pilot/seed0/<condition>/config.json
outputs/pilot/seed0/<condition>/schedule_audit.json
outputs/pilot/seed0/<condition>/metrics.csv
outputs/pilot/seed0/<condition>/stdout.log
outputs/pilot/decision.json
outputs/pilot/learning_curves.png   # only if matplotlib exists
```

## 5. If the pilot static-vs-uniform anchor is weak

Do not interpret slow-vs-fast yet. Check only the published prerequisite at near-paper budget:

```bash
CONDITIONS=uniform,static bash run_gate.sh full 0
```

This uses only two conditions. On four available GPUs, only two are occupied.

If the static power-law anchor still fails, stop the topic rather than tuning the intervention.

## 6. If the anchor is healthy

Confirm with paired model seeds:

```bash
bash run_gate.sh confirm 0,1,2
```

Read:

```bash
cat outputs/confirm/decision.json
```

The most important fields are:

```text
means.anchor_auc_gain_static_minus_uniform
means.persistence_auc_slow_minus_fast
means.local_asym_auc_slow_minus_uniform
means.local_asym_auc_fast_minus_uniform
decision
```

Always inspect the raw `metrics.csv` / learning curves in addition to the automatic gate.

## 7. Full budget only if justified

```bash
bash run_gate.sh full 0,1,2
```

`full` uses 199,920 optimizer steps per condition, nearly matching the seed paper's 200k S5 setting while preserving complete balanced cycles.

## Optional: choose GPUs / precision

```bash
GPUS=0,3 PRECISION=bf16 bash run_gate.sh pilot 0
```

With two GPUs, conditions run in two waves. `fp16` is the default for seed fidelity; `bf16` is available when the hardware/environment makes it preferable, but do not mix precision inside one paired comparison.

## Optional: run conditions manually

```bash
python experiment.py \
  --condition balanced_slow \
  --profile pilot \
  --seed 0 \
  --output outputs
```

Useful for debugging, but the locked four-condition launcher is preferred for scientific runs.

## Interpretation discipline

A strong slow-vs-fast difference establishes that temporal ordering/persistence is a real variable under the seed-like optimizer schedule.

Do not immediately overclaim “head scaffolding.” If the effect replicates, the one allowed mechanism diagnostic is a constant-LR rerun to test whether the separation is actually an interaction with LR decay:

```bash
LR_SCHEDULE=constant WARMUP_STEPS=0 bash run_gate.sh confirm 0,1,2
```

Do not use this diagnostic to rescue a null.
