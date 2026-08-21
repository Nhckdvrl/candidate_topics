# Server handoff

## Goal

Do not tune for a positive result. Run the frozen G0 as written and decide whether the generative policy's sampled action diversity has task-relative structure that scalar ACE loses.

## Environment

```bash
cd candidate_topics/08_generative_policy_task_geometry
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

A single GPU is enough for one seed. The experiment is low-dimensional; use separate GPUs/nodes for independent seeds rather than distributed training.

## Recommended first run

Run three independent seeds in parallel:

```bash
GPU=0 SEED=0 ./run_g0.sh
GPU=1 SEED=1 ./run_g0.sh
GPU=2 SEED=2 ./run_g0.sh
```

Each command trains two policies:

```text
main             null_gain=1.0
no_null_control  null_gain=0.0
```

Defaults per condition:

```text
300 base tasks
4 hidden posture modes per base task
8-step action chunks
30k optimizer steps
256 evaluation states
128 sampled chunks/state
200 closed-loop rollout episodes
```

For a cheap runtime sanity check only:

```bash
GPU=0 SEED=99 BASE_TASKS=30 MODES=2 TRAIN_STEPS=300 STATES=24 SAMPLES=16 ROLLOUTS=10 ./run_g0.sh
```

Do not interpret the smoke run scientifically.

## Expected outputs

For seed 0:

```text
results/g0_seed0/main/
  train_dataset.npz
  ace_calibration_ranges.npy
  checkpoint_*.pt
  train_meta.json
  train_log.json
  state_metrics_checkpoint_*.csv
  eval_checkpoint_*.json
  G0_GATE.json

results/g0_seed0/no_null_control/
  ... same layout ...
```

The file to read first is:

```text
results/g0_seed*/main/G0_GATE.json
```

Key fields:

```text
rollout_success
median_null_task_ratio
spearman_ace_risk
spearman_taskvar_risk
matched_entropy.n_pairs
matched_entropy.mean_abs_ace_z_gap
matched_entropy.risk_diff_high_minus_low
matched_risk_diff_ci95
verdict
failed_clauses
```

## Decision rule

Treat a seed as G0-positive only if its `verdict` is `GO_TO_FRANKA`.

For the research direction to survive the pilot, require at least 2/3 seeds to be positive and inspect the no-null controls. The main condition should show a visibly stronger null-space structure than the control; if main and control are essentially indistinguishable, interpret the effect as diffusion sampling/model error rather than learned functional redundancy.

Do not rescue a failed seed by changing ACE alpha, task/null rank tolerance, risk threshold, perturbation scale, or matching tolerance. Those are fixed in `VALIDATION.md` for this pilot.

## What to inspect if training fails technically

Technical debugging is allowed when the policy does not learn at all. Check:

```text
train_meta.json -> demonstration success rate should be high
train_log.json  -> diffusion loss should decrease
rollout_success -> final policy should exceed 0.80 before geometry is interpreted
```

If expert success is low, that is an implementation bug in the analytic controller/data generator. If expert success is high but all three learned policies remain below 0.80, report that as a model/training failure of this implementation; do not reinterpret geometry from a broken policy.

## Static checks already run

```text
PYTHONPATH=. pytest -q
# 8 passed

bash -n run_g0.sh
```

A full CUDA training run was not executed in the authoring environment. A short CPU smoke run successfully completed dataset generation, training, checkpoint loading, DDPM sampling, rollout evaluation, CSV emission, and G0 analysis; as expected, the deliberately undertrained smoke policy did not meet the scientific success gate.
