# DLM Trajectory Fate

**Question:** before a visible denoising transition happens, does a DLM hidden state contain information about whether the current answer will recover or be overwritten?

## Why this is a candidate topic

Two 2026 results leave a narrow adjacent gap:

1. **Time Is a Feature** / `dLLM-MidTruth` shows that complete intermediate `x0` predictions can oscillate during denoising: an answer can become correct and later become wrong again.
2. **Probing Functional Correctness in Diffusion Language Models** / `dlm-probing` shows that DLM hidden states increasingly predict **final** functional correctness.

The tempting next question is to replace the target `final correctness` with `future fate of the current state`.

However, a naive version is confounded: if we label a currently-wrong state as “recoverable” whenever it later becomes correct, the probe can succeed simply by reading the already-known **final-correctness** signal. This repository therefore treats generic recover/overwrite probes as controls and makes the primary novelty test **final-outcome controlled**.

## Primary scientific test

At a fixed denoising step, condition on current surface correctness **and final outcome**.

### Transient recovery

Among trajectories that are **wrong now and wrong at the end**:

- positive: they become observably correct at least once later (`wrong -> correct -> wrong`);
- negative: they never become observably correct later.

Can the current hidden state predict that future transient recovery?

### Transient overwrite

Among trajectories that are **correct now and correct at the end**:

- positive: they become observably wrong at least once later (`correct -> wrong -> correct`);
- negative: they remain observably correct.

Can the current hidden state predict that future transient overwrite?

Because the final outcome is identical inside each comparison, success cannot be explained by merely reproducing the known final-correctness probe.

## Important implementation choices

- **Surface state = complete `x0` before token transfer.** This matches `dLLM-MidTruth` temporal voting. Decoding the partially committed `x` would measure a different process.
- **No-answer-yet is not wrong.** Strict parsing stores an `observed` mask; a step without the requested `####` / `\\boxed{}` answer field is unavailable rather than incorrect.
- **Deterministic denoising is the primary G0.** `temperature=0` removes future Gumbel randomness that is invisible to the current hidden state. The stochastic `dlm-probing` geometry is retained as a reference/robustness run.
- **Same-step comparisons.** Pre-transition positives and negatives are compared at the same absolute denoising step, preventing a probe from exploiting the fact that diffusion time itself is encoded.
- **Three controls per probe.** Current hidden state is compared with (i) observable uncertainty/progress features and (ii) the same layer at step 0, which controls static problem difficulty.
- **Problem-level independence.** Fixed-step analyses contain one row per problem; no `(problem, step)` leakage is allowed.

## Fast falsification pipeline

The validation is deliberately staged so we do not spend an 8B hidden-state run before knowing that the required classes exist.

### G-1: surface census (200 GSM8K problems)

```bash
cd 02_dlm_trajectory_fate
NUM_EXAMPLES=200 ./run_surface_preflight_4gpu.sh
```

Primary geometry, chosen for speed and determinism:

```text
model           GSAI-ML/LLaDA-8B-Instruct
prompt          dLLM-MidTruth GSM8K format
steps           64
generation      128 tokens
block length    32
temperature     0
GPUs            4 independent shards
```

This run stores no hidden states. It asks only whether strict surface trajectories contain enough `transient_recovery` / `transient_overwrite` examples to estimate a probe. Default gate: at least 10 examples in each class at some saved step in the 200-problem preflight.

If the gate fails, **do not run the expensive hidden-state G0**. Inspect `surface_class_counts.csv` and either stop or switch geometry.

### G0: hidden-state pilot (1000 problems)

```bash
NUM_EXAMPLES=1000 ./run_pilot_4gpu.sh
```

For selected steps and upper hidden-state tuple indices `24,25,28`, use the same basic probe family as the reference work:

```text
mean pooled hidden state
-> StandardScaler
-> PCA(max 64)
-> LogisticRegression(C=1, lbfgs)
-> out-of-fold AUC
```

Surface baseline:

- mean masked-token entropy;
- probability of the sampled/selected token;
- clean maximum-token probability;
- fraction unmasked;
- prompt token length;
- current answer availability/correctness.

Initial-state control: the same hidden layer at denoising step 0 on the exact same problem subset.

Uncertainty is reported with paired bootstrap confidence intervals over out-of-fold predictions.

### G0 control: reproduce the seed measurement

Before interpreting a negative novelty result, the chosen geometry must reproduce the established **final-correctness** hidden signal. We require a later-step final-correctness probe with approximately:

```text
AUC >= 0.65
and AUC - step0-hidden AUC >= 0.03
```

If this does not happen in the fast MidTruth geometry, the result is **not** counted as evidence against the topic. Run:

```bash
NUM_EXAMPLES=1000 ./run_reference_geometry_4gpu.sh
```

which uses the public `dlm-probing`-style GSM8K geometry (`128` steps, `512` generated tokens, block `32`, temperature `0.2`, probing prompt).

### One-command staged run

```bash
./run_fast_validation_4gpu.sh
```

It runs the 200-example surface gate first and exits before the 1000-example hidden run when class support is inadequate.

## Decision rule

The novelty claim is based on `transient_recovery` and `transient_overwrite`, not the easier generic recover/overwrite tasks.

Continue only if:

1. there is adequate final-controlled class support;
2. the final-correctness reference probe is reproduced;
3. at least one novel task is predictable **before** the visible transition (default minimum lead >= 4 steps);
4. current hidden-state AUC is materially above both the surface baseline and step-0 hidden baseline.

Default strong-row gate:

```text
AUC >= 0.65
95% bootstrap lower bound > 0.55
AUC - surface baseline >= 0.03
AUC - step0 hidden baseline >= 0.03
lead >= 4 denoising steps
```

Stop if the signal exists only in generic recover/overwrite labels, only near/after the visible transition, disappears under final-outcome control, or is explained by surface uncertainty/static difficulty.

## Outputs

Surface preflight:

```text
artifacts/preflight_midtruth/surface_class_counts.csv
artifacts/preflight_midtruth/surface_summary.json
```

Full G0:

```text
artifacts/g0_midtruth/probes/task_class_counts.csv
artifacts/g0_midtruth/probes/step_layer_auc.csv
artifacts/g0_midtruth/probes/pretransition_auc.csv
artifacts/g0_midtruth/probes/decision.json
artifacts/g0_midtruth/probes/fate_labels.npz
```

## Reference code

- Time Is a Feature / dLLM-MidTruth: https://github.com/aim-uofa/dLLM-MidTruth
- Probing Functional Correctness in Diffusion Language Models / dlm-probing: https://github.com/guan404ming/dlm-probing

See [`VALIDATION.md`](VALIDATION.md) for the audit, exact alignment with the reference implementations, known confounds, and validation gates.
