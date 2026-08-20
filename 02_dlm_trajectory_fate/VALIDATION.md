# Audit and G0 validation protocol

This document records what was kept, what was changed, and why. The purpose is not to build a broad experimental framework; it is to get a **fast, interpretable yes/no answer** about whether the topic has a real signal beyond existing work.

## 1. Reference-code audit

### dLLM-MidTruth (`aim-uofa/dLLM-MidTruth`)

The public `eval/generate.py` performs each denoising iteration as follows:

1. forward current partially committed sequence `x`;
2. form a complete prediction `x0` for all still-masked positions;
3. when temporal voting is enabled, decode/parse that complete `x0`;
4. transfer only a confidence-ranked subset of `x0` tokens into persistent `x`.

Therefore **complete `x0` before transfer** is the correct surface object for temporal oscillation. The existing candidate implementation had this right and it is preserved.

The public GSM8K evaluation uses a deterministic configuration (`temperature=0`) and commonly 128 generated tokens / 64 diffusion steps. That is now the primary fast G0 geometry.

### dlm-probing (`guan404ming/dlm-probing`)

The public `src/core/modal_midstep_probe.py` uses:

```text
LLaDA-8B-Instruct
GSM8K test
steps = 128
generation length = 512
block length = 32
temperature = 0.2
saved steps = {0,1,4,16,32,64,127}
4 generation-position regions
StandardScaler -> PCA(64) -> LogisticRegression
5-fold stratified CV
```

Important details verified from code and preserved where relevant:

- additive float64 Gumbel sampling for LLaDA when temperature > 0;
- per-instance `torch.manual_seed(0)` reset;
- hidden states are captured before token transfer;
- the global layer-wise probe averages the four region means, so storing one global generation mean is equivalent for this G0; four regions are unnecessary unless we later do region-specific analysis.

The exact reference geometry is available in `run_reference_geometry_4gpu.sh` and is used as a validation fallback, not the first expensive run.

## 2. Problems found in the previous validation

### P0 — the novelty target was confounded with final correctness

Previous labels were:

```text
current wrong   -> recoverable if any later step is correct
current correct -> overwrite if any later step is wrong
```

A high recoverability probe can therefore be driven by the already-known `final correctness` representation. That would not establish a new “trajectory fate” variable.

**Fix:** final-outcome-controlled primary tasks:

```text
transient_recovery:
  current wrong, final wrong
  any later observed correct vs never later observed correct

transient_overwrite:
  current correct, final correct
  any later observed wrong vs never later observed wrong
```

Generic `recover_any` / `overwrite_any` remain secondary descriptive controls. Conditional final-outcome probes are also reported explicitly so we can see when a result is merely the known signal.

### P0 — “no answer yet” was silently treated as an incorrect answer

With strict intermediate parsing, early `x0` predictions frequently do not yet contain the requested answer marker. The old boolean correctness array mapped `no parseable answer` to `False`, which can fabricate transitions.

**Fix:** store two arrays per parser:

```text
observed[t] = is a parseable requested answer present?
correct[t]  = if observed, does it equal gold?
```

Fate labels ignore unobserved steps. A missing `####` / `\\boxed{}` is not evidence of being wrong.

### P0 — a stochastic primary run makes a negative result ambiguous

At `temperature=0.2`, future Gumbel draws are not available in the current hidden state. “Cannot predict future fate” could therefore mean either no representation exists or the future transition was determined by later random noise.

**Fix:** primary G0 uses `temperature=0`. The stochastic public probing geometry is kept as reference/robustness, not as the main falsification test.

### P1 — the first run was too expensive for a feasibility screen

The old launcher immediately ran 1000 problems with 128 denoising steps, 512 generated tokens, and hidden-state capture.

**Fix:** two-stage gate:

1. `200` problems, `64x128`, deterministic, **surface only**;
2. only if transient classes exist, `1000` problems with sparse hidden capture.

This can kill a low-support topic before paying the 8B hidden-state cost.

### P1 — pooled lead-time analysis could leak diffusion time

DLM hidden states encode denoising time. Pooling positives and negatives from different absolute steps can produce a superficially good fate classifier that is partly a time classifier.

**Fix:** pre-transition analyses compare positives and negatives at the **same absolute saved step**. Positive examples closer than the requested lead threshold are removed rather than pooled into another time bin.

### P1 — surface/static-difficulty controls were incomplete

A fate probe can succeed because some questions are simply easier, or because current confidence already predicts the transition.

**Fix:** every hidden probe is compared on the same CV splits against:

- current surface uncertainty/progress features;
- hidden state from step 0 for the exact same examples.

A novel result must beat both.

### P1 — no positive-control requirement

If the fast geometry fails to expose the final-correctness signal reported by the seed probing paper, a negative fate result is uninterpretable.

**Fix:** G0 first checks that later hidden states reproduce final correctness above step-0 difficulty. Failure returns:

```text
GEOMETRY_NOT_VALIDATED_RUN_REFERENCE_GEOMETRY
```

rather than “topic dead”.

### P2 — avoidable probability-tensor overhead

The old implementation materialized full float64 softmax probabilities to obtain confidence for every denoising step.

**Fix:** token-transfer confidence is computed exactly from:

```text
exp(chosen_logit - logsumexp(logits))
```

without materializing the full probability tensor. Entropy / clean-max / selected-probability baselines are computed only at saved steps and in small position chunks.

### P2 — stale shard mixing

A wildcard load could silently combine output from incompatible prior runs.

**Fix:** each shard stores run metadata; loader checks identical geometry, exact shard count, unique problem IDs, expected example count, and presence/absence of hidden states. Launchers use isolated run directories.

## 3. What remains intentionally simple

We do **not** add an MLP probe, nonlinear representation learner, causal steering, multiple models, or multiple tasks to G0. A linear probe is sufficient to answer the first feasibility question and stays closest to the seed measurement.

We also do not claim an intrinsic deterministic property of a prompt. Labels describe the fate of a **specific deterministic denoising trajectory** under the specified generation geometry.

## 4. Primary run order

### Engineering smoke

```bash
NUM_EXAMPLES=20 ./run_surface_preflight_4gpu.sh
```

Check that all shards finish and strict answers are being parsed.

### G-1 class-support census

```bash
NUM_EXAMPLES=200 ./run_surface_preflight_4gpu.sh
```

Gate:

```text
max min(class0, class1) for a final-controlled transient task >= 10
```

If not, stop before hidden extraction. Inspect whether the cause is:

- strict answer markers rarely appear;
- final accuracy is degenerate;
- oscillations are genuinely too rare in this deterministic geometry.

Fallback parser exists only as a diagnostic. A phenomenon that appears only with “last number anywhere in the reasoning text” is not a clean primary result.

### G0 hidden-state test

```bash
NUM_EXAMPLES=1000 ./run_pilot_4gpu.sh
```

Saved steps:

```text
0,1,2,4,8,16,24,32,40,48,56,60,62,63
```

Upper hidden-state tuple indices:

```text
24,25,28
```

The point is not to map every layer. Three upper locations are enough to decide whether a strong signal exists.

## 5. Probe statistics

For every task, step, and selected layer:

1. select valid problems at that exact step;
2. require a minimum class count (default 30 for the 1000-problem run);
3. generate identical stratified CV splits;
4. fit current-hidden, step0-hidden, and surface-baseline models on the same splits;
5. collect out-of-fold predictions;
6. compute AUC and paired bootstrap 95% confidence intervals.

This avoids comparing means from unrelated CV folds and lets us bootstrap differences directly.

### Main novelty rows

Only these can support the paper premise:

```text
transient_recovery
transient_overwrite
```

and, for the strongest claim, only rows where the visible transition is at least several denoising steps in the future.

### Secondary/control rows

```text
final_correct_replication
recover_any
overwrite_any
finish_correct_from_wrong
finish_wrong_from_correct
```

A strong result only on these controls is not enough.

## 6. Automated decision states

`decision.json` returns one of:

```text
STOP_LOW_NOVEL_CLASS_SUPPORT
GEOMETRY_NOT_VALIDATED_RUN_REFERENCE_GEOMETRY
STOP_NO_NOVEL_PRETRANSITION_SIGNAL
CONTINUE
```

Default `CONTINUE` requires at least one final-controlled transient row with:

```text
lead >= 4
AUC >= 0.65
AUC lower 95% bootstrap bound > 0.55
delta vs surface >= 0.03
delta vs step0 hidden >= 0.03
```

These thresholds are a project triage gate, not a publication-level hypothesis test. If the effect is only marginal at G0, the right response is to stop rather than optimize the probe until it looks positive.

## 7. Validation completed before PR

The repository-side implementation is validated without pretending that an 8B GPU experiment was run locally:

- Python bytecode/syntax compilation;
- shell `bash -n` checks;
- answer-parser tests;
- observed-vs-unobserved fate-label tests;
- final-controlled transient label tests;
- transfer-schedule and selected-probability equivalence tests;
- shard metadata / stale-run protection tests;
- synthetic transient-signal probe test;
- synthetic end-to-end `train_probes.py` and `summarize_surface.py` CLI runs.

The actual LLaDA generation remains the server-side experiment. If its positive-control final-correctness probe does not reproduce, the pipeline explicitly refuses to interpret the novelty result and directs us to the reference geometry.
