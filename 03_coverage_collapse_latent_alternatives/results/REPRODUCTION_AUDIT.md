# Final Seed-Paper Reproduction Audit

## Decision

**`ARCHIVED — FIRST_FORK_BEHAVIORAL_PREMISE_NOT_PRESENT`**

The requested final audit was completed on both learning-rate interpretations.
Topic 03 is archived and G0-B latent probing was not run.

The decisive result is not merely a failed coverage gate: the first fork is
already perfectly viable in this task implementation. Across both learning
rates, all 200 problems at all five checkpoints had `mean_p_viable_first =
1.0`, sampled first-branch entropy `0`, teacher-forced native-choice accuracy
`1.0`, and `wrong_commit_rate = 0`.

## Protocol

- Qwen2.5-0.5B, 200 held-out problems, e01/e02/e04/e08/e16.
- 64 samples/problem, temperature 1.0, top-p 0.95, max tokens 512.
- `1e-5`: exact Qwen2.5-0.5B value in the pinned upstream `run_sft.sh`.
- `2e-5`: exact value reported in the seed-paper appendix.
- Metrics: pass@1, pass@2, pass@8, pass@32.
- Teacher-forced branch audit: token probabilities at the first decision point,
  rather than sampled branch entropy.
- Paired bootstrap: 100,000 resamples, seed 42, pairing by problem ID.

The full tracked tables are [REPRODUCTION_AUDIT_COVERAGE.csv](REPRODUCTION_AUDIT_COVERAGE.csv)
and [REPRODUCTION_AUDIT_TEACHER_FORCED.csv](REPRODUCTION_AUDIT_TEACHER_FORCED.csv).

## R1 — late coverage degradation

R1 fails for both learning rates.

| LR | pass@32 trajectory (e01, e02, e04, e08, e16) | e08 - e16 | paired 95% CI |
|---|---|---:|---:|
| 1e-5 | 0.9842, 0.9832, 1.0000, 1.0000, 1.0000 | ~0.0000 | [-0.000000001, 0.000000158] |
| 2e-5 | 0.7471, 0.9606, 0.9908, 0.9914, 0.9844 | 0.0070 | [-0.006857, 0.024346] |

The only clear changes are early-to-late improvement or saturation. No early
checkpoint has a practically meaningful, statistically supported pass@32 drop
relative to e16. For 2e-5, the apparent e08 peak is not supported by the
paired interval. For 1e-5, e04/e08/e16 are numerically indistinguishable at
pass@32.

## R2 — fork mechanism

R2 fails. The teacher-forced audit shows increasing confidence polarization in
the correct direction, but never the required mixture of confident correct and
confident incorrect choices:

- all checkpoints and both learning rates: `output_choice_acc = 1.0`;
- all checkpoints and both learning rates: `mean_p_true_viable_pair` is at
  least `0.999998`;
- all checkpoints and both learning rates: `wrong_commit_rate = 0.0` and
  `strong_wrong_commit_rate = 0.0`;
- mean absolute margin increases toward late training, while pair entropy
  collapses toward zero.

Thus confidence becomes more polarized, but there is no incorrect branch
population to support the Figure 4 mechanism.

## Interpretation

The sampled correctness variation is downstream arithmetic/execution noise,
not suppression of an alternative globally viable first branch. The latent
probe question therefore lacks its required experimental object in this
server-side reproduction. This is a kill of the Topic 03 behavioral premise,
not evidence that hidden states can never encode alternative reasoning.

## Runs and code correction

- `1e-5` coverage run: `audit_lr1e-5_64_20260822_064701`
- `2e-5` coverage run: `audit_lr2e-5_64_20260822_060143`
- teacher-forced summaries: `artifacts/reproduction_audit/*_teacher_forced.csv`
- raw upstream run directories are local because `external/reasoning_forks` is
  a separate ignored checkout.

The teacher-forced extractor now runs its model forward pass under
`torch.inference_mode()`, preventing hidden states that still require
gradients from being converted to NumPy arrays.
