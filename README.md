# Candidate Research Topics

This repository collects research-topic candidates that are still in the **hypothesis / pilot** stage.

The default workflow for each topic is:

1. identify a strong seed paper or established phenomenon;
2. rotate only one adjacent variable;
3. define a cheap falsification experiment;
4. audit the measurement before spending compute;
5. stop early if the core premise fails;
6. only then expand into a full paper.

## Current candidates

| Priority | Topic | Core question | First falsification test |
| --- | --- | --- | --- |
| active | [Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | After global output-distribution movement enters a low late-training regime, does non-trivial feature-level learning continue? | First reproduce fixed-horizon behavioral KL on Pythia-410M; only then test middle-layer residual movement with matched cosine, standardized drift and CKA control. |
| 2 | [DLM Trajectory Fate](./02_dlm_trajectory_fate/) | Can hidden states distinguish recoverable/doomed and stable/overwritten states before the surface transition happens? | On 1,000 GSM8K examples, conditional hidden-state probes must beat entropy/confidence baselines with non-trivial lead time. |
| 3 | [Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | When a reasoning branch disappears from behavior, is branch-specific viability information erased or merely suppressed? | Before any training-dynamics study, the base/early model must reliably encode viability of unchosen branches at controlled graph forks. |

## Topic 01 implementation status

`01_behavior_vs_representation_stabilization` has been audited and rewritten as a staged falsification pipeline:

```text
G0-A: fixed-Δ behavior reproduction
  ↓ pass only
G0-B: cheap one-layer representation screen
  ↓ pass only
G1: sparse feature / crosscoder analysis
```

The implementation now avoids unequal checkpoint-gap confounding, uses a fixed byte-chunk corpus, reports raw + robust KL with cluster bootstrap confidence intervals, hooks one explicit GPT-NeoX `resid_pre` layer, and treats CKA as a control rather than the sole falsifier.

See:

- [Topic 01 README](./01_behavior_vs_representation_stabilization/README.md)
- [Topic 01 validation contract](./01_behavior_vs_representation_stabilization/VALIDATION.md)
