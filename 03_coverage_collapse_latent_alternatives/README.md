# Coverage Collapse vs. Latent Viability of Suppressed Branches

## Status

**ARCHIVED / KILLED — `FIRST_FORK_BEHAVIORAL_PREMISE_NOT_PRESENT`**

Topic 03 has been permanently stopped under the registered falsification logic. The final seed-paper reproduction audit tested both plausible learning-rate interpretations (`1e-5` from the official code and `2e-5` from the paper text), restored e08, increased sampling to 64 generations/problem so `pass@32` could be measured, and added a teacher-forced first-decision probability audit.

Neither trajectory reproduced a meaningful late coverage collapse, and—more decisively—the first fork was already essentially perfectly viable at every checkpoint: native first-branch accuracy was `1.0`, `mean_p_true_viable_pair >= 0.999998`, and `wrong_commit_rate = 0`.

Therefore the latent question lacks its required experimental object. **G0-B latent probing is not authorized and should not be used to rescue this topic.**

## Final records

- [Archive summary](./ARCHIVE_SUMMARY.md) — complete scientific postmortem, failure reasons, and reusable lessons.
- [Final seed-paper reproduction audit](./results/REPRODUCTION_AUDIT.md) — exact protocol and kill decision.
- [Coverage table](./results/REPRODUCTION_AUDIT_COVERAGE.csv)
- [Teacher-forced first-fork table](./results/REPRODUCTION_AUDIT_TEACHER_FORCED.csv)
- [Initial G0-A results](./G0_RESULTS.md)
- [Literature / collision audit](./LITERATURE_AUDIT.md)
- [Validation contract](./VALIDATION.md)

## Original research question

The candidate asked:

> When SFT causes sampled reasoning coverage to shrink at a known binary fork, does a branch-viability signal learned at an earlier high-coverage checkpoint remain usable at the late checkpoint even when the native first-fork readout and latent readout disagree?

The intended contribution was deliberately narrower than generic representation collapse or hidden-state decodability. A positive result would have required all of the following: a genuine coverage-shrinking SFT trajectory, exact graph-ground-truth branch viability, a frozen early readout that transfers to the late checkpoint, matched target counterfactual validity, shortcut controls, and a label-free native-vs-latent disagreement test.

## Why it was killed before latent probing

The final audit established two failures.

### R1 — late coverage degradation failed

`pass@32` trajectories for e01/e02/e04/e08/e16 were:

```text
1e-5: 0.9842, 0.9832, 1.0000, 1.0000, 1.0000
2e-5: 0.7471, 0.9606, 0.9908, 0.9914, 0.9844
```

For `2e-5`, the apparent e08→e16 drop was only `0.0070` with paired 95% CI `[-0.006857, 0.024346]`; for `1e-5`, the late checkpoints were effectively saturated.

### R2 — wrong first-fork commitment failed even more strongly

Across both learning rates and all five checkpoints:

```text
output_choice_acc = 1.0
mean_p_true_viable_pair >= 0.999998
wrong_commit_rate = 0.0
strong_wrong_commit_rate = 0.0
```

The sampled first branch was likewise always viable. The remaining correctness variation therefore comes from downstream arithmetic / execution errors rather than suppression of the globally viable first branch.

That distinction is decisive. A high hidden-state probe score after this result would only show that graph/branch information is decodable; it would not establish that a behaviorally suppressed correct alternative remains latent.

## Reusable lesson

> **Mechanism research requires mechanism-level phenomenon replication.** Reproducing or approximately matching an aggregate metric is not enough. Before studying hidden representations of a proposed failure mode, verify that the exact decision point / state transition / failure event to be explained actually occurs at useful frequency and effect size.

The correct workflow is:

```text
1. reproduce aggregate phenomenon
2. localize the claimed mechanism-level event
3. verify that event is real and large enough
4. only then study representation / intervention / causality
```

Topic 03 failed at step 2 and is therefore archived.
