# Candidate Research Topics

This repository collects research-topic candidates in the **hypothesis / pilot** stage, together with archived negative results.

The default workflow for each topic is:

1. identify a strong seed paper or established phenomenon;
2. rotate only one adjacent variable;
3. define a cheap falsification experiment;
4. verify the measurement before spending compute;
5. stop early if the core premise fails;
6. only then expand into a full paper.

## Current candidates

| Priority | Topic | Core question | First falsification test |
| --- | --- | --- | --- |
| 1 | [Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | When a reasoning branch disappears from behavior, is branch-specific viability information erased or merely suppressed? | Before any training-dynamics study, the base/early model must reliably encode viability of unchosen branches at controlled graph forks. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Behavioral KL stabilized as expected, but matched residual representation movement stabilized at least as fast; no temporal decoupling survived robustness checks. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / FALSIFIED AS A BROAD CLAIM** | Exploratory signal did not survive preregistered independent GSM1K confirmation, while the positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |

## What the archived topics are for

Archived folders are intentionally kept with their code, validation contracts, results, and failure analyses. They serve two purposes:

1. preserve reusable experimental infrastructure;
2. record why a natural-looking research question failed, so the same weak premise is not repeatedly rediscovered.

A failed topic should not be revived by post-hoc metric/layer/model sweeps unless a genuinely new external observation changes the scientific premise.
