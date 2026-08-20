# Candidate Research Topics

This repository collects research-topic candidates in the **hypothesis / pilot** stage, together with archived negative results.

The current topic-selection rule is deliberately stricter than simple "paper + adjacent gap":

1. start from a **natural scientific question** that is interesting before mentioning LLMs, DLMs, probes, or checkpoints;
2. require a strong established phenomenon / empirical tension rather than betting that a new phenomenon exists;
3. use modern AI systems only when they provide a uniquely clean experimental axis;
4. check both recent literature collision and collision with our own archived/failed topics;
5. define the cheapest experiment that can invalidate the proposed interpretation;
6. freeze the primary measurement before broad search over models/layers/thresholds;
7. reserve confirmation data whenever discovery involves tuning a measurement;
8. stop early if the core claim fails; do not rescue it with post-hoc metric/model sweeps.

## Current candidates

| Priority | Topic | Natural question | Current gate |
| --- | --- | --- | --- |
| **Measurement repair** | [Confidence and Error Correction](./04_confidence_error_correction/) | If two learners are equally far from the correct answer, does being strongly committed to one wrong answer make corrective learning easier or harder? | **G-1v1 failed before training (61 pairs). One locked G-1v2 repair is allowed:** log-space balanced-permutation debiasing + independent permutation/prompt reliability. `<200` v2 pairs or failed reliability ⇒ archive. |
| **Active validation** | [Temporal Forgetting: Lost Skill or Lost Entry Point?](./05_temporal_forgetting_reentry/) | If a learner once solved a problem reliably but later fails, has the skill been lost or mainly become inaccessible? | MATH-500 × all 8 public Qwen2.5-7B RL checkpoints × repeated samples; require robust F/N/S groups before re-entry. |
| 3 | [Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | When a reasoning branch disappears from behavior, is branch-specific viability information erased or merely suppressed? | Before any training-dynamics study, the base/early model must reliably encode viability of unchosen branches at controlled graph forks. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Behavioral KL stabilized as expected, but matched residual representation movement stabilized at least as fast; no temporal decoupling survived robustness checks. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / FALSIFIED AS A BROAD CLAIM** | Exploratory signal did not survive preregistered independent GSM1K confirmation, while the positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |

## What the archived topics are for

Archived folders keep code, validation contracts, results, and failure analyses so the same weak premise is not repeatedly rediscovered.

A failed topic should not be revived by post-hoc metric/layer/model sweeps unless a genuinely new external observation changes the scientific premise.
