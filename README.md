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
| **Active validation** | [Temporal Forgetting: Lost Skill or Lost Entry Point?](./05_temporal_forgetting_reentry/) | If a learner once solved a problem reliably but later fails, has the skill been lost or mainly become inaccessible? | MATH-500 × all 8 public Qwen2.5-7B RL checkpoints × repeated samples; require robust F/N/S groups before re-entry. |
| 3 | [Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | When a reasoning branch disappears from behavior, is branch-specific viability information erased or merely suppressed? | Before any training-dynamics study, the base/early model must reliably encode viability of unchosen branches at controlled graph forks. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Behavioral KL stabilized as expected, but matched residual representation movement stabilized at least as fast; no temporal decoupling survived robustness checks. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / FALSIFIED AS A BROAD CLAIM** | Exploratory signal did not survive preregistered independent GSM1K confirmation, while the positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |
| [04 — Confidence and Error Correction](./04_confidence_error_correction/) | **ARCHIVED / KILLED AT MEASUREMENT / IDENTIFICATION GATE** | G-1v1 produced 61 clean matched pairs. One locked log-space measurement repair restored the commitment range but still yielded only 130 pairs, below the predeclared `<200` hard stop. No corrective SFT was run, so the correction hypothesis itself remains untested. | [Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md) |

## What the archived topics are for

Archived folders keep code, validation contracts, results, and failure analyses so the same weak premise is not repeatedly rediscovered.

The archive reason should be interpreted precisely:

- Topic 01 failed its substantive G0 hypothesis.
- Topic 02 failed locked independent confirmation of its proposed new claim.
- Topic 04 failed before hypothesis testing because the required clean comparison could not be identified at sufficient scale.

A failed topic should not be revived by post-hoc metric/layer/model sweeps unless a genuinely new external observation changes the scientific premise and motivates a newly registered question or identification strategy.
