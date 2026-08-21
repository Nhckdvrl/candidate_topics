# Candidate Research Topics

This repository collects research-topic candidates in the **hypothesis / pilot** stage, together with archived negative results and identification failures.

A cross-topic record of failed candidates and reusable selection lessons is maintained in [`FAILURES_AND_LESSONS.md`](./FAILURES_AND_LESSONS.md). **Read it before registering a new topic.**

The current topic-selection rule is deliberately stricter than simple "paper + adjacent gap":

1. start from a **natural scientific question** that is interesting before mentioning LLMs, DLMs, probes, or checkpoints;
2. require a strong established phenomenon / empirical tension rather than betting that a new phenomenon exists;
3. use modern AI systems only when they provide a uniquely clean experimental axis;
4. check both recent literature collision and collision with our own archived/failed topics;
5. test **conceptual identifiability before scaling experiments**: the proposed observable/intervention must actually distinguish the competing explanations;
6. apply the **complexity-smell rule**: if making the result interpretable requires an expanding chain of gates, matching rules, baselines, and alternative-explanation exclusions, stop adding controls and reconsider whether the question/construct is natural and directly measurable;
7. prefer a **one-clean-contrast** first experiment whose interpretation is nearly forced by the question;
8. define the cheapest experiment that can invalidate the proposed interpretation;
9. freeze the primary measurement before broad search over models/layers/thresholds;
10. reserve confirmation data whenever discovery involves tuning a measurement;
11. stop early if the core claim or identification strategy fails; do not rescue it with post-hoc metric/model/control sweeps.

## Current candidates

| Priority | Topic | Natural question | Current gate |
| --- | --- | --- | --- |
| 1 | [When Does Uncontrollability Generalize?](./06_uncontrollability_generalization/) | With the same amount of uncontrollable experience, does spreading it across many unrelated contexts make passivity transfer more strongly to a novel controllable task than concentrating it in one context? | Run the yoked 2x2 behavioral pilot: first establish local controllability sensitivity, then cross-task transfer, then the frozen first-action diversity interaction. |
| 3 | [Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | When a reasoning branch disappears from behavior, is branch-specific viability information erased or merely suppressed? | Before any training-dynamics study, the base/early model must reliably encode viability of unchosen branches at controlled graph forks. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Behavioral KL stabilized as expected, but matched residual representation movement stabilized at least as fast; no temporal decoupling survived robustness checks. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / FALSIFIED AS A BROAD CLAIM** | Exploratory signal did not survive preregistered independent GSM1K confirmation, while the positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |
| [04 — Confidence and Error Correction](./04_confidence_error_correction/) | **ARCHIVED / KILLED AT MEASUREMENT / IDENTIFICATION GATE** | G-1v1 produced 61 clean matched pairs. One locked log-space measurement repair restored the commitment range but still yielded only 130 pairs, below the predeclared `<200` hard stop. No corrective SFT was run, so the correction hypothesis itself remains untested. | [Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md) |
| [05 — Temporal Forgetting: Lost Skill or Lost Entry Point?](./05_temporal_forgetting_reentry/) | **ARCHIVED / CONCEPTUAL IDENTIFICATION FAILURE** | Prefix-based rescue changes the task condition and cannot identify whether uncued old competence is still retained; `old route` is not a stable observable, and teacher-forced NLL is conditional on the same cue. The run stopped during partial sampling before scoring or any hypothesis-level gate, so no empirical conclusion is reported. | [Archive summary](./05_temporal_forgetting_reentry/ARCHIVE_SUMMARY.md) |

## What the archived topics are for

Archived folders keep code, validation contracts, partial results where appropriate, and failure analyses so the same weak premise is not repeatedly rediscovered.

The archive reason should be interpreted precisely:

- **Topic 01** failed its substantive G0 hypothesis.
- **Topic 02** failed locked independent confirmation of its proposed new claim.
- **Topic 04** failed before hypothesis testing because the required clean comparison could not be identified at sufficient scale.
- **Topic 05** was stopped even earlier at the conceptual-identification level: the proposed intervention could not distinguish retained competence from task simplification / conditional continuation, so continuing to sample would not resolve the scientific question.

The main methodological lesson from Topic 05 is important enough to make explicit:

> **Many controls do not rescue a non-identifying intervention. Before running a large pilot, verify that the proposed observation could in principle discriminate the latent explanations of interest.**

A second practical lesson is the complexity smell:

> **If the gate and kill line get longer every time the question is clarified, that is evidence to revisit the question itself. Good controls make a clear question rigorous; they should not be responsible for making an unclear construct exist.**

A failed topic should not be revived by post-hoc metric/layer/model/control sweeps unless a genuinely new external observation changes the scientific premise and motivates a newly registered question or identification strategy.