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
8. define the earliest experiment that can invalidate the proposed interpretation, but **estimate its real wall-clock / rollout / engineering cost before calling it cheap**;
9. freeze the primary measurement before broad search over models/layers/thresholds;
10. reserve confirmation data whenever discovery involves tuning a measurement;
11. stop early if the core claim or identification strategy fails; do not rescue it with post-hoc metric/model/control sweeps;
12. for mechanism work, require **mechanism-level phenomenon replication** before hidden-state analysis: the exact decision point / failure event being explained must actually occur in the chosen system, not merely an aggregate metric that could be produced by another failure mode;
13. if identification relies on **natural disagreement / crossover / reversal / transition**, require evidence that this variation exists at useful **instance-level density**; aggregate score differences are not enough;
14. for stochastic per-item gates, calibrate the gate against an explicit null rather than assuming a large-looking threshold is noise-proof.

## Current candidates

| Priority | Topic | Natural question | Current gate |
| --- | --- | --- | --- |
| 1 | [10 — Is DLM Generation Order Invariant to Problem Isomorphisms?](./10_dlm_generation_order_invariance/) | When a problem is changed only by an exact symmetry, does a DLM preserve the order in which it solves the problem? | Run one paired Sudoku-isomorphism test and measure mapped-cell finalization-rank invariance. |
| 2 | [11 — What Does Diffusion Confidence Actually Know?](./11_dlm_confidence_internal_consistency/) | Does native DLM confidence track trajectory-internal consistency independently of whether the final answer is externally correct? | Run one programmatic `internal consistency × external correctness` factorial contrast; kill if construction requires judging or hand-labeling. |
| 3 | [12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?](./12_reasoning_necessity_vs_rl_leverage/) | Are the layers most necessary for reasoning also the layers where isolated RL updates can improve reasoning most effectively? | Compare the complete frozen layer-ablation curve with the complete single-layer-RL leverage curve on a matched model/task. |
| 4 | [13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?](./13_repetition_temporal_spacing/) | With exactly the same repeated training multiset, does temporal spacing between identical exposures change pretraining damage? | Reproduce one strong repetition regime, then compare clustered/random/even spacing while fixing unique positions and repeated slots. |
| 5 | [14 — Does Power-Law Learning Need a Persistent Head?](./14_powerlaw_persistent_head/) | Does a power-law curriculum work because some skills remain frequent long enough to scaffold others, or is momentary frequency asymmetry sufficient? | Reproduce static power-law > uniform, then compare count-matched slow- vs fast-rotation schedules with the same per-block spectrum. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [09 — Does a VLA Know Its Own Limits?](./09_vla_own_limits/) | **ARCHIVED / KILLED AT G0 IDENTIFICATION SUPPORT** | The frozen 3,600-rollout LIBERO-10 panel found insufficient natural bidirectional same-state competence crossover. 2k never robustly beat 3k or 9k on any of 150 states; 3k vs 9k produced only 3 robust wins in each direction against a predeclared requirement of 15. The 3+3 reversals survived a within-state sampling-noise null (`p=0.001`), so the issue was not that all crossover was noise: it was genuinely too sparse to support the paired self-knowledge test. G1 was never run. | [Archive summary](./09_vla_own_limits/ARCHIVE_SUMMARY.md) |
| [08 — Does Action Diversity Track Functional Uncertainty?](./08_generative_policy_task_geometry/) | **ARCHIVED / KILLED AT THE OPERATIONAL BAR** | The original planar-arm design was killed before running: its "functional risk" outcome was the first-order linearisation of the same Jacobian used to build its task-sensitive projector, so its key gate held for any action distribution including isotropic noise. The stripped-down PushT test confirmed action/outcome decoupling, but that was unsurprising for contact dynamics; the stronger operational claim that scalar action entropy is effectively broken did not hold. | [Archive summary](./08_generative_policy_task_geometry/ARCHIVE_SUMMARY.md) |
| [07 — Old Blocks New, or New Erases Old?](./07_memory_interference_architecture/) | **ARCHIVED / INCONCLUSIVE AT FROZEN DISCOVERY GATE** | The Transformer reproduced positive mean PI>RI asymmetry, but the preregistered Transformer–GatedDeltaNet gap was only `ΔI=0.0729` with paired bootstrap 95% CI `[-0.0313, 0.1771]`, below the `0.10` GO threshold, with `0/4` sign-transition levels. The seed phenomenon was real, but the proposed architecture explanatory axis was not strong or clean enough to justify confirmation or post-hoc expansion. | [Archive summary](./07_memory_interference_architecture/ARCHIVE_SUMMARY.md) |
| [06 — When Does Helplessness Become a Worldview?](./06_helplessness_worldview/) | **ARCHIVED / KILLED AT ACQUISITION PREMISE** | Qwen3-8B showed almost no controllability acquisition or transfer. One preregistered Qwen3-32B v2 removed the action ceiling and still produced only +2–5pp late-training C/U separation, with no predicted diversity amplification (`D=-4.17pp`). The higher-order natural question remains unresolved because the chosen LLM agent never robustly instantiated the prerequisite learned-uncontrollability state. | [Archive summary](./06_helplessness_worldview/ARCHIVE_SUMMARY.md) |
| [05 — Temporal Forgetting: Lost Skill or Lost Entry Point?](./05_temporal_forgetting_reentry/) | **ARCHIVED / CONCEPTUAL IDENTIFICATION FAILURE** | Prefix-based rescue changes the task condition and cannot identify whether uncued old competence is still retained; `old route` is not a stable observable, and teacher-forced NLL is conditional on the same cue. The run stopped during partial sampling before scoring or any hypothesis-level gate, so no empirical conclusion is reported. | [Archive summary](./05_temporal_forgetting_reentry/ARCHIVE_SUMMARY.md) |
| [04 — Confidence and Error Correction](./04_confidence_error_correction/) | **ARCHIVED / KILLED AT MEASUREMENT / IDENTIFICATION GATE** | G-1v1 produced 61 clean matched pairs. One locked log-space measurement repair restored the commitment range but still yielded only 130 pairs, below the predeclared `<200` hard stop. No corrective SFT was run, so the correction hypothesis itself remains untested. | [Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md) |
| [03 — Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | **ARCHIVED / KILLED AT BEHAVIORAL PREMISE** | Final seed-paper reproduction audit tested both official-code `1e-5` and paper-text `2e-5`, e01/e02/e04/e08/e16, 64 samples/problem, `pass@32`, and teacher-forced first-fork probabilities. No meaningful late coverage degradation reproduced, and every checkpoint had essentially perfect viable first-branch choice with `wrong_commit_rate=0`. The intended suppressed-alternative mechanism therefore had no experimental object. | [Archive summary](./03_coverage_collapse_latent_alternatives/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / FALSIFIED AS A BROAD CLAIM** | Exploratory signal did not survive preregistered independent GSM1K confirmation, while the positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Behavioral KL stabilized as expected, but matched residual representation movement stabilized at least as fast; no temporal decoupling survived robustness checks. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |

## What the archived topics are for

Archived folders keep code, validation contracts, partial results where appropriate, and failure analyses so the same weak premise is not repeatedly rediscovered.

The archive reason should be interpreted precisely:

- **Topic 01** failed its substantive G0 hypothesis.
- **Topic 02** failed locked independent confirmation of its proposed new claim.
- **Topic 03** failed at the mechanism-level behavioral premise: the exact first-fork wrong-commitment event the latent story needed was absent.
- **Topic 04** failed before hypothesis testing because the required clean comparison could not be identified at sufficient scale.
- **Topic 05** was stopped even earlier at the conceptual-identification level: the proposed intervention could not distinguish retained competence from task simplification / conditional continuation.
- **Topic 06** failed at the prerequisite-acquisition layer: the chosen learner did not robustly instantiate the base phenomenon.
- **Topic 07** reproduced the motivating phenomenon, but the frozen architecture contrast was not large or stable enough to justify confirmation.
- **Topic 08** ultimately failed the stronger operational significance bar: the natural decoupling existed, but the deployed entropy proxy was weak rather than useless.
- **Topic 09** failed at natural common support: the paired identification required abundant bidirectional competence crossover, but the checkpoint family was nearly monotone at the state level. The broader self-knowledge question was not tested.

Several lessons recur strongly across the archive:

> **Many controls do not rescue a non-identifying intervention. Before running a large pilot, verify that the proposed observation could in principle discriminate the latent explanations of interest.**

> **If the gate and kill line get longer every time the question is clarified, revisit the question itself rather than automatically adding another control.**

> **Replicating a motivating phenomenon is not evidence that the proposed explanatory axis matters. Require the explanatory manipulation to produce a large, clean separation before investing in mechanisms.**

> **An aggregate metric is not the mechanism. Verify the exact local event required by the story before probing hidden states.**

Topic 09 adds two selection rules that should be enforced before another expensive run:

> **Different aggregate performance does not imply useful instance-level crossover. If the experiment depends on natural winner reversals, inspect or independently establish the joint per-item distribution first.**

> **A scientifically simple gate can still cost an afternoon. Resource cost is part of candidate selection: estimate the full environment-reproduction, inference and rollout burden before committing.**

A failed or inconclusive topic should not be revived by post-hoc metric/layer/model/control sweeps unless a genuinely new external observation changes the scientific premise and motivates a newly registered question or identification strategy.
