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
14. for stochastic per-item gates, calibrate the gate against an explicit null rather than assuming a large-looking threshold is noise-proof;
15. separate **phenomenon existence** from **meaningful-regime existence**: a clean effect on a toy or weak object does not justify a project if the scientifically relevant regime cannot be instantiated without model/data/configuration fishing;
16. separate **aggregate signal** from **distributed mechanism**: a sequence-level score may strongly track a property even when the specific local/causal readout required by the mechanistic story is effectively absent.

## Current candidates

| Priority | Topic | Natural question | Current gate |
| --- | --- | --- | --- |
| 1 | [15 — Does Training-Time World Modeling Act Through a Predictive Policy State?](./15_predictive_policy_state/) | When future prediction helps during training, does the deployed policy actually act through information about the future that remains in its native internal state? | On released Light-WAM, run a normal action pass and a full WAM-adapter-bypass pass on episode-disjoint samples; require the adapters to add held-out future predictability in the checkpoint's actual future-training latent space **and** to causally improve action readout. A positive result only advances to matched future-on/future-off training; it is not called mediation yet. |
| 2 | [12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?](./12_reasoning_necessity_vs_rl_leverage/) | Are the layers most necessary for reasoning also the layers where isolated RL updates can improve reasoning most effectively? | Compare the complete frozen layer-ablation curve with the complete single-layer-RL leverage curve on a matched model/task. |
| 3 | [13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?](./13_repetition_temporal_spacing/) | With exactly the same repeated training multiset, does temporal spacing between identical exposures change pretraining damage? | Reproduce one strong repetition regime, then compare clustered/random/even spacing while fixing unique positions and repeated slots. |
| 4 | [14 — Does Power-Law Learning Need a Persistent Head?](./14_powerlaw_persistent_head/) | Does a power-law curriculum work because some skills remain frequent long enough to scaffold others, or is momentary frequency asymmetry sufficient? | Reproduce static power-law > uniform, then compare count-matched slow- vs fast-rotation schedules with the same per-block spectrum. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [11 — What Does Diffusion Confidence Actually Know?](./11_dlm_confidence_internal_consistency/) | **ARCHIVED / FALSIFIED AT FROZEN G0** | Both protocol prerequisites passed strongly (`arithmetic gap=0.426`, `semantic-alias gap=0.215`), but the preregistered retroactive consistency effect on unchanged middle reasoning-result tokens was `-0.000003`, 95% CI `[-0.000055, 0.000025]`, versus a locked `0.010` minimum-worthy floor. Full-sequence confidence remained strongly consistency-sensitive, but that metric includes the manipulated suffix and therefore does not rescue the global/distributed interpretation. | [Archive summary](./11_dlm_confidence_internal_consistency/ARCHIVE_SUMMARY.md) |
| [10 — Is DLM Generation Order Invariant to Problem Isomorphisms?](./10_dlm_generation_order_invariance/) | **ARCHIVED / POSITIVE 4×4 G0, FAILED NON-TOY QUALIFICATION** | The published UPO 4×4 setting produced a real and independently confirmed effect: exact Sudoku isomorphisms caused roughly 40–45% solve/fail flips. But 9×9 LLaDA-8B had `0/8` exact solves, and a seed-aligned Dream-7B reconstruction reached only `6/100` exact at epoch 2 and `3/100` at epoch 5 while training loss collapsed. The phenomenon exists, but a scientifically meaningful 9×9 object could not be established without unresolved model/data/configuration fishing or substantially larger infrastructure. | [Archive summary](./10_dlm_generation_order_invariance/ARCHIVE_SUMMARY.md) |
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
- **Topic 10** passed a clean 4×4 phenomenon-existence gate but failed the meaningful-regime gate: two 9×9 qualification routes did not produce a competent object, and further rescue would require unresolved experimental-object search.
- **Topic 11** passed both protocol prerequisites but falsified its substantive project-level hypothesis: aggregate/full confidence was consistency-sensitive, while the preregistered retroactive signal on earlier unchanged reasoning-result tokens was effectively zero and far below the frozen minimum-worthy effect.

Several lessons recur strongly across the archive:

> **Many controls do not rescue a non-identifying intervention. Before running a large pilot, verify that the proposed observation could in principle discriminate the latent explanations of interest.**

> **If the gate and kill line get longer every time the question is clarified, revisit the question itself rather than automatically adding another control.**

> **Replicating a motivating phenomenon is not evidence that the proposed explanatory axis matters. Require the explanatory manipulation to produce a large, clean separation before investing in mechanisms.**

> **An aggregate metric is not the mechanism. Verify the exact local event required by the story before probing hidden states.**

Topic 09 adds two selection rules that should be enforced before another expensive run:

> **Different aggregate performance does not imply useful instance-level crossover. If the experiment depends on natural winner reversals, inspect or independently establish the joint per-item distribution first.**

> **A scientifically simple gate can still cost an afternoon. Resource cost is part of candidate selection: estimate the full environment-reproduction, inference and rollout burden before committing.**

Topic 10 adds another selection rule:

> **Phenomenon existence and meaningful-regime existence are separate gates. A striking toy-scale effect is not enough if the scientifically relevant regime cannot be instantiated without searching over model, data, prompt, training, or decoding choices.**

Topic 11 adds two more:

> **Aggregate signal is not distributed mechanism. A score can strongly reflect a property at sequence level while the local/causal signal required by the mechanistic interpretation is effectively absent.**

> **A positive secondary metric must not rescue a failed frozen primary when the secondary includes the intervention itself or otherwise measures a different object.**

A failed or inconclusive topic should not be revived by post-hoc metric/layer/model/control sweeps unless a genuinely new external observation changes the scientific premise and motivates a newly registered question or identification strategy.
