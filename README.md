# Candidate Research Topics

This repository collects research-topic candidates in the **hypothesis / pilot** stage, together with archived negative results, gray-zone stops, and identification failures.

A cross-topic record of failed candidates and reusable selection lessons is maintained in [`FAILURES_AND_LESSONS.md`](./FAILURES_AND_LESSONS.md). **Read it before registering a new topic.**

The current topic-selection rule is deliberately stricter than simple "paper + adjacent gap":

1. start from a **natural scientific question** that is interesting before mentioning models, probes, hidden states, checkpoints, SAEs, or implementation tricks;
2. require a strong established phenomenon / empirical tension rather than betting that a new phenomenon exists;
3. use modern AI systems only when they provide a uniquely clean experimental axis;
4. check both recent literature collision and collision with our own archived/failed topics;
5. test **conceptual identifiability before scaling experiments**: the proposed observable/intervention must actually distinguish the competing explanations;
6. apply the **complexity-smell rule**: if making the result interpretable requires an expanding chain of gates, matching rules, baselines, and alternative-explanation exclusions, stop adding controls and reconsider the question;
7. prefer a **one-clean-contrast** first experiment whose interpretation is nearly forced by the question;
8. define the earliest experiment that can invalidate the proposed interpretation, and estimate its real engineering / rollout / wall-clock cost before calling it cheap;
9. freeze the primary measurement before broad search over models/layers/thresholds;
10. reserve confirmation data whenever discovery involves tuning a measurement;
11. stop early if the core claim or identification strategy fails; do not rescue it with post-hoc metric/model/control sweeps;
12. for mechanism work, require **mechanism-level phenomenon replication** before hidden-state analysis: the exact decision point / failure event being explained must actually occur in the chosen system;
13. if identification relies on natural disagreement / crossover / reversal / transition, require useful **instance-level density** rather than aggregate score differences;
14. calibrate stochastic per-item gates against an explicit null;
15. separate **phenomenon existence** from **meaningful-regime existence**;
16. separate **aggregate signal** from **distributed mechanism**;
17. for whole-profile correspondence claims, separate shared global geometry from fine-grained correspondence;
18. apply the **positive-result excitement test before coding**: assume the cleanest positive result is true and ask whether a strong audience would actually care;
19. require an explicit **method-opening / then-what test**: a real phenomenon should expose a concrete lever, failure mode, or optimization target rather than ending at “we found a pattern.”

## Current candidates

| Priority | Topic | Natural question | Current gate |
| --- | --- | --- | --- |
| 1 | [24 — Where Does Closed-Loop Robustness Live in Hierarchical Robot Foundation Policies?](./24_hierarchical_feedback_attribution/) | When a hierarchical robot foundation policy recovers from a physical disturbance, which feedback layer actually caused the recovery? | Both pre-registration instrument gates passed before registration: P0 replay fidelity `10/10` in all three conditions with `0.000 rad` trajectory divergence, and P0b WBC seam liveness `D = 2.4e-02..4.6e-02 rad` over an exactly zero repeatability floor. The frozen physical-disturbance G0 now requires the `force=0` control column to reproduce P0 under the G0 code path, an effective push at the largest force, and `>=24/30` matched configs in every grid cell, before either attribution delta is read. |
| 2 | [21 — Where Does Long-Context Semantic Execution Break?](./21_semtrace_semantic_state_failure/) | When the same code remains lexically accessible in the middle of a long context, where does its operational computation break? | First reproduce the official SemTrace forced-sequential positional drop on Qwen2.5-Coder-7B-Instruct. Then require same-content start/middle pairs with start semantic success, lexical success at both positions, parseable middle semantic failure, aggregate semantic drop `>=0.15`, and at least 16 exact critical-cell cases. |
| 3 | [22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?](./22_medeinst_evidence_update/) | In an exact MedEinst Bias Trap, was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis? | Require clean released pair locality; reproduce the published Bias Trap phenomenon on seed-supported Qwen3-14B under zero-shot CoT; then require the same fixed random pair set to retain a dense exact Bias Trap event in direct-answer mode before any fixed-position mechanism work. |
| 4 | [16 — Does Citation Turn Hypotheses into Facts?](./16_citation_transmutation/) | When the same scientific claim is repeated through citations without new primary evidence, does its expressed epistemic certainty systematically increase? | First reproduce known transmutation cases, then audit ~100–300 high-precision same-claim citation edges. Proceed only if no-new-evidence edges show non-trivial unsupported certainty inflation and claim/evidence labels remain reliable. |
| 5 | [13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?](./13_repetition_temporal_spacing/) | With exactly the same repeated training multiset, does temporal spacing between identical exposures change pretraining damage? | Reproduce one strong repetition regime, then compare clustered/random/even spacing while fixing unique positions and repeated slots. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [23 — Do Robot Foundation Policies Learn Motor Equivalence Classes?](./23_motor_equivalence_classes/) | **ARCHIVED / FROZEN PANEL PREREQUISITES FAIL** | The frozen Psi0 + SIMPLE panel never supplied the required intersection. CloseDoor was fully competent (`30/30`) but freezing the right arm cost only `0.033` and disabling both arms still gave `30/30`, so there was no canonical arm motor program to substitute. OpenFaucet genuinely used the right arm but canonical success was only `11/30 = 0.367` against the frozen `0.70` competence bar. Crucially, the original four-condition CloseDoor design would have produced an overwhelming false positive: paired `right_disabled-full_hold = 0.967`, 95% CI `[0.90,1.00]`, with 29 apparent substitution events. | [Archive summary](./23_motor_equivalence_classes/ARCHIVE_SUMMARY.md) |
| [19 — Do Robot Foundation Policies Learn Task-Structured Feedback?](./19_task_structured_feedback/) | **ARCHIVED / PRIMARY METRIC IDENTIFICATION FAILURE** | The frozen joint-axis score was a tight numerical null on 8 observed configs (`ΔR=-0.0038`, 95% CI `[-0.0275,+0.0178]`), but raw Psi0 target responses were mostly orthogonal to the injected joint-space direction. A redundant arm can correct the same task-space error through a different joint coordination, so the frozen projection metric cannot identify task-space correction. In addition, wrist-moving was not guaranteed to be CloseDoor-task-relevant, and two configs were systematically missing under the collector. The scientific question remains unresolved; no post-hoc task-space metric rescue was run. | [Archive summary](./19_task_structured_feedback/ARCHIVE_SUMMARY.md) |
| [20 — Representation or Access?](./20_numeracy_representation_access/) | **ARCHIVED / ROBUST PHENOMENON, FROZEN CAUSAL ROUTES FAILED** | Same-prompt numerical ranking remained highly decodable when generation failed, and a scientific-notation wrong-answer attractor replicated on two untouched seeds. But G1 stopped at its frozen fresh prerequisite and G2's perfectly decodable, rank-orthogonal notation coordinate produced exactly `0/32` rescues with `ΔR=0`, CI `[0,0]`. Continuing would require post-hoc layer/token/subspace search; the remaining format-bias observation alone is too narrow for the intended paper scale. | [Archive summary](./20_numeracy_representation_access/ARCHIVE_SUMMARY.md) |
| [17 — Can a Cited Method Actually Be Reconstructed?](./17_shortcut_method_fidelity/) | **ARCHIVED / G0 INVALID, MEASUREMENT FAILURE** | The corrected open-full-text preflight inspected 21 lineages and flagged 14, but direct evidence review found 7 definite ontology/retrieval false positives, 7 unresolved multi-hop/supplement cases, and 0 confirmed documentary failures. Universal critical-unit templates, one-hop citation resolution and incomplete supplement recovery made the apparent effect non-identifiable. | [Archive summary](./17_shortcut_method_fidelity/ARCHIVE_SUMMARY.md) |
| [18 — Is Negative Behavioral Adaptation Intrinsically Harder?](./18_negative_behavioral_adaptation/) | **ARCHIVED / VALID G0, INCONCLUSIVE MODEL HETEROGENEITY** | Phi and Gemma showed large inhibition gaps while Qwen did not. The pooled effect could not override the frozen cross-family consistency failure; the result is model heterogeneity, not the general inhibition bottleneck needed for the method opening. | [Archive summary](./18_negative_behavioral_adaptation/ARCHIVE_SUMMARY.md) |
| [15 — Does Training-Time World Modeling Act Through a Predictive Policy State?](./15_predictive_policy_state/) | **ARCHIVED / KILLED AT MATCHED-TRAINING MEDIATION GATE** | Future supervision reliably made the adapter state more future-predictive, but the action path did not benefit; matched future-on/off training gave negative `M→Y` evidence even after restoring action capacity. | [Archive summary](./15_predictive_policy_state/ARCHIVE_SUMMARY.md) |
| [14 — Does Power-Law Learning Need a Persistent Head?](./14_powerlaw_persistent_head/) | **ARCHIVED / KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT** | The motivating power-law prerequisite was huge, but exact same-data Slow−Fast temporal reordering was near zero in four of five independently locked mapping/seed replications. One spectacular seed was preserved as an anomaly, not used to rescue the topic. | [Archive summary](./14_powerlaw_persistent_head/ARCHIVE_SUMMARY.md) |
| [12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?](./12_reasoning_necessity_vs_rl_leverage/) | **ARCHIVED / VALID G0, INCONCLUSIVE_DO_NOT_TUNE** | Raw necessity-vs-RL-leverage correlation was moderately positive, but the fine-grained relation vanished after removing broad depth structure; top peaks did not align. The result supports shared coarse depth geometry, not a strong layer-level law. | [Archive summary](./12_reasoning_necessity_vs_rl_leverage/ARCHIVE_SUMMARY.md) |
| [11 — What Does Diffusion Confidence Actually Know?](./11_dlm_confidence_internal_consistency/) | **ARCHIVED / FALSIFIED AT FROZEN G0** | Both prerequisites passed strongly, but the preregistered retroactive consistency signal on earlier unchanged reasoning-result tokens was effectively zero and far below the minimum-worthy effect. | [Archive summary](./11_dlm_confidence_internal_consistency/ARCHIVE_SUMMARY.md) |
| [10 — Is DLM Generation Order Invariant to Problem Isomorphisms?](./10_dlm_generation_order_invariance/) | **ARCHIVED / POSITIVE 4×4 G0, FAILED NON-TOY QUALIFICATION** | The 4×4 isomorphism effect was real and strong, but no scientifically meaningful 9×9 object could be established without unresolved model/data/configuration fishing. | [Archive summary](./10_dlm_generation_order_invariance/ARCHIVE_SUMMARY.md) |
| [09 — Does a VLA Know Its Own Limits?](./09_vla_own_limits/) | **ARCHIVED / KILLED AT G0 IDENTIFICATION SUPPORT** | The frozen LIBERO panel found too little natural bidirectional same-state competence crossover to support the paired self-knowledge test. | [Archive summary](./09_vla_own_limits/ARCHIVE_SUMMARY.md) |
| [08 — Does Action Diversity Track Functional Uncertainty?](./08_generative_policy_task_geometry/) | **ARCHIVED / KILLED AT THE OPERATIONAL BAR** | The original design was circular; the rebuilt PushT test found a real action/outcome decoupling but showed the deployed entropy monitor was weak-but-informative rather than systematically broken. | [Archive summary](./08_generative_policy_task_geometry/ARCHIVE_SUMMARY.md) |
| [07 — Old Blocks New, or New Erases Old?](./07_memory_interference_architecture/) | **ARCHIVED / INCONCLUSIVE AT FROZEN DISCOVERY GATE** | Transformer PI>RI replicated, but the preregistered Transformer–GatedDeltaNet explanatory contrast was not large or stable enough to justify confirmation. | [Archive summary](./07_memory_interference_architecture/ARCHIVE_SUMMARY.md) |
| [06 — When Does Helplessness Become a Worldview?](./06_helplessness_worldview/) | **ARCHIVED / KILLED AT ACQUISITION PREMISE** | The chosen LLM agent never robustly acquired the prerequisite controllability-dependent behavioral state, so the higher-order transfer question could not be meaningfully tested. | [Archive summary](./06_helplessness_worldview/ARCHIVE_SUMMARY.md) |
| [05 — Temporal Forgetting: Lost Skill or Lost Entry Point?](./05_temporal_forgetting_reentry/) | **ARCHIVED / CONCEPTUAL IDENTIFICATION FAILURE** | Prefix rescue changes the task condition and cannot distinguish retained competence from task simplification, search-space reduction, or conditional continuation. | [Archive summary](./05_temporal_forgetting_reentry/ARCHIVE_SUMMARY.md) |
| [04 — Confidence and Error Correction](./04_confidence_error_correction/) | **ARCHIVED / KILLED AT MEASUREMENT / IDENTIFICATION GATE** | A locked measurement repair left only 130 clean matched pairs, below the preregistered support floor; corrective SFT was never run. | [Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md) |
| [03 — Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | **ARCHIVED / KILLED AT BEHAVIORAL PREMISE** | The seed-paper reproduction audit found no meaningful late coverage degradation or wrong-commitment event, so the intended suppressed-alternative mechanism had no experimental object. | [Archive summary](./03_coverage_collapse_latent_alternatives/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / FALSIFIED AS A BROAD CLAIM** | The exploratory hidden-state signal did not survive locked independent GSM1K confirmation while the positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Behavioral KL stabilized as expected, but matched residual representation movement stabilized at least as fast; no temporal decoupling survived robustness checks. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |

## What the archived topics are for

Archived folders keep code, validation contracts, results, and failure analyses so the same weak premise is not repeatedly rediscovered.

The archive reason should be interpreted precisely: `hypothesis false`, `measurement failed`, `prerequisite absent`, `confirmation failed`, `explanatory axis too weak`, and `clean causal null` are different outcomes.

Topic 19 adds two important identification rules:

> **For redundant control systems, joint-axis restoration is not the same thing as task-space correction. If the scientific claim is about task-space feedback, the dependent variable must be defined in task/outcome space before data collection.**

> **A kinematically large end-effector perturbation is not automatically task-relevant. Manipulation relevance should be grounded in contact geometry, object state, or realized outcome.**

Topic 20 adds a particularly useful distinction:

> **A behavioral phenomenon can be extremely stable while the most obvious decodable representation is causally inert. Readability and behavioral alignment are not a license for open-ended mechanism search.**

Topic 23 adds three related intervention-design rules:

> **Before interpreting successful behavior after “removing” a mechanism or route, first show that the mechanism is causally used in the canonical behavior. A treatment label is not proof that the named causal object was actually removed.**

> **A perfect negative control does not rescue a non-identifying treatment. `full_hold = 0` ruled out accidental environment success, but it did not prove that `right_disabled` removed a right-arm motor program.**

> **Statistical strength cannot compensate for construct failure. Topic 23 produced a paired effect of `0.967` with 95% CI `[0.90,1.00]` and 29/30 apparent events, yet the scientific interpretation was false because the canonical arm program did not exist.**

Several lessons recur strongly across the archive:

> **Many controls do not rescue a non-identifying intervention. Before running a large pilot, verify that the proposed observation could in principle discriminate the explanations of interest.**

> **If the gate and kill line get longer every time the question is clarified, revisit the question itself rather than automatically adding another control.**

> **Replicating a motivating phenomenon is not evidence that the proposed explanatory axis matters. Require the explanatory manipulation to produce a large, clean separation before investing in mechanisms.**

> **An aggregate metric is not the mechanism. Verify the exact local event required by the story before probing hidden states.**

> **A perfectly decodable feature can still be causally irrelevant at the tested site. Manipulation checks plus a zero behavioral effect are evidence to stop, not an invitation to search the network until something moves.**

> **Use a second model/seed to replicate a strong discovery, not to search for a system where a gray-zone result becomes positive.**

> **Do not let one spectacular seed create a project. Robust aggregation and independently locked confirmation protect against promoting an outlier into a general law.**

A failed or inconclusive topic should not be revived by post-hoc metric/layer/model/control sweeps unless a genuinely new external observation changes the scientific premise and motivates a newly registered question or identification strategy.
