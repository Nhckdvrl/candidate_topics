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
16. separate **aggregate signal** from **distributed mechanism**: a sequence-level score may strongly track a property even when the specific local/causal readout required by the mechanistic story is effectively absent;
17. for **whole-profile correspondence claims**, separate shared global geometry from fine-grained correspondence: two stable layer/checkpoint/time profiles can correlate because both follow the same broad axis without the same local units actually matching;
18. apply the **positive-result excitement test before coding**: assume the cleanest positive result is true and ask whether a strong audience would actually be excited, rather than merely say “reasonable”; a clean but unsurprising result is not enough;
19. require an explicit **method-opening / then-what test**: if the phenomenon is established, it should expose a concrete lever, failure mode, or optimization target that naturally supports a follow-up method contribution. Reject topics whose endpoint is essentially “we proved the pattern exists... and then nothing follows.”

## Current candidates

| Priority | Topic | Natural question | Current gate |
| --- | --- | --- | --- |
| 1 | [20 — Representation or Access? Why Can LLMs Encode Numerical Magnitude but Fail to Use It?](./20_numeracy_representation_access/) | When mixed-notation numerical comparison fails, is the ordering absent from the model's representation, or is it already present immediately before generation but not used by the decision? | On seed-exact Qwen3-8B + `int_sci_compare`, use the same balanced 5-shot prompt for probing and greedy generation. Proceed only if the locked hard subset retains probe accuracy `>=0.80`, a probe-generation gap `>=0.15`, at least 30 `probe-correct / generation-wrong` cases, full-test probe accuracy `>=0.90`, and `<5%` invalid outputs. |
| 2 | [19 — Do Robot Foundation Policies Learn Task-Structured Feedback?](./19_task_structured_feedback/) | Given equally large body perturbations, does a foundation robot policy selectively correct deviations that change active end-effector task geometry while accommodating Jacobian-null, task-equivalent body deviations? | On released Ψ₀ + SIMPLE CloseDoor, physically construct equal-norm wrist-moving vs full-wrist-pose-null right-arm perturbations at the same deployed state, re-render both observations, use common-random-number first-target queries, and require episode-level `ΔR = R_task-R_null >= 0.20` with 95% CI above zero. |
| 3 | [16 — Does Citation Turn Hypotheses into Facts?](./16_citation_transmutation/) | When the same scientific claim is repeated through citations without new primary evidence, does its expressed epistemic certainty systematically increase? | First reproduce known transmutation cases, then audit ~100–300 high-precision same-claim citation edges. Proceed only if no-new-evidence edges show non-trivial unsupported certainty inflation and claim/evidence labels remain reliable. |
| 4 | [13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?](./13_repetition_temporal_spacing/) | With exactly the same repeated training multiset, does temporal spacing between identical exposures change pretraining damage? | Reproduce one strong repetition regime, then compare clustered/random/even spacing while fixing unique positions and repeated slots. |

## Archived topics

| Topic | Final decision | Why it was stopped | Summary |
| --- | --- | --- | --- |
| [17 — Can a Cited Method Actually Be Reconstructed?](./17_shortcut_method_fidelity/) | **ARCHIVED / G0 INVALID, MEASUREMENT FAILURE** | The corrected open-full-text preflight inspected 21 lineages and flagged 14, but direct evidence review found 7 definite ontology/retrieval false positives, 7 unresolved multi-hop/supplement cases, and 0 confirmed documentary failures. Universal critical-unit templates, one-hop citation resolution and incomplete supplement recovery made the apparent effect non-identifiable; this triggers the preregistered complexity-smell stop. | [Archive summary](./17_shortcut_method_fidelity/ARCHIVE_SUMMARY.md) |
| [18 — Is Negative Behavioral Adaptation Intrinsically Harder?](./18_negative_behavioral_adaptation/) | **ARCHIVED / VALID G0, INCONCLUSIVE MODEL HETEROGENEITY** | The fully matched three-family panel produced Phi Δ=0.391, Gemma Δ=0.203 and Qwen Δ=0.047; pooled Δ=0.214 [0.135, 0.292]. The large pooled effect could not override the frozen cross-model consistency failure, while two strong families prevented a clean-null kill. The result is model heterogeneity, not the general inhibition bottleneck needed for the method opening; no rescue sweep is allowed. | [Archive summary](./18_negative_behavioral_adaptation/ARCHIVE_SUMMARY.md) |
| [15 — Does Training-Time World Modeling Act Through a Predictive Policy State?](./15_predictive_policy_state/) | **ARCHIVED / KILLED AT MATCHED-TRAINING MEDIATION GATE** | G0 on two released Light-WAM checkpoints found the explicit WAM-adapter pathway strongly causal for action but with zero held-out future-predictive gain. Two matched future-on/future-off trainings (isolated: adapters the only shared module; capacity-restored: LoRA added back but gradient-routed to action loss only) both replicated `T→M` (future supervision reliably makes the adapter state more future-predictive) but falsified `M→Y`: action loss was reliably worse under future-on, and `ΔC=C_on−C_off` was negative with a 95% CI excluding zero in every evaluated checkpoint of both designs. Giving the action path ~37× more non-shared capacity did not reverse the result. | [Archive summary](./15_predictive_policy_state/ARCHIVE_SUMMARY.md) |
| [14 — Does Power-Law Learning Need a Persistent Head?](./14_powerlaw_persistent_head/) | **ARCHIVED / KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT** | The clean power-law prerequisite was extremely strong (`median Static−Uniform exact-AUC=+0.9300`, `5/5` positive), so the experimental object was unquestionably alive. But the exact same-data Slow−Fast intervention had median `+0.0095`, with `4/5` independently locked mapping/seed replications inside `|gap|<=0.06`. One seed showed a large `+0.7106` Slow advantage, but it did not replicate and cannot support a general persistence law without post-hoc mapping search. G1 was not run. | [Archive summary](./14_powerlaw_persistent_head/ARCHIVE_SUMMARY.md) |
| [12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?](./12_reasoning_necessity_vs_rl_leverage/) | **ARCHIVED / VALID G0, INCONCLUSIVE_DO_NOT_TUNE** | The functional-necessity profile was highly stable across MATH500/GSM8K (`rho=0.878`), and raw necessity-vs-RL-leverage correlation was moderately positive (`rho=0.355`, 90% bootstrap CI `[0.300,0.402]`). But the preregistered fine-grained relation vanished after removing broad quadratic depth structure (`rho=-0.238`), top-5 overlap was `1` versus random expectation `0.89`, and specific layer peaks did not align. The result supports shared coarse depth organization, not a strong layer-level law. | [Archive summary](./12_reasoning_necessity_vs_rl_leverage/ARCHIVE_SUMMARY.md) |
| [11 — What Does Diffusion Confidence Actually Know?](./11_dlm_confidence_internal_consistency/) | **ARCHIVED / FALSIFIED AT FROZEN G0** | Both protocol prerequisites passed strongly (`arithmetic gap=0.426`, `semantic-alias gap=0.215`), but the preregistered retroactive consistency effect on unchanged middle reasoning-result tokens was `-0.000003`, 95% CI `[-0.000055, 0.000025]`, versus a locked `0.010` minimum-worthy floor. Full-sequence confidence remained strongly consistency-sensitive, but that metric includes the manipulated suffix and therefore does not rescue the global/distributed interpretation. | [Archive summary](./11_dlm_confidence_internal_consistency/ARCHIVE_SUMMARY.md) |
| [10 — Is DLM Generation Order Invariant to Problem Isomorphisms?](./10_dlm_generation_order_invariance/) | **ARCHIVED / POSITIVE 4×4 G0, FAILED NON-TOY QUALIFICATION** | The published UPO 4×4 setting produced a real and independently confirmed effect: exact Sudoku isomorphisms caused roughly 40–45% solve/fail flips. But 9×9 LLaDA-8B had `0/8` exact solves, and a seed-aligned Dream-7B reconstruction reached only `6/100` exact at epoch 2 and `3/100` exact at epoch 5 while training loss collapsed. The phenomenon exists, but a scientifically meaningful 9×9 object could not be established without unresolved model/data/configuration fishing or substantially larger infrastructure. | [Archive summary](./10_dlm_generation_order_invariance/ARCHIVE_SUMMARY.md) |
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
- **Topic 12** produced a valid, reproducible necessity profile but failed to establish the proposed fine-grained mapping to causal RL adaptation leverage. The moderate raw correlation was explained by shared broad depth structure; local layer correspondence vanished after the frozen depth control.
- **Topic 14** is a clean substantive null: the motivating power-law advantage was extremely strong, but exact same-data temporal reordering produced no stable persistence effect across four of five independently locked mapping/seed replications. The one large seed-1 effect is preserved as an anomaly, not used as a post-hoc rescue.
- **Topic 17** failed at measurement identification: the automated missing-unit signal collapsed under direct review into ontology false positives and unresolved citation/supplement branches. The underlying real-world question remains open, but the promised one-shot G0 is not viable without bespoke expert reconstruction becoming the project.
- **Topic 18** produced a valid matched behavioral G0 but failed the model-generality bar: Phi and Gemma showed large inhibition gaps while Qwen did not. The pooled effect is not allowed to erase family heterogeneity, so the topic is archived in its frozen no-tune gray zone.

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

Topic 12 adds three more:

> **Two stable structures do not imply a meaningful mapping. Establish that the local peaks/troughs correspond, not merely that both profiles vary along the same global axis.**

> **For layer-wise/checkpoint-wise profile comparisons, remove or explicitly model the obvious global geometry before interpreting raw correlation as a mechanistic law.**

> **Use a second model to replicate a strong discovery, not to search for a model where a gray-zone result becomes positive.**

Topic 14 adds three more:

> **A strong prerequisite can make a negative mechanistic result especially informative. If the base phenomenon is huge but the explanatory intervention is near zero, accept the null instead of blaming the testbed.**

> **Randomize arbitrary identities when the intended claim is general. Replication across model seeds is not enough if all runs reuse the same arbitrary mapping, subset, or assignment.**

> **Do not let one spectacular seed create a project. Robust aggregation and per-seed consistency requirements protect against promoting a mapping-specific outlier into a general law.**

A failed or inconclusive topic should not be revived by post-hoc metric/layer/model/control sweeps unless a genuinely new external observation changes the scientific premise and motivates a newly registered question or identification strategy.