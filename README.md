# Candidate Research Topics

This repository tracks research-topic candidates from first hypothesis through preregistered validation, negative results, and archive.

Two documents are mandatory before registering a new topic:

- [`FAILURES_AND_LESSONS.md`](./FAILURES_AND_LESSONS.md) — why previous candidates failed and what should transfer;
- [`advisor_topic_search/COLLISION_AND_INTERNAL_HISTORY_POLICY.md`](./advisor_topic_search/COLLISION_AND_INTERNAL_HISTORY_POLICY.md) — how to distinguish normal literature overlap from real collision, and how prior local failures constrain new search.

## Status precedence

When documents disagree, use this order:

```text
actual local G0/G1 result + ARCHIVE_SUMMARY
>
numbered-topic README
>
root README
>
advisor_topic_search/ACTIVE_CANDIDATES.md
>
ROUND_*.md historical search logs
```

A later search round must never resurrect a topic that has a real scientific/identification stop. A measurement-invalid run must be classified by what actually failed; it is not automatically a scientific kill.

## Selection rules

1. Start from a natural scientific question, not a probe or model component.
2. Require a real experimental object before mechanism work.
3. Check both recent literature and **our own archived topics**.
4. External overlap is expected in modern AI. The question is whether enough independent ACL / EMNLP / NAACL narrative remains—not whether nobody has touched the area.
5. A real collision means the nearest work already occupies the same core question, decisive contrast, title-level conclusion, and most of the mechanism/intervention story.
6. Internal failures are stronger evidence than external overlap. Do not recycle a failed hypothesis or identification route under a new name.
7. Test conceptual identifiability before scaling.
8. Prefer one clean first contrast with frozen measurement and explicit stop conditions.
9. Separate prerequisite phenomenon, explanatory axis, and mechanism.
10. A robust seed phenomenon does not imply our proposed explanation is correct.
11. Positive-result excitement matters: the cleanest result should support a main-conference-sized story, not just a reasonable follow-up.
12. Stop rather than model/prompt/layer/schedule-shop after a frozen scientific negative.
13. Measurement repair is allowed only when the failure mode is concretely localized and the repair is narrower than the scientific experiment. Prefer output-preserving, outcome-blind repairs that keep model/sample/seed/thresholds frozen and do not expose the scorer to ground truth. Once the bounded repair for the localized defect still fails support, archive the route rather than tuning indefinitely.

---

# Current numbered topics

| Priority | Topic | Status | Current gate |
|---:|---|---|---|
| 1 | [24 — Where Does Closed-Loop Robustness Live in Hierarchical Robot Foundation Policies?](./24_hierarchical_feedback_attribution/) | **ACTIVE / G0 NEXT** | Replay fidelity and WBC seam-liveness instrument gates passed; run the frozen physical-disturbance attribution G0. |
| 2 | [25 — When Does Test-Time Reasoning Help Context Use, and When Does It Hurt?](./25_reasoning_context_use_boundary/) | **REGISTERED / REPRODUCTION RECEIPT NEXT** | Reproduce the pinned Weakest-Link Qwen3-8B think/no-think seed relation; only then run the frozen matched atomic-vs-composed G0. |
| 3 | [16 — Does Citation Turn Hypotheses into Facts?](./16_citation_transmutation/) | **REGISTERED / G0 NOT RUN** | Identification is hardened; next step is the high-precision evidence-provenance G0. |

---

# Archived numbered topics

| Topic | Final decision | Why it stopped | Summary |
|---|---|---|---|
| [23 — Motor Equivalence Classes](./23_motor_equivalence_classes/) | **ARCHIVED / FROZEN PANEL PREREQUISITES FAIL** | No task in the frozen panel simultaneously supplied adequate competence and the required canonical arm program. | [Archive summary](./23_motor_equivalence_classes/ARCHIVE_SUMMARY.md) |
| [22 — MedEinst Evidence Update](./22_medeinst_evidence_update/) | **ARCHIVED / MEASUREMENT_CANONICALIZATION_FAILURE / NO SCIENTIFIC VERDICT** | V3 rescored the exact frozen v2 CoTs and improved invalidity from `62.5%` to `32.42%`; all substantive Bias Trap gates passed (`109` control-correct, `43` traps, BTR `0.3945`, Wilson lower `0.3078`, `14` transitions), but support remained far above the frozen `<=10%` invalid ceiling. Direct G0c was not run. | [Archive summary](./22_medeinst_evidence_update/ARCHIVE_SUMMARY.md) |
| [21 — SemTrace Semantic-State Failure](./21_semtrace_semantic_state_failure/) | **ARCHIVED / STOP_UPSTREAM_SEED_NOT_REPRODUCED** | The exact official Qwen2.5-Coder-7B prerequisite completed, but edge mean was `0.000625` vs required `>=0.30` and edge-to-middle drop was `0.000625` vs required `>=0.20`; custom mechanism G0 was not run. | [Archive summary](./21_semtrace_semantic_state_failure/ARCHIVE_SUMMARY.md) |
| [20 — Representation or Access?](./20_numeracy_representation_access/) | **ARCHIVED / FROZEN CAUSAL ROUTES FAILED** | Robust behavior and decodability survived, but frozen causal rescue routes failed; remaining observation was too narrow for the intended paper. | [Archive summary](./20_numeracy_representation_access/ARCHIVE_SUMMARY.md) |
| [19 — Task-Structured Feedback](./19_task_structured_feedback/) | **ARCHIVED / PRIMARY METRIC IDENTIFICATION FAILURE** | Joint-axis restoration did not identify task-space correction in a redundant controller. | [Archive summary](./19_task_structured_feedback/ARCHIVE_SUMMARY.md) |
| [18 — Negative Behavioral Adaptation](./18_negative_behavioral_adaptation/) | **ARCHIVED / INCONCLUSIVE MODEL HETEROGENEITY** | Large effects appeared in some families but not others; the frozen general cross-family claim failed. | [Archive summary](./18_negative_behavioral_adaptation/ARCHIVE_SUMMARY.md) |
| [17 — Cited Method Reconstruction](./17_shortcut_method_fidelity/) | **ARCHIVED / MEASUREMENT FAILURE** | Apparent documentary failures collapsed under direct ontology/retrieval audit. | [Archive summary](./17_shortcut_method_fidelity/ARCHIVE_SUMMARY.md) |
| [15 — Predictive Policy State](./15_predictive_policy_state/) | **ARCHIVED / MEDIATION GATE FAILED** | Future supervision increased future-predictive state but did not improve the matched action path. | [Archive summary](./15_predictive_policy_state/ARCHIVE_SUMMARY.md) |
| [14 — Power-Law Persistent Head](./14_powerlaw_persistent_head/) | **ARCHIVED / NO MEANINGFUL TEMPORAL PERSISTENCE EFFECT** | Exact same-data Slow−Fast temporal reordering was near zero in four of five locked replications. | [Archive summary](./14_powerlaw_persistent_head/ARCHIVE_SUMMARY.md) |
| [13 — Temporal Spacing of Repeated Pretraining Data](./13_repetition_temporal_spacing/) | **ARCHIVED / NO_EVIDENCE_SPACING_IN_LOCKED_TEST** | Repetition damage reproduced in 4/4 trials, but `clustered-even` changed sign (`-0.001534,+0.010758,+0.001005,-0.009134`), so the proposed spacing explanation was not stable. | [Archive summary](./13_repetition_temporal_spacing/ARCHIVE_SUMMARY.md) |
| [12 — Necessity vs. RL Adaptation Leverage](./12_reasoning_necessity_vs_rl_leverage/) | **ARCHIVED / INCONCLUSIVE_DO_NOT_TUNE** | Fine-grained relation disappeared after removing broad depth structure. | [Archive summary](./12_reasoning_necessity_vs_rl_leverage/ARCHIVE_SUMMARY.md) |
| [11 — Diffusion Confidence](./11_dlm_confidence_internal_consistency/) | **ARCHIVED / FALSIFIED AT FROZEN G0** | Prerequisites passed, but retroactive consistency signal was effectively zero. | [Archive summary](./11_dlm_confidence_internal_consistency/ARCHIVE_SUMMARY.md) |
| [10 — DLM Generation Order Invariance](./10_dlm_generation_order_invariance/) | **ARCHIVED / FAILED NON-TOY QUALIFICATION** | Strong 4×4 effect did not yield a clean meaningful 9×9 object without configuration fishing. | [Archive summary](./10_dlm_generation_order_invariance/ARCHIVE_SUMMARY.md) |
| [09 — VLA Own Limits](./09_vla_own_limits/) | **ARCHIVED / IDENTIFICATION SUPPORT FAILED** | Too little natural bidirectional same-state competence crossover existed for the paired test. | [Archive summary](./09_vla_own_limits/ARCHIVE_SUMMARY.md) |
| [08 — Action Diversity vs. Functional Uncertainty](./08_generative_policy_task_geometry/) | **ARCHIVED / OPERATIONAL BAR FAILED** | Rebuilt test found decoupling but not the strong monitor failure required by the story. | [Archive summary](./08_generative_policy_task_geometry/ARCHIVE_SUMMARY.md) |
| [07 — Old Blocks New, or New Erases Old?](./07_memory_interference_architecture/) | **ARCHIVED / FROZEN DISCOVERY GATE INCONCLUSIVE** | PI>RI replicated, but the proposed architecture contrast was too weak/unstable to justify confirmation. | [Archive summary](./07_memory_interference_architecture/ARCHIVE_SUMMARY.md) |
| [06 — Helplessness Worldview](./06_helplessness_worldview/) | **ARCHIVED / ACQUISITION PREMISE FAILED** | The chosen agent never robustly acquired the prerequisite controllability-dependent state. | [Archive summary](./06_helplessness_worldview/ARCHIVE_SUMMARY.md) |
| [05 — Temporal Forgetting: Lost Skill or Lost Entry Point?](./05_temporal_forgetting_reentry/) | **ARCHIVED / CONCEPTUAL IDENTIFICATION FAILURE** | Prefix rescue changes the task and cannot distinguish retained uncued competence from simplification/search reduction/conditional continuation. | [Archive summary](./05_temporal_forgetting_reentry/ARCHIVE_SUMMARY.md) |
| [04 — Confidence and Error Correction](./04_confidence_error_correction/) | **ARCHIVED / MEASUREMENT-COMMON-SUPPORT FAILURE** | Locked measurement repair left too few clean matched pairs for the preregistered comparison. | [Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md) |
| [03 — Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | **ARCHIVED / BEHAVIORAL PREMISE FAILED** | Seed reproduction showed no meaningful late coverage degradation / wrong-commitment object. | [Archive summary](./03_coverage_collapse_latent_alternatives/ARCHIVE_SUMMARY.md) |
| [02 — DLM Trajectory Fate](./02_dlm_trajectory_fate/) | **ARCHIVED / LOCKED CONFIRMATION FAILED** | Exploratory hidden-state signal collapsed on independent confirmation while positive control remained strong. | [Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md) |
| [01 — Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | **ARCHIVED / KILLED** | Representation movement stabilized at least as fast as behavior; no temporal decoupling survived. | [Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md) |

---

# Why archives matter

Archived topics are experimental evidence, not discarded brainstorming.

Before proposing a new topic, ask:

```text
Which old topic is closest?
What exactly failed there?
Was it the phenomenon, measurement, identification, explanatory axis, or confirmation?
Does the new idea genuinely change that failed premise or route?
```

A broad scientific domain may be revisited. A failed hypothesis or non-identifying intervention may not be silently recycled.

Especially important recent examples:

- **Topic 05:** a natural storage-vs-access question can still fail because the intervention does not identify uncued retention.
- **Topic 13:** a robust motivating phenomenon does not make the proposed temporal-spacing explanation true.
- **Topic 21:** official artifact availability does not mean the exact prerequisite phenomenon reproduces on the selected platform.
- **Topic 22:** measurement repair can be justified when a new defect is explicitly localized, but after the bounded output-preserving canonicalization repair still leaves `32.42%` invalid support, the local route must stop. This archives the measurement route—not the scientific hypothesis.
- **Topic 20:** a decodable feature can be causally inert.
- **Topic 23:** a statistically overwhelming effect can still be scientifically false when the treatment does not manipulate the claimed object.