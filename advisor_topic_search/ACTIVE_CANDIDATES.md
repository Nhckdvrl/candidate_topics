# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 是历史搜索记录；一旦和后来的本地实验冲突，必须以 numbered-topic 的实际 G0/G1 结果为准。

Last updated: 2026-08-24
Source of current ranking: `ROUND_14_2026-08-24.md`

---

# 0. Hard rules after internal-history reconciliation

1. **External overlap is normal.** 2026 年 AI/NLP 不追求“完全没人做过”。Collision audit 问的是：最接近工作都成立以后，是否仍剩下 ACL / EMNLP / NAACL 篇幅的独立主问题、主实验、主结论和后续机制/方法空间。
2. **Internal scientific failure is evidence.** 已经在本仓库跑过并停止的 hypothesis / identification route，不能因为新一轮搜题忘记结果而重新升为 active。
3. **Measurement failure和scientific failure必须分开。** 一个 scorer / parser / runtime 失败不能自动归档科学问题；但 repair 必须针对已定位的 measurement defect，而且不能借机换模型、换样本、松阈值或 outcome-driven tuning。
4. Status precedence:

```text
numbered-topic actual local result
>
numbered-topic README / archive summary if terminal
>
ACTIVE_CANDIDATES
>
ROUND search logs
```

5. Before promoting any search candidate, audit `FAILURES_AND_LESSONS.md` and numbered-topic directories for internal collision.
6. `REPRODUCTION_RECEIPT_POLICY.md` and `MOTHER_TOPIC_BRANCHING_POLICY.md` remain mandatory.
7. Generic reasoning / CoT / test-time-compute mechanism remains advisor-low-priority.
8. A second measurement repair is defensible only when the new failure mode is itself newly localized and the repair is **narrower, outcome-blind, and preferably output-preserving**. If that repair still fails support, stop the route.

Detailed policy: `COLLISION_AND_INTERNAL_HISTORY_POLICY.md`.

---

# A. TOP SCIENCE / TOP EXECUTABLE SEARCH OBJECTS

## A1. Does Knowledge Arbitration Have a Training History?

**Seed:** ACL 2026 Main, *How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models*.

**Question:**

> If two models consume the exact same training multiset and end with the same empirical data distribution, can different temporal histories of reliable vs conflicting evidence leave them with persistently different parametric-vs-context knowledge-use policies?

This is a learning-history / hysteresis question, not generic `training order matters`.

A full main-conference narrative remains available despite neighboring training-order work:

- fixed-multiset hysteresis phenomenon;
- directionality / reversibility / washout;
- developmental window;
- replay/interleaving intervention;
- representation only after behavior stands;
- continual-pretraining / post-training generalization.

**Blocker:** no trustworthy accessible official reproduction artifact has been verified. Do not reverse-engineer the seed until it reproduces by tuning.

**Status:** `SCIENCE_TOP / HOLD_FOR_ACCESSIBLE_OFFICIAL_ARTIFACT`.

---

## A2. Positional Imprinting of Parametric Knowledge

**Seed:** NAACL 2025 Main / Oral, *Where is the answer? An empirical study of positional bias for parametric knowledge extraction in language model*.

**Question:**

> If facts ultimately receive identical early/late position exposure counts and the same later washout training, does the position in which a fact was initially acquired leave a persistent difference in final parametric accessibility?

External work on positional bias and training-history persistence does **not** kill this topic by itself. The relevant test is whether the acquisition-context object can support a full story:

- persistent fact-level history effect;
- washout law and asymmetry;
- storage/accessibility diagnostics;
- whether shuffle/D-AR erase history dependence rather than merely improve average accuracy;
- internal trace / causal analysis;
- cross-model/domain generalization.

**Artifact:** complete official code/data path verified at `omron-sinicx/WhereIsTheAnswer`, frozen upstream commit `910fcddec93f7400b58257d70abf1dab31f1e179`.

**Important G0 repair before implementation:** prefer within-model fact-level counterbalancing after receipt, so early-P1 and early-P5 fact groups share the same optimizer/LR trajectory.

Frozen prerequisite: `g0/POSITIONAL_IMPRINTING_RECEIPT.md`.

**Status:** `TOP_EXECUTABLE / ARTIFACT_VERIFIED / RECEIPT_PENDING / DO_NOT_REGISTER_YET`.

---

# B. EXISTING NUMBERED ACTIVE OBJECTS

## B1. MedEinst / Topic 22 — evidence update

This is **not a new advisor-search candidate**. It is an already-registered numbered topic with an active measurement repair.

Scientific question:

> In exact counterfactual Bias Trap pairs, was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?

Repository truth:

- G0a pair locality passed on all 5,383 released pairs;
- G0b-v1 was measurement-invalid (`81.25%` invalid) because of known Qwen3 decoding/budget/marker defects;
- G0b-v2 fixed those defects on the same 256 pairs/model/seed;
- every substantive v2 Bias Trap gate passed on the resolvable outputs;
- pair invalid rate remained `160/256 = 62.5%` against the frozen `<=10%` gate;
- all thinking traces closed and no branch hit the 32,768-token ceiling;
- failure localized to `unresolved_final`: free-form diagnosis wording could not be mapped by exact/sub-string parsing to the benchmark's closed 49-pathology vocabulary;
- direct mode was not run.

### Why Topic 22 is not terminal

The previous Round-13 archive decision was too mechanical. V2 exposed a **newly localized label-interface defect**, not another failure of the same repaired runtime problem.

G0b-v3 is therefore allowed as a strictly scoring-only, outcome-blind repair:

- reuse the exact frozen v2 CoT outputs;
- do not regenerate them;
- same model/pairs/seed/decoding/gates;
- semantic canonicalizer sees only post-thinking final-answer text + 49 closed labels;
- no narrative, GT, case type, or control/trap identity;
- explicit abstention;
- two fixed label orders, accept only agreement;
- 49/49 self-mapping preflight before benchmark rescoring.

If v3 still has invalid rate `>10%`, stop the local measurement route. If v3 measurement is healthy but substantive gates fail, that is a real seed-reproduction stop. Only a full CoT pass authorizes direct-mode G0c.

See:

- `../22_medeinst_evidence_update/G0_RESULTS.md`
- `../22_medeinst_evidence_update/VALIDATION_AUDIT.md`
- `../22_medeinst_evidence_update/MEASUREMENT_FAILURE_V2.md`
- `../22_medeinst_evidence_update/g0_recanonicalize_v3.py`

**Status:** `NUMBERED_TOPIC_22 / ACTIVE / G0B_V3_READY / NO_SCIENTIFIC_VERDICT_YET`.

---

# C. ACTIVE HOLD / SECONDARY SEARCH OBJECTS

## C1. Parametric Encoding Specificity Across Input Structures

**Seed:** ACL 2026 Findings, SParK-Eval.

**Question:**

> When structured-data training looks like poor knowledge acquisition under ordinary QA, how much of the loss is true storage failure versus format-bound access?

The paper-sized object is not `try a table-like prompt`. It is whether parametric knowledge becomes format-invariant after acquisition, measured through a controlled encoding × retrieval-format matrix plus development/intervention/generalization.

Neighboring format-diversity work consumes part of the intervention story but does not automatically remove the mother question.

**Blockers:** Findings seed, artifact not yet complete/verified, construct must avoid prompt-engineering interpretation.

**Status:** `ACTIVE_HOLD / ARTIFACT+CONSTRUCT_GATE`.

## C2. ChronoScope — Temporal Scope Dynamics

**Seed:** ACL 2026 Main, *Evaluating Temporal Consistency in Multi-Turn Language Models*.

Potential paper object:

> How does an established historical reference-time state decay, suffer interference, recover under reinstatement, and compete with the present-day default?

This is broader than one `represented vs used` probe and can support decay/interference/reinstatement/default-attractor experiments.

Existing screen: `g0/chronoscope_drift_g0.py`.

**Status:** `ACTIVE_B / EXECUTABLE / ADVISOR_FIT_BELOW_A`.

---

# D. DEEP AUDIT / RESOURCE — not active top pool

## D1. Context-shaped truth geometry → source arbitration

Promotion requires a full independent story linking conflict behavior, geometry, causal source choice, and intervention—not another geometry plot.

**Status:** `DEEP_AUDIT`.

## D2. Table DRE

Could still support a main-conference mechanism paper if a clean, dominant referencing/binding bottleneck and causal rescue are established. Current mother framing is not yet strong enough.

**Status:** `DEEP_AUDIT`.

## D3. Memory Dial

Useful controlled memorization-pressure knob. No sufficiently large next scientific question identified yet.

**Status:** `RESOURCE`.

## D4. Knowledge Entropy Decay

Excellent learning-dynamics design reference with strong artifact, but the seed already spans phenomenon → interpretation → intervention.

**Status:** `DESIGN_REFERENCE`.

## D5. ImplicitMemBench

Interesting cognitive-memory inspiration, but construct validity remains unresolved.

**Status:** `WATCH / INSPIRATION_ONLY`.

---

# E. INTERNAL TERMINAL OBJECTS — do not accidentally resurrect

## E1. SemTrace / Topic 21

**Terminal local result:** `STOP_UPSTREAM_SEED_NOT_REPRODUCED`.

```text
edge mean = 0.000625  (required >=0.30)
edge-to-middle drop = 0.000625  (required >=0.20)
```

Custom mechanism G0 was never run.

**Status:** `ARCHIVED / DO_NOT_RELIST_AS_EXECUTABLE`.

## E2. Temporal Forgetting / Topic 05

The broad storage-vs-access question remains scientifically legitimate, but our registered Topic 05 failed conceptual identification because prefix rescue changes the task.

A future revisit requires a genuinely new identification strategy.

**Status:** `INTERNAL_COLLISION / TOPIC_05_ARCHIVED / NEW_IDENTIFICATION_REQUIRED`.

## E3. Temporal Spacing / Topic 13

Repetition damage reproduced in 4/4 locked trials, but `clustered-even` changed sign:

```text
-0.001534
+0.010758
+0.001005
-0.009134
```

Final verdict: `NO_EVIDENCE_SPACING_IN_LOCKED_TEST`.

**Status:** `ARCHIVED / DO_NOT_REOPEN_WITH_SCHEDULE_SEARCH`.

---

# F. Search-log kills / downgrades

These remain historical search decisions. External collision kills only when the closest work leaves insufficient independent paper narrative—not merely because neighbors exist.

| Candidate | Current reason not to promote |
|---|---|
| Round-09 thinking helps/hurts context use | advisor-low-priority reasoning object + closest work compresses the title-level story |
| source-level repetition → generalized source trust | frozen local G0 failed (`-1.319 pp`, CI crosses 0) |
| SMI residual → semantic/fan interference | closest literature occupies the main scientific question |
| spaced repetition for CPT | direct method/story overlap plus internal Topic 13 negative on our spacing explanation |
| testing effect / retrieval practice for CPT | direct quiz/test-enhanced CPT work occupies the main method story |
| Incomplete Learning follow-up | seed itself consumes major causal decomposition/intervention space |
| generic Agentic-RL feedback internalization | advisor fit low + direct recent scientific overlap |

---

# G. Current queue

There is no Topic 25.

```text
Existing numbered work:
1. Topic 22 MedEinst
   -> run frozen-output G0b-v3 canonicalization repair
   -> if CoT passes, run direct G0c
   -> only then consider mechanism

Advisor topic search:
1. Positional Imprinting
   -> exact official receipt
   -> if reproduced, within-model fact-level mother G0

2. PK/ICK arbitration history
   -> science-top hold until trustworthy official artifact is accessible

3. Encoding Specificity
   -> artifact + construct audit

4. ChronoScope
   -> secondary executable object

5. continue broad search
```

The search target is **not zero collision**. It is:

```text
mature literature
+ a distinct scientific object
+ enough remaining ACL/EMNLP/NAACL narrative
+ exact reproducible experimental handle
+ no contradiction with our own real scientific/identification failures
```
