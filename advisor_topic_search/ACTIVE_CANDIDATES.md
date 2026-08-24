# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 是历史搜索记录；一旦和后来的本地实验冲突，必须以 numbered topic 的 `G0_RESULTS.md` / `ARCHIVE_SUMMARY.md` 为准。

Last updated: 2026-08-24
Source of current ranking: `ROUND_13_2026-08-24.md`

---

# 0. Hard rules after internal-history reconciliation

1. **External overlap is normal.** 2026 年 AI/NLP 不追求“完全没人做过”。Collision audit 问的是：最接近工作都成立以后，是否仍剩下 ACL / EMNLP / NAACL 篇幅的独立主问题、主实验、主结论和后续机制/方法空间。
2. **Internal failure is evidence, not literature overlap.** 已经在本仓库跑过并停止的 hypothesis / identification / measurement route，不能因为新一轮搜题忘记结果而重新升为 active。
3. Status precedence:

```text
numbered-topic local result / ARCHIVE_SUMMARY
>
numbered-topic README
>
ACTIVE_CANDIDATES
>
ROUND search logs
```

4. Before promoting any search candidate, audit `FAILURES_AND_LESSONS.md` and numbered-topic directories for internal collision.
5. `REPRODUCTION_RECEIPT_POLICY.md` and `MOTHER_TOPIC_BRANCHING_POLICY.md` remain mandatory.
6. Generic reasoning / CoT / test-time-compute mechanism remains advisor-low-priority.
7. **Measurement repair has a budget.** One principled repair of a demonstrated defect can be legitimate. If the frozen route remains catastrophically invalid after that repair, archive the route rather than tune parser/prompt/extraction repeatedly.

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

**Important G0 repair before implementation:** prefer within-model fact-level counterbalancing after receipt, so early-P1 and early-P5 fact groups share the same optimizer/LR trajectory. Do not interpret three independently trained trajectories as positional imprinting if generic SGD path dependence remains an alternative.

Frozen prerequisite: `g0/POSITIONAL_IMPRINTING_RECEIPT.md`.

**Status:** `TOP_EXECUTABLE / ARTIFACT_VERIFIED / RECEIPT_PENDING / DO_NOT_REGISTER_YET`.

---

# B. ACTIVE HOLD / SECONDARY SEARCH OBJECTS

## B1. Parametric Encoding Specificity Across Input Structures

**Seed:** ACL 2026 Findings, SParK-Eval.

**Question:**

> When structured-data training looks like poor knowledge acquisition under ordinary QA, how much of the loss is true storage failure versus format-bound access?

The paper-sized object is not `try a table-like prompt`. It is whether parametric knowledge becomes format-invariant after acquisition, measured through a controlled encoding × retrieval-format matrix plus development/intervention/generalization.

Neighboring format-diversity work consumes part of the intervention story but does not automatically remove the mother question.

**Blockers:** Findings seed, artifact not yet complete/verified, construct must avoid prompt-engineering interpretation.

**Status:** `ACTIVE_HOLD / ARTIFACT+CONSTRUCT_GATE`.

## B2. ChronoScope — Temporal Scope Dynamics

**Seed:** ACL 2026 Main, *Evaluating Temporal Consistency in Multi-Turn Language Models*.

Potential paper object:

> How does an established historical reference-time state decay, suffer interference, recover under reinstatement, and compete with the present-day default?

This is broader than one `represented vs used` probe and can support decay/interference/reinstatement/default-attractor experiments.

Existing screen: `g0/chronoscope_drift_g0.py`.

**Status:** `ACTIVE_B / EXECUTABLE / ADVISOR_FIT_BELOW_A`.

---

# C. DEEP AUDIT / RESOURCE — not active top pool

## C1. Context-shaped truth geometry → source arbitration

External neighbors do not kill it merely because truth-vector / knowledge-conflict work is crowded. Promotion requires a full independent story linking conflict behavior, geometry, causal source choice, and intervention—not another geometry plot.

**Status:** `DEEP_AUDIT`.

## C2. Table DRE

Could still support a main-conference mechanism paper if a clean, dominant referencing/binding bottleneck and causal rescue are established. Current mother framing is not yet strong enough.

**Status:** `DEEP_AUDIT`.

## C3. Memory Dial

Useful controlled memorization-pressure knob. No sufficiently large next scientific question identified yet.

**Status:** `RESOURCE`.

## C4. Knowledge Entropy Decay

Excellent learning-dynamics design reference with strong artifact, but the seed already spans phenomenon → interpretation → intervention. Do not force a tiny adjacent gap.

**Status:** `DESIGN_REFERENCE`.

## C5. ImplicitMemBench

Interesting cognitive-memory inspiration, but construct validity of mapping prompt-induced behavior to human-style implicit memory remains unresolved.

**Status:** `WATCH / INSPIRATION_ONLY`.

---

# D. INTERNAL TERMINAL OBJECTS — do not accidentally resurrect

## D1. MedEinst / Topic 22

**Terminal local route result:** `MEASUREMENT_RUNTIME_FAILURE / NO_SCIENTIFIC_VERDICT`.

Repository truth:

- G0a pair structure passed on all 5,383 released pairs;
- the first Qwen3-14B CoT run was measurement-invalid (`81.25%` invalid);
- one principled measurement repair was frozen and rerun on the same 256 pairs/model/seed;
- repaired substantive gates all passed on the resolvable subset;
- invalid-output rate remained `160/256 = 62.5%` against the frozen `<=10%` gate;
- all thinking traces closed and none hit the 32,768-token ceiling;
- dominant failure was `unresolved_final`;
- direct mode was not run because G0b was a frozen prerequisite.

The scientific question `encoding failure vs update failure` is not falsified. The **local CoT measurement route is archived** because another parser/prompt/extraction repair after seeing this result would become measurement tuning.

See:

- `../22_medeinst_evidence_update/G0_RESULTS.md`
- `../22_medeinst_evidence_update/ARCHIVE_SUMMARY.md`

**Status:** `ARCHIVED / DO_NOT_REPAIR_AGAIN_LOCALLY`.

## D2. SemTrace / Topic 21

**Terminal local result:** `STOP_UPSTREAM_SEED_NOT_REPRODUCED`.

The exact official frozen run completed on `Qwen/Qwen2.5-Coder-7B-Instruct`, but semantic accuracy was essentially zero at all positions:

```text
edge mean = 0.000625  (required >= 0.30)
edge-to-middle drop = 0.000625  (required >= 0.20)
```

The custom paired mechanism G0 was never run. This is a prerequisite/platform failure for our selected Topic 21 regime, not a general refutation of the paper.

**Status:** `ARCHIVED / DO_NOT_RELIST_AS_EXECUTABLE`.

## D3. Temporal Forgetting / Topic 05

The broad storage-vs-access question is scientifically legitimate, but our registered Topic 05 failed **conceptual identification**: prefix rescue changes the task and cannot distinguish retained uncued competence from task simplification, search-space reduction, or conditional continuation.

A future revisit requires a genuinely new identification strategy, explicitly explaining why Topic 05's failure no longer applies.

**Status:** `INTERNAL_COLLISION / TOPIC_05_ARCHIVED / NEW_IDENTIFICATION_REQUIRED`.

## D4. Temporal Spacing / Topic 13

The locked four-trial test reproduced repetition damage in 4/4 trials, but `clustered-even` changed sign:

```text
-0.001534
+0.010758
+0.001005
-0.009134
```

Final verdict: `NO_EVIDENCE_SPACING_IN_LOCKED_TEST`.

This is a substantive negative for the registered spacing explanation because the prerequisite repetition damage was present. Do not revive by searching schedules/models/repeated pools.

**Status:** `ARCHIVED / DO_NOT_REOPEN_WITH_SCHEDULE_SEARCH`.

---

# E. Search-log kills / downgrades

These remain historical search decisions, but remember the external-collision rule: a topic is killed by literature only when the closest work leaves insufficient independent paper narrative—not merely because neighbors exist.

| Candidate | Current reason not to promote |
|---|---|
| Round-09 thinking helps/hurts context use | advisor-low-priority reasoning object + closest work compresses the title-level story |
| source-level repetition → generalized source trust | frozen local G0 failed (`-1.319 pp`, CI crosses 0) |
| SMI residual → semantic/fan interference | closest literature already occupies the main scientific question rather than merely sharing the domain |
| spaced repetition for CPT | direct method/story overlap plus internal Topic 13 negative on our spacing explanation |
| testing effect / retrieval practice for CPT | direct quiz/test-enhanced CPT work occupies the main method story |
| Incomplete Learning follow-up | seed itself consumes the major causal decomposition and interventions |
| generic Agentic-RL feedback internalization | advisor fit low + direct recent scientific overlap |

---

# F. Current queue

There is no Topic 25.

```text
1. Positional Imprinting
   -> exact official receipt
   -> if reproduced, redesign mother G0 with within-model fact-level counterbalancing
   -> only register after mother phenomenon survives

2. PK/ICK arbitration history
   -> remain science-top hold until trustworthy official artifact is accessible

3. Encoding Specificity
   -> artifact + construct audit

4. ChronoScope
   -> secondary executable object, advisor fit below learning/memory acquisition topics

5. continue broad search
```

The search target is **not zero collision**. It is:

```text
mature literature
+ a distinct scientific object
+ enough remaining ACL/EMNLP/NAACL narrative
+ exact reproducible experimental handle
+ no contradiction with our own archived results
```