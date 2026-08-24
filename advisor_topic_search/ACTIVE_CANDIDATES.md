# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 保存搜索历史与完整审计；本文件只回答：**现在到底哪些题还活着，下一步做什么？**
>
> 已经 KILL 的题不能因为换模型、换数据、换 probe 偷偷复活；若复活必须写明新的 scientific reason。

Last updated: 2026-08-24
Source of current ranking: `ROUND_10_2026-08-24.md`

---

# 0. Current advisor-fit override

Generic reasoning / CoT / test-time compute / reasoning RL / multi-hop reasoning mechanism is now **advisor-low-priority**, even if the paper-scale science is otherwise strong.

Preferred advisor-facing objects are:

- learning dynamics / developmental trajectories;
- knowledge acquisition and retention;
- memory / interference / storage-vs-access distinctions;
- representation and internal organization after a behavioral object exists;
- semantic organization / model development;
- old learning- or memory-science questions made testable by checkpoints and causal intervention.

Mother-topic policy remains mandatory after reproduction receipt.

---

# A. CURRENT TOP SCIENCE — not yet executable

## A1. Does Knowledge Arbitration Have a Training History?

**Seed:** ACL 2026 Main, *How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models*.

**Research question:**

> **If two models consume exactly the same training examples and end with exactly the same empirical data distribution, can different histories of when reliable vs conflicting evidence appeared leave them with persistently different parametric-vs-context knowledge-use policies?**

This is a path-dependence / hysteresis question about learning, not a reasoning benchmark question.

**Why it is currently #1:**

- ACL Main seed;
- seed already establishes a surprising, controlled learning object: balanced PK/ICK arbitration requires repetition + moderate inconsistency + a skewed knowledge distribution;
- exact next question is different from the seed's static-data-statistics question;
- same exact training multiset can be held fixed while only temporal organization changes;
- automatic synthetic labels, no paid API, no annotation;
- strong mother-topic branching: behavioral hysteresis, critical window/development, mechanism, schedule intervention, continual-pretraining/post-training generalization;
- no exact 2025–2026 collision found for fixed-multiset history dependence of this particular learned policy.

**Blocking issue:** no trustworthy accessible official reproduction repository was verified. Under the receipt-first policy, do not reimplement from the paper merely to force promotion.

**Status:** `SCIENCE_TOP / #1 NON-REASONING / HOLD_FOR_OFFICIAL_ARTIFACT`.

---

## A2. Parametric Encoding Specificity Across Input Structures

**Seed:** ACL 2026 Findings, *SParK-Eval: Evaluating Structure-Aware Knowledge Acquisition in LLMs for Domain Adaptation to Industrial Records*.

**Research question:**

> **When a fact trained from a table/list fails natural-language QA, was it truly not stored, or is access cue-dependent on the structure in which it was encoded?**

**Why alive:**

- natural old-memory distinction: storage vs access / encoding specificity;
- same facts can be crossed with encoding format × retrieval-cue format;
- positive or null result is interpretable;
- branches into development, representation, mixed-format intervention, and domain generalization.

**Risks:**

- Findings rather than ACL Main;
- no complete official reproduction package verified;
- EMNLP 2025 formatting-generalization work already shows that format-diverse training improves QA extraction, narrowing novelty.

Novelty survives only if the object is **cue-dependent recovery of already-trained structured facts**, not `format matters` or `format augmentation helps`.

**Status:** `SCIENCE_HOLD / #2 / VENUE+ARTIFACT+COLLISION_GATE`.

---

# B. PREVIOUS ACTIVE OBJECTS — retained, but ranking is no longer authoritative

These objects predate the new advisor-fit reset. They are not deleted because their exact prerequisites/G0s remain useful, but they no longer outrank A1 merely because code already exists.

## B1. SemTrace — lexical access survives while long-context semantic execution fails

Seed: ACL 2026 Main, *Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding*.

Existing object: same program / same model; edge semantic success, lexical success at edge+middle, parseable middle semantic failure.

Existing implementation: numbered Topic 21 plus upstream receipt/G0 scripts.

**Current status:** `EXECUTABLE / ADVISOR-FIT_REVIEW_REQUIRED`.

Reason: the phenomenon is concrete and artifact-complete, but the downstream framing can easily drift into generic reasoning/computation mechanism. Do not automatically keep it #1 under the new advisor constraint.

## B2. ChronoScope — historical reference-time drift

Seed: ACL 2026 Main, *Evaluating Temporal Consistency in Multi-Turn Language Models*.

Object: under Gold Context, a chain begins correct under an explicit historical scope and later substitutes the valid present-day answer under implicit scope.

Existing G0: `advisor_topic_search/g0/chronoscope_drift_g0.py`.

**Current status:** `EXECUTABLE / MEMORY-DISCOURSE OBJECT / ADVISOR-FIT_REVIEW`.

This is less exposed to the generic-reasoning objection than SemTrace, but it still requires the frozen prerequisite cell before mechanism work.

## B3. MedEinst — failed update under decisive counterevidence

Seed: ACL 2026 Main, *MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis*.

Object: classic mental-set / belief-update question with control–trap pairs.

Existing G0: `advisor_topic_search/g0/medeinst_pair_structure.py`.

**Current status:** `EXECUTABLE / OLD-COGNITIVE-QUESTION / ADVISOR-FIT_REVIEW`.

The main unresolved gate remains whether edits are local enough and critical cells dense enough without medical expert annotation/control proliferation.

---

# C. HOLD / WATCH

## C1. In-context representation deployment bottleneck

Seed: ACL 2026 Main, *Language Models Struggle to Use Representations Learned In-Context*.

Strong science, but no trustworthy complete reproduction package was found and the object can drift toward generic `represented != used` mechanism work.

**Status:** `HOLD_FOR_ARTIFACT / LOWER_AFTER_ADVISOR_RESET`.

## C2. Table DRE — structural understanding vs value referencing

Potentially exact and structured, but generic localization/binding literature is crowded.

**Status:** `DEEP_AUDIT / ARTIFACT_AND_COLLISION_GATE`.

## C3. Context-shaped truth geometry → source choice

Knowledge-conflict/truth-vector space is crowded; artifact not verified.

**Status:** `DEEP_AUDIT / HOLD_FOR_ARTIFACT`.

## C4. Temporal Forgetting — storage loss vs access loss

Checkpoint-rich and scientifically attractive, but the prior storage-vs-access mechanism design was assumption-heavy and cross-checkpoint alignment is not itself a scientific answer.

**Status:** `WATCH / DO_NOT_RESCUE_WITH_NEW_PROBE`.

## C5. ImplicitMemBench — non-declarative memory

Classic memory framing is attractive, but current benchmark does not yet provide one clean storage/access mother contradiction.

**Status:** `WATCH / OLD-QUESTION_REFERENCE`.

---

# D. DOWNGRADED / KILLED — do not recycle

| Candidate | Verdict | Main reason |
|---|---|---|
| Round-09 thinking helps/hurts context use | `DOWNGRADE / DO_NOT_REGISTER` | advisor-low-priority reasoning object + RecaLLM/Lost-in-Thought near-exact sign-structure collision |
| source-level repetition → generalized source trust | `KILL` | frozen G0: mean transfer `-1.319 pp`, CI crosses 0, failed all paper-scale gates |
| SMI residual → semantic/fan interference | `KILL AS NEW TOPIC` | CoNLL 2024 already studies LLM fan effect, including pretraining-induced fan |
| Incomplete Learning follow-up | `KILL AS FOLLOW-UP` | seed already decomposes major causes and interventions |
| generic Agentic-RL feedback internalization | `KILL` | direct 2026 collisions + prior internal kill |
| Mem2Act recall→action gap | `KILL` | seed-owned + internal structural collision |
| BOULDER generic multi-turn degradation | `KILL` | crowded lost-in-conversation / intent-mismatch space |
| code→CoT training-order advantage | `KILL` | too many coupled explanations |
| AR-Bench generic information-gain mechanism | `KILL` | directly studied |
| Reasoning Trap tool hallucination follow-up | `KILL` | seed already provides mechanism |
| MathIF reasoning-loses-control | `KILL / WATCH ONLY` | crowded reasoning trajectory space |
| Instruction tuning → misinformation | `KILL` | base→instruct changes too many factors |
| RFC-Bench reference-free misinformation | `KILL` | conditions differ in available information |
| Fact mutability → source routing | `KILL` | relation-family confound |
| general parametric-vs-context reconciliation | `REFERENCE ONLY` | adjacent work already traces entity flow/intervention |
| LAD / MP-STRUCT | `KILL` | vocabulary/entropy entanglement; control tree grows |
| reversal-curse semantics follow-up | `KILL` | seed already owns core explanation |
| personalization factuality mechanism | `KILL` | seed already has representational account + steering |
| new-knowledge hallucination mechanism | `KILL` | seed already has mechanism + mitigation |
| general RAG context interference | `KILL / CROWDED` | heavily occupied |

---

# E. Current queue discipline

There is **no new numbered-topic registration from Round 10**.

Current scientific search priority:

```text
1. A1 PK/ICK training-history dependence — wait for / locate exact official artifact
2. continue searching for a different artifact-complete, non-reasoning mother phenomenon
3. A2 encoding specificity only if artifact and collision gates improve
4. previous executable objects require advisor-fit re-review before receiving large mechanism work
```

The current target shape is:

```text
classic learning / memory distinction
+ externally anchored LLM phenomenon
+ reproducible training/checkpoint object
-> mother phenomenon
-> independent behavioral / development / mechanism / intervention branches
```

Do not promote a topic merely because its code is complete, and do not promote a fashionable reasoning object merely because the result is interesting.