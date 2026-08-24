# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 保存搜索历史与完整审计；本文件只回答：**现在到底哪些题还活着，下一步做什么？**
>
> 已经 KILL 的题不能因为换模型、换数据、换 probe 偷偷复活；若复活必须写明新的 scientific reason。

Last updated: 2026-08-24
Source of current ranking: `ROUND_11_2026-08-24.md`

---

# 0. Current advisor-fit override

Generic reasoning / CoT / test-time compute / reasoning RL / multi-hop reasoning mechanism remains **advisor-low-priority**, even if the paper-scale science is otherwise strong.

Preferred advisor-facing objects are:

- learning dynamics / developmental trajectories;
- knowledge acquisition and retention;
- memory / interference / storage-vs-access distinctions;
- representation and internal organization after a behavioral object exists;
- semantic organization / model development;
- old learning- or memory-science questions made newly testable by checkpoints and causal intervention.

Two hard policies remain active:

1. `REPRODUCTION_RECEIPT_POLICY.md`: no numbered registration before exact local reproduction;
2. `MOTHER_TOPIC_BRANCHING_POLICY.md`: after reproduction, require several independent scientific branches rather than one fragile causal arrow.

---

# A. CURRENT TOP — artifact verified, receipt next

## A1. Positional Imprinting of Parametric Knowledge

**Seed:** NAACL 2025 Main Long Paper / Oral, *Where is the answer? An empirical study of positional bias for parametric knowledge extraction in language model*.

**Seed phenomenon:** moving the same answer-bearing sentence to later positions in training documents strongly reduces later closed-book extraction under vanilla autoregressive training; official code/data-generation/training/evaluation artifact is public.

**New research question:**

> **If each fact is eventually exposed exactly the same number of times at early and late document positions, with the same total budget and an identical final training tail, can the position at which it was first/early encoded still leave a persistent effect on final extractability?**

This is **not** `position matters` and not generic `training order matters`. The matched object is acquisition-context history after aggregate position exposure has been equalized.

Canonical matched schedule shape:

```text
EARLY-FIRST: 1,1,5,5 + 1,5,1,5
LATE-FIRST:  5,5,1,1 + 1,5,1,5
INTERLEAVED: 1,5,1,5 + 1,5,1,5
```

All conditions share the same P1/P5 multiset and the same recent `1,5,1,5` washout tail.

**Why currently #1:**

- eligible NAACL Main seed;
- advisor-aligned learning / memory-development object, not reasoning;
- surprising seed is already externally established;
- complete official artifact verified at `omron-sinicx/WhereIsTheAnswer`;
- upstream commit frozen: `910fcddec93f7400b58257d70abf1dab31f1e179`;
- same facts / same position multiset / same tail makes the new scientific variable unusually clean;
- exact-collision audit found neighbors on generic training order and training-order representation, but no matched endpoint test of **position-of-acquisition history**;
- mother branches are genuinely independent: washout law, storage-vs-access, developmental window, representation, intervention, domain/model generalization.

**Important collision boundary:**

- seed already owns static positional bias and shuffle/D-AR mitigation;
- ACL 2024 PIT already shows QA-vs-document training order affects knowledge acquisition;
- *Fresh in Memory* already shows training-order recency is encoded in activations.

Therefore novelty survives only as the strict matched positional-history endpoint question above. Do not sell `LLMs remember training order` as new.

**Important engineering constraint:** official training inherits Hugging Face `Trainer`; JSONL order alone does not guarantee optimizer exposure order. Any history experiment must explicitly freeze the train sampler / phase schedule while preserving optimizer state.

**Frozen receipt card:** `advisor_topic_search/g0/POSITIONAL_IMPRINTING_RECEIPT.md`.

**Next step:** exact official seed reproduction only. No history code or mechanism work before receipt.

**Status:** `#1 / ARTIFACT_VERIFIED / RECEIPT_PENDING / DO_NOT_REGISTER_YET`.

---

# B. STRONG SCIENCE — blocked / hold

## B1. Does Knowledge Arbitration Have a Training History?

**Seed:** ACL 2026 Main, *How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models*.

**Research question:**

> **If two models consume exactly the same training examples and end with exactly the same empirical data distribution, can different histories of when reliable vs conflicting evidence appeared leave them with persistently different parametric-vs-context knowledge-use policies?**

This remains a strong path-dependence / hysteresis question about learning.

**Round-11 correction:** a web trace pointed to an apparent repository named `Training-Dynmaics-of-PK-ICK`, but the exact GitHub object is currently inaccessible / 404 and cannot satisfy the receipt policy.

Generic `training order matters` is also less novel than first assessed because ACL 2024 PIT and newer early-exposure work already establish broad order/history effects. The candidate survives only as **fixed-multiset hysteresis of a learned knowledge-arbitration policy**.

**Status:** `SCIENCE_TOP / HOLD_FOR_ACCESSIBLE_OFFICIAL_ARTIFACT`.

## B2. Parametric Encoding Specificity Across Input Structures

**Seed:** ACL 2026 Findings, *SParK-Eval: Evaluating Structure-Aware Knowledge Acquisition in LLMs for Domain Adaptation to Industrial Records*.

**Research question:**

> **When a fact trained from a table/list fails natural-language QA, was it truly not stored, or is access cue-dependent on the structure in which it was encoded?**

Alive because it instantiates a clean storage-vs-access / encoding-specificity distinction.

Risks remain:

- Findings rather than ACL Main;
- no complete official reproduction package verified;
- formatting-generalization work already narrows novelty.

Novelty survives only as **cue-dependent recovery of already-trained structured facts**, not `format matters` or `format augmentation helps`.

**Status:** `SCIENCE_HOLD / VENUE+ARTIFACT+COLLISION_GATE`.

---

# C. PREVIOUS EXECUTABLE OBJECTS — retained, but require advisor-fit review

## C1. SemTrace

Seed: ACL 2026 Main, *Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding*.

Existing numbered object: Topic 21.

**Status:** `EXECUTABLE / ADVISOR-FIT_REVIEW_REQUIRED`.

Reason: concrete artifact-complete phenomenon, but downstream framing can drift into generic reasoning/computation mechanism.

## C2. ChronoScope

Seed: ACL 2026 Main, *Evaluating Temporal Consistency in Multi-Turn Language Models*.

Existing G0: `advisor_topic_search/g0/chronoscope_drift_g0.py`.

**Status:** `EXECUTABLE / MEMORY-DISCOURSE OBJECT / ADVISOR-FIT_REVIEW`.

## C3. MedEinst

Seed: ACL 2026 Main, *MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis*.

Existing G0: `advisor_topic_search/g0/medeinst_pair_structure.py`.

**Status:** `EXECUTABLE / OLD-COGNITIVE-QUESTION / ADVISOR-FIT_REVIEW`.

Main unresolved gate: whether edits are local and critical cells dense enough without expert-control proliferation.

---

# D. HOLD / WATCH / RESOURCE REFERENCES

## D1. Memory Dial

ACL 2026 Findings with accessible official code. Useful as a controlled memorization-pressure resource.

Potential `same immediate recall, different later stability` question is narrowed by recent early-exposure/retention work.

**Status:** `WATCH AS RESOURCE / NOT TOP`.

## D2. Knowledge Entropy Decay

ICLR 2025 Oral with strong OLMo-based artifact and excellent learning-dynamics alignment.

Seed already links entropy decay to reduced acquisition/retention and includes a resuscitation intervention, consuming the obvious one-step follow-up space.

**Status:** `STRONG DESIGN REFERENCE / DO NOT FORCE A GAP`.

## D3. In-context representation deployment bottleneck

Seed: ACL 2026 Main, *Language Models Struggle to Use Representations Learned In-Context*.

**Status:** `HOLD_FOR_ARTIFACT / LOWER_AFTER_ADVISOR_RESET`.

## D4. Table DRE

Potentially exact and structured, but generic localization/binding literature is crowded.

**Status:** `DEEP_AUDIT / ARTIFACT_AND_COLLISION_GATE`.

## D5. Context-shaped truth geometry -> source choice

Knowledge-conflict/truth-vector space is crowded; artifact not verified.

**Status:** `DEEP_AUDIT / HOLD_FOR_ARTIFACT`.

## D6. Temporal Forgetting — storage loss vs access loss

Checkpoint-rich and scientifically attractive, but prior storage-vs-access mechanism design was assumption-heavy.

**Status:** `WATCH / DO_NOT_RESCUE_WITH_NEW_PROBE`.

## D7. ImplicitMemBench

Classic memory framing is attractive, but benchmark does not yet provide one clean storage/access mother contradiction.

**Status:** `WATCH / OLD-QUESTION_REFERENCE`.

---

# E. DOWNGRADED / KILLED — do not recycle

| Candidate | Verdict | Main reason |
|---|---|---|
| Round-09 thinking helps/hurts context use | `DOWNGRADE / DO_NOT_REGISTER` | advisor-low-priority reasoning object + RecaLLM/Lost-in-Thought near-exact collision |
| source-level repetition -> generalized source trust | `KILL` | frozen G0: mean transfer `-1.319 pp`, CI crosses 0 |
| SMI residual -> semantic/fan interference | `KILL AS NEW TOPIC` | CoNLL 2024 already studies LLM fan effect |
| OAKS repeated revision -> proactive interference | `KILL AS NEW TOPIC` | direct 2026 proactive/retroactive-interference literature + less clean parametric fit |
| spaced repetition for CPT | `KILL AS NEW TOPIC` | 2026 *When to Review* directly applies spaced repetition to continual pretraining |
| testing effect / retrieval practice for CPT | `KILL AS NEW TOPIC` | 2026 TELLME directly applies quiz/test-enhanced learning to CPT |
| annotation-entropy LoRA dynamics as main seed | `NOT ELIGIBLE TOP SEED` | ACL Student Research Workshop, outside current main-seed venue policy |
| Incomplete Learning follow-up | `KILL AS FOLLOW-UP` | seed already decomposes major causes and interventions |
| generic Agentic-RL feedback internalization | `KILL` | direct 2026 collisions + prior internal kill |
| Mem2Act recall->action gap | `KILL` | seed-owned + internal structural collision |
| BOULDER generic multi-turn degradation | `KILL` | crowded lost-in-conversation / intent-mismatch space |
| code->CoT training-order advantage | `KILL` | too many coupled explanations |
| AR-Bench generic information-gain mechanism | `KILL` | directly studied |
| Reasoning Trap tool hallucination follow-up | `KILL` | seed already provides mechanism |
| MathIF reasoning-loses-control | `KILL / WATCH ONLY` | crowded reasoning trajectory space |
| instruction tuning -> misinformation | `KILL` | base->instruct changes too many factors |
| RFC-Bench reference-free misinformation | `KILL` | conditions differ in available information |
| fact mutability -> source routing | `KILL` | relation-family confound |
| general parametric-vs-context reconciliation | `REFERENCE ONLY` | adjacent work already traces entity flow/intervention |
| LAD / MP-STRUCT | `KILL` | vocabulary/entropy entanglement; control tree grows |
| reversal-curse semantics follow-up | `KILL` | seed already owns core explanation |
| personalization factuality mechanism | `KILL` | seed already has representational account + steering |
| new-knowledge hallucination mechanism | `KILL` | seed already has mechanism + mitigation |
| general RAG context interference | `KILL / CROWDED` | heavily occupied |

---

# F. Current queue discipline

There is **still no Topic 25 registration**.

Current queue:

```text
1. A1 Positional Imprinting
   -> exact official reproduction receipt
   -> if REPRODUCED, freeze sampler/history G0
   -> only if mother G0 survives, register

2. B1 PK/ICK arbitration history
   -> wait for / locate trustworthy accessible official artifact

3. B2 encoding specificity
   -> only reopen if venue/artifact/collision gates improve

4. continue broad search for an advisor-fit object that beats A1 on science without losing artifact quality
```

The target shape remains:

```text
classic learning / memory question
+ externally anchored LLM phenomenon
+ exact public training/checkpoint object
+ clean falsifiable first experiment
-> reproduced mother phenomenon
-> independent behavior / development / mechanism / intervention branches
```

Do not promote a topic merely because code is complete. Do not promote a fashionable reasoning object because the result looks impressive. Do not turn a failed mother G0 into model/layer/prompt shopping.
