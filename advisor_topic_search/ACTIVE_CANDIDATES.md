# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 是历史搜索记录；一旦和后来的本地实验冲突，必须以 numbered-topic 的实际 G0/G1 结果为准。

Last updated: 2026-08-25
Source of current ranking: numbered-topic local results through Topic 26 + Round 08 repeated search / collision audit

---

# 0. Hard rules after internal-history reconciliation

1. **External overlap is normal.** Collision audit 问的是：最接近工作都成立以后，是否仍剩下 ACL / EMNLP / NAACL 篇幅的独立主问题、主实验、主结论和后续机制/方法空间。
2. **Internal scientific failure is evidence.** 已经在本仓库跑过并停止的 hypothesis / identification route，不能因为新一轮搜题忘记结果而重新升为 active。
3. **Measurement/artifact failure和scientific failure必须分开。** scorer/parser/runtime/metadata contract 失败不能自动判科学问题为假。
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
8. Measurement repair is defect-based but bounded; do not search aliases/mappers/prompts/orders/models/thresholds after a bounded repair still leaves support unhealthy.
9. **Artifact public != experiment executable.** Before registration, verify the exact instance-level metadata contract and count eligible support for the frozen first contrast.
10. **High prerequisite tax + low information gain is a hard downgrade.** Topic 25 is the canonical example.
11. **Mechanism is downstream, not novelty by itself.** Prefer `natural question -> cheap behavioral object -> locked confirmation -> representation/causal analysis`.
12. **Do not pad the pool.** Three strong topics are preferable to five where #4/#5 have materially worse collision, construct, or execution profiles.

Detailed policies:

- `COLLISION_AND_INTERNAL_HISTORY_POLICY.md`
- `REPRODUCTION_RECEIPT_POLICY.md`
- `MOTHER_TOPIC_BRANCHING_POLICY.md`

Round-08 rationale and fresh collision audit:

- `ROUND_08_2026-08-25.md`

---

# A. TOP SCIENCE / TOP EXECUTABLE SEARCH OBJECTS

## A1. Cue Competition / Kamin Blocking in LLM In-Context Associative Learning

**Seed:** NeurIPS 2025 Main, *Large Language Models as Model Organisms for Human Associative Learning*.

**Question:**

> When a new cue co-occurs with an outcome just as often as in a control condition, does an already-established predictor block the LLM from learning the new cue–outcome association?

Sharper scientific interpretation:

> **Is LLM in-context association driven by mere co-occurrence, or is it prediction-error / informativeness-sensitive?**

Frozen first-shape target:

```text
BLOCKING
Phase 1: A -> X
Phase 2: A+B -> X
Test:    B -> ?

MATCHED CONTROL
same B-X exposure count
but no previously established A -> X predictor
```

Advantages:

- classic learning-theory mother question;
- synthetic nonce cues/outcomes;
- automatic labels;
- no external API or human annotation;
- no expensive seed-reproduction prerequisite;
- positive and null G0 both scientifically meaningful;
- full-paper runway through unblocking, overshadowing, acquisition-vs-performance, then mechanism.

Important internal-history boundary: this is **not** the previously failed latent-inhibition topic. Latent inhibition pre-exposes the target cue without outcome; blocking first establishes a competing predictor of the same outcome.

Fresh Round-08 collision search found no direct LLM blocking/unblocking paper.

**Status:** `TOP_POOL / #1 / NEW / G0_DIRECT / LOW_PREREQUISITE`.

---

## A2. Information Non-Monotonicity under Progressive Truthful Clues

**Seed / artifact:** EMNLP 2024 Main, *Do great minds think alike? Investigating Human-AI Complementarity in Question Answering with CAIMIRA* + released ProtoBowl progressive-clue / agent-response artifacts.

**Question:**

> If every new clue is truthful and relevant to the same answer, can strictly more information make an LLM abandon a correct answer for a wrong one?

Critical event:

```text
same question + same agent
clues 1..t      -> correct
clues 1..t+1    -> wrong
new clue         = truthful evidence for the same gold answer
```

First shot should use the released response trajectories only:

```text
wrong -> wrong
wrong -> correct
correct -> correct
correct -> wrong  <-- target
```

Advantages:

- released ~3.82k progressive questions and ~304k agent-response rows;
- first G0 can be artifact-only with no new inference;
- exact adjacent-transition object;
- natural evidence-integration question, not generic `more context hurts`;
- follow-up factors: competitor introduction, clue specificity/ambiguity, clue-order path dependence;
- mechanism only at a frozen correct->wrong boundary.

Fresh Round-08 collision search found generic `more evidence/context can hurt` work, but no direct paper centered on adjacent correct→wrong reversals under monotonically accumulating truthful QuizBowl clues.

**Status:** `TOP_POOL / #2 / ARTIFACT_FIRST / VERIFY_TRAJECTORY_SCHEMA`.

---

## A3. Positional Imprinting of Parametric Knowledge

**Seed:** NAACL 2025 Main / Oral, *Where is the answer? An empirical study of positional bias for parametric knowledge extraction in language model*.

**Question:**

> If facts ultimately receive identical early/late position exposure counts and the same later washout training, does the position in which a fact was initially acquired leave a persistent difference in final parametric accessibility?

Core distinction:

```text
final / cumulative position statistics
vs
historical acquisition path
```

Potential full story:

- persistent fact-level acquisition-history effect;
- washout law / reversibility;
- early-vs-late developmental window;
- replay/interleaving intervention;
- storage/access diagnostics only after the behavioral object stands;
- internal trace only after locked behavior;
- cross-model/domain generalization.

**Artifact:** complete official code/data path verified at `omron-sinicx/WhereIsTheAnswer`, frozen upstream commit `910fcddec93f7400b58257d70abf1dab31f1e179`.

**Important G0 requirement:** prefer within-model fact-level counterbalancing after receipt, so early-P1 and early-P5 fact groups share the same optimizer/LR trajectory and later exposure is exactly equalized.

Frozen prerequisite: `g0/POSITIONAL_IMPRINTING_RECEIPT.md`.

Fresh Round-08 search did not find direct work on **first-acquisition position after later position exposure is equalized**.

**Status:** `TOP_EXECUTABLE / #3 / ARTIFACT_VERIFIED / RECEIPT_PENDING / DO_NOT_REGISTER_YET`.

---

# B. ACTIVE HOLD / SECONDARY SEARCH OBJECTS

## B1. Does Knowledge Arbitration Have a Training History?

**Seed:** ACL 2026 Main, *How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models*.

**Question:**

> If two models consume the exact same training multiset and end with the same empirical data distribution, can different temporal histories of reliable vs conflicting evidence leave them with persistently different parametric-vs-context knowledge-use policies?

This remains scientifically strong, but no trustworthy accessible official reproduction artifact has been verified. Do not reverse-engineer the seed until a low-tax official path exists.

**Status:** `SCIENCE_TOP / HOLD_FOR_ACCESSIBLE_OFFICIAL_ARTIFACT`.

---

## B2. Parametric Encoding Specificity Across Input Structures

**Seed:** ACL 2026 Findings, SParK-Eval.

**Question:**

> When structured-data training looks like poor knowledge acquisition under ordinary QA, how much of the loss is true storage failure versus format-bound access?

Blockers: Findings seed, artifact not yet complete/verified, construct must avoid prompt-engineering interpretation.

**Status:** `ACTIVE_HOLD / ARTIFACT+CONSTRUCT_GATE`.

---

## B3. Belief Consistency vs Rigidity

ACL 2026 reports a real dissociation: code/math continued-pretraining families can improve belief consistency while logical failures increase.

Natural question:

> Does training make a model more consistent, or merely more rigid?

Not promoted because the current route depends too strongly on model-family attribution, nearby stubbornness/deference work is already active, and no equally clean low-tax mother experiment was found.

**Status:** `WATCH / HOLD`.

---

# C. DEEP AUDIT / RESOURCE — not active top pool

## C1. Context-shaped truth geometry -> source arbitration

Promotion requires a full independent story linking conflict behavior, geometry, causal source choice, and intervention—not another geometry plot.

**Status:** `DEEP_AUDIT`.

## C2. Table DRE

Could support a main-conference paper if a clean, dominant referencing/binding bottleneck and causal rescue are established. Current mother framing is not yet strong enough.

**Status:** `DEEP_AUDIT`.

## C3. Memory Dial

Useful controlled memorization-pressure knob. No sufficiently large next scientific question identified yet.

**Status:** `RESOURCE`.

## C4. Knowledge Entropy Decay

Excellent learning-dynamics design reference with strong artifact, but the seed already spans phenomenon -> interpretation -> intervention.

**Status:** `DESIGN_REFERENCE`.

## C5. ImplicitMemBench

Interesting cognitive-memory inspiration and a useful conditioning resource, but construct validity remains unresolved. Round 08 deliberately did not promote its inhibition/preference asymmetry because blocking has the cleaner learning-theory object and the final pool should not over-concentrate on one memory family.

**Status:** `WATCH / INSPIRATION_ONLY`.

---

# D. INTERNAL TERMINAL / PAUSED ROUTES — do not accidentally resurrect

## D1. Topic 25 — Reasoning × Context Use Boundary

**Final local result:** `SEED_RELATION_NOT_REPRODUCED`.

Frozen receipt:

```text
Qwen3-8B gold-only          = 0.41493
Qwen3-8B-Think gold-only    = 0.45746
Qwen3-8B noisy pooled       = 0.31182
Qwen3-8B-Think noisy pooled = 0.35179
```

Required seed relation `thinking noisy >= thinking gold-only` was false. G0 was correctly not run.

**Status:** `TERMINAL / SEED_RELATION_NOT_REPRODUCED / G0_NOT_RUN`.

---

## D2. Topic 26 — Temporal Scope Interference & Reinstatement / ChronoScope

**Final local result:** `STOP_INSUFFICIENT_EXACT_SUPPORT`.

Frozen preflight:

```text
raw structural candidates = 324,637
eligible exact-support     = 0
selected                   = 0 / 512
```

The pinned official artifact contained zero turn-level `present_day_answer` fields across 3,335,698 turns. This is an artifact/measurement-contract stop, not evidence against the scientific phenomenon.

**Status:** `ARCHIVED_LOCAL_ROUTE / INSUFFICIENT_EXACT_SUPPORT / NO_SCIENTIFIC_VERDICT`.

---

## D3. MedEinst / Topic 22

**Final local route result:** `MEASUREMENT_CANONICALIZATION_FAILURE / NO_SCIENTIFIC_VERDICT`.

- pair locality passed on all 5,383 released pairs;
- substantive v3 gates passed on scorable support;
- pair invalid rate remained `32.42%`, above frozen `<=10%` ceiling;
- direct G0c and mechanism work were not run.

**Status:** `ARCHIVED_LOCAL_ROUTE / MEASUREMENT_CANONICALIZATION_FAILURE / NO_SCIENTIFIC_VERDICT`.

---

## D4. SemTrace / Topic 21

**Terminal local result:** `STOP_UPSTREAM_SEED_NOT_REPRODUCED`.

```text
edge mean = 0.000625  (required >=0.30)
edge-to-middle drop = 0.000625  (required >=0.20)
```

Custom mechanism G0 was never run.

**Status:** `ARCHIVED / DO_NOT_RELIST_AS_EXECUTABLE`.

---

## D5. Temporal Forgetting / Topic 05

The broad storage-vs-access question remains scientifically legitimate, but our registered Topic 05 failed conceptual identification because prefix rescue changes the task. The 2026 external literature has also become substantially more crowded on temporal forgetting and factual access failure.

**Status:** `INTERNAL_COLLISION / TOPIC_05_ARCHIVED / DO_NOT_REOPEN_WITH_GENERIC_STORAGE-VS-ACCESS`.

---

## D6. Temporal Spacing / Topic 13

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

# E. Search-log kills / downgrades

| Candidate | Current reason not to promote |
|---|---|
| OAKS historical-state intrusion / proactive interference | 2026 preprint `Transformers Remember First, Forget Last` (arXiv:2603.00270) directly reports proactive-interference primacy intrusion across 39 LLMs; mother question now collision-heavy |
| SMI contextual diversity vs repetition | `Facts in Stats` already manipulates contextual diversity + rebuilding corpus metric carries TB-scale prerequisite tax |
| generic belief rigidity / stubbornness | nearby direct challenge/deference literature + weak causal attribution from model-family comparisons |
| ImplicitMemBench inhibition/preference asymmetry | construct gate + same-family redundancy; blocking is cleaner |
| generic temporal forgetting / storage-vs-access after SFT | external 2026 literature now crowded + Topic 05 identification history |
| generic thinking helps/hurts context use | Topic 25 frozen seed relation failed |
| ChronoScope temporal interference/reinstatement route | Topic 26 exact instance-level metadata support = 0; no scientific verdict |
| source-level repetition -> generalized source trust | frozen local G0 failed (`-1.319 pp`, CI crosses 0) |
| SMI residual -> semantic/fan interference | closest literature occupies the main scientific question |
| spaced repetition for CPT | direct method/story overlap plus internal Topic 13 negative |
| testing effect / retrieval practice for CPT | direct quiz/test-enhanced CPT work occupies the main method story |
| Incomplete Learning follow-up | seed itself consumes major causal decomposition/intervention space |
| generic Agentic-RL feedback internalization | advisor fit low + direct recent scientific overlap |

---

# F. Current queue

```text
Existing numbered work:
1. Topic 24
   -> frozen physical-disturbance attribution G0

2. Topic 16
   -> evidence-provenance G0

Advisor topic search — Round 08 finalists:
1. Blocking / cue competition
   -> implement one fully counterbalanced synthetic G0
   -> freeze schedule before running models

2. Progressive truthful-clue reversal
   -> audit released response trajectory schema
   -> count exact eligible correct->wrong transitions before new inference
   -> freeze canonicalization and transition definition

3. Positional Imprinting
   -> finish exact official receipt
   -> if reproduced, metadata-contract preflight
   -> within-model fact-level mother G0

Holds:
4. PK/ICK arbitration history
   -> wait for trustworthy official artifact

5. Encoding Specificity
   -> artifact + construct + metadata-contract audit
```

The search target is **not zero collision**. It is:

```text
mature literature
+ a distinct scientific mother object
+ enough remaining ACL/EMNLP/NAACL narrative
+ exact reproducible experimental handle
+ verified instance-level metadata contract
+ low prerequisite tax
+ no contradiction with our own real scientific/identification/measurement-route failures
```
