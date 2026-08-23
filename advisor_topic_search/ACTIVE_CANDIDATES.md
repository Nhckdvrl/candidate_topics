# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 保存搜索历史与完整审计；本文件只回答：**现在到底哪些题还活着，下一步做什么？**
>
> 每轮结束后必须同步更新。已经 KILL 的题不能因为换模型、换数据、换 probe 偷偷复活；若复活必须写明新的 scientific reason。

Last updated: 2026-08-23
Source of current ranking: `ROUND_07_2026-08-23.md`

---

# A. ACTIVE — 可以直接进入 prerequisite / G0

## A1. SemTrace — lexical retrieval survives, semantic execution fails in the middle

**Seed**: ACL 2026 main, *Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding*.

**Research question**:

> **Why does context position selectively destroy operational state transition while lexical access to the same code remains available?**

**Why alive**:

- ACL main seed;
- same model / same synthetic program / move only relevant code position;
- huge reported semantic middle-position failure;
- lexical recall provides an intact control on the same content;
- official reproduction package complete;
- 7B/8B open models;
- intermediate program states are exactly computable;
- no manual annotation / paid API prerequisite.

**Frozen critical cell**:

```text
same program + same model
edge:   semantic correct
middle: semantic wrong
edge:   lexical correct
middle: lexical correct
```

**First G0**:

1. reproduce edge-vs-middle semantic drop on one seed-listed 7B/8B model;
2. verify lexical recall remains intact on the same instances;
3. measure `edge-correct / middle-wrong / lexical-intact` density;
4. only then localize exact intermediate program-state failure / causal patching.

**Kill line**:

- critical cell sparse;
- lexical and semantic failures collapse together;
- result reduces to generic position degradation;
- mechanism requires broad layer/token fishing.

**Collision boundary**: not generic state tracking, binding, or lost-in-the-middle. Novelty must be **selective operational-computation collapse under intact lexical access**.

**Status**: `SURVIVAL_TOP / #1 / CODE_NEXT`

---

## A2. ChronoScope — reference-time state loss or failure to use it?

**Seed**: ACL 2026 main, *Evaluating Temporal Consistency in Multi-Turn Language Models*.

Official repo: `yashkumaratri/ChronoScope`.

**Research question**:

> **When a model drifts from an established historical scope back to a present-day answer, has the conversational reference-time state been lost, or is it still represented but unable to override present-day parametric knowledge?**

**Why alive**:

- ACL main seed;
- >1.4M deterministically generated temporal chains;
- classic reference-time / discourse-state question;
- Gold Context removes previous-answer error propagation;
- official evaluator explicitly stores historical gold and `present_day_answer`;
- official `drift` error means wrong historical answer exactly matches valid present-day truth;
- open models directly supported: Qwen2.5-7B, Qwen3-4B, Llama-3.1-8B, Mistral-7B, Gemma-7B, GPT-OSS-20B, etc.;
- local HF inference and automatic scoring already implemented;
- no new annotation / closed API dependency.

**Frozen prerequisite cell**:

```text
Gold Context
same chain + same model
initial explicit temporal turn = correct
later implicit-scope turn = wrong
wrong answer = official present_day_answer
```

Existing screen:

```text
advisor_topic_search/g0/chronoscope_drift_g0.py
```

The screen consumes the official evaluator JSON and reports chain/turn drift density by family and position.

**Mechanism only after G0**:

1. freeze one model and one/two clean families, starting with carryover;
2. make a matched explicit-year-restatement version of the same follow-up;
3. compare explicit-correct vs implicit-drift temporal-scope representation;
4. bounded natural counterfactual activation patching;
5. test whether the intervention specifically changes present-day substitution back to historical gold.

**Kill line**:

- first-correct → later-present-drift sparse on accessible models;
- drift disappears in Gold Context;
- explicit year restatement does not rescue the same instances;
- probe only decodes lexical year tokens, not maintained implicit state;
- result reduces to generic context-vs-parametric conflict already explained elsewhere;
- rescue depends on layer/strength fishing.

**Collision boundary**: not generic temporal QA, generic multi-turn memory, or generic RAG conflict. The object is **implicit conversational reference-time state and its causal competition with the present-day prior**.

**Status**: `SURVIVAL_TOP / #2 / RUN_DRIFT_G0`

---

## A3. MedEinst — encoding failure or failed belief update under Einstellung?

**Seed**: ACL 2026 main, *MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis*.

**Research question**:

> **When decisive counterevidence is introduced, does the model fail to encode it, or is it encoded but unable to update the already-formed diagnostic state?**

**Why alive**:

- classic Einstellung / mental-set scientific question;
- 5,383 control–trap pairs;
- seed already defines the Bias-Trap critical failure;
- open models show substantial trap behavior;
- diagnosis labels automatic;
- no need to re-prove the existence of Einstellung.

**Mandatory prerequisite**:

Verify control→trap is genuinely local discriminative-evidence editing, not diffuse narrative rewrite.

Existing G0:

```text
advisor_topic_search/g0/medeinst_pair_structure.py
```

**Promotion condition**:

- edits local;
- control accuracy high enough;
- Bias-Trap critical cell dense;
- decisive evidence span can be located reproducibly without large expert annotation.

**Kill line**:

- edit diffuse;
- critical cell sparse;
- key evidence requires manual clinical judgment item-by-item;
- explanation starts requiring many clinical confound controls.

**Status**: `SURVIVAL_TOP / #3 / RUN_PAIR_LOCALITY_G0`

---

# B. HOLD / DEEP AUDIT — strong science, prerequisite/resource not cleared

## B1. In-context representation deployment bottleneck

**Seed**: ACL 2026 main, *Language Models Struggle to Use Representations Learned In-Context*.

**Research question**:

> **What deployment-specific routing/readout computation makes an already learned in-context representation inert under one interface but usable under another?**

Seed directly reports latent novel semantics despite poor downstream deployment, so the dissociation is not invented by us.

**Blocking issue**: no trustworthy complete official reproduction package found. Rebuilding generator + prompts + probe regime would reintroduce too much prerequisite risk.

**Promotion gate**:

1. official code/data appears; or
2. appendix enables exact untuned critical-cell reproduction on one open model.

**Status**: `SCIENCE_TOP / HOLD_FOR_ARTIFACT`

---

## B2. Table DRE — structural understanding vs value referencing

**Seed**: ACL 2026 main, *When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors*.

**Research question**:

> **When table structure is understood but the model cites the wrong value, is the failure localization, entity–value binding, or late readout substitution?**

Pros: structured/exact labels, programmatic perturbations, small/open models.

Main risk: generic entity/positional binding mechanism is already crowded. Need a dense `structure-correct / reference-wrong` cell that cannot be reduced to generic retrieval/binding.

**Status**: `DEEP_AUDIT / ARTIFACT_AND_COLLISION_GATE`

---

## B3. Context-shaped truth geometry → source choice

**Seed**: ACL 2026 main, *How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs*.

**Research question**:

> **Do context-induced transformations of truth representations causally determine whether context or parametric memory controls behavior under conflict?**

Seed already reports geometry changes; next step must establish specific source-selection causality, not another truth-vector plot/steer.

Risks: knowledge-conflict/truth-vector literature crowded; official reproduction artifact not verified.

**Status**: `DEEP_AUDIT / HOLD_FOR_ARTIFACT`

---

# C. WATCH — scientifically useful, currently not worth first bet

## C1. Belief consistency — stability vs correctness

**Seed**: ACL 2026 main, *Assessing Belief Consistency on the Logical Conversation Process*.

Interesting dissociation: some code/math post-training families improve belief consistency while logical failure increases.

Question:

> does training improve belief-state persistence, or merely increase stubborn commitment even when the state is wrong?

Why WATCH: no clean official artifact found, sampling-heavy evaluation, model-family comparison confounds, prompt sensitivity.

**Status**: `WATCH / STABILITY_VS_CORRECTNESS`

## C2. GSM-Infinite — computation cliff

ICML 2025 synthetic exact reasoning resource.

Potential question: around the complexity cliff, are correct intermediate node values still represented but failing to propagate through the next dependency edge?

Why WATCH: very executable, but generic state-tracking / propagation mechanism may not be scientifically distinct from SemTrace and adjacent work.

**Status**: `WATCH+ / BACKUP_RESOURCE`

## C3. Correct trace → wrong final answer

ACL 2026 reports very dense cells where an intermediate trace is correct but final answer is wrong in small open models.

Why WATCH: CoT faithfulness / hidden computation / reasonless-token space is crowded. Need sharper causal distinction than “why doesn’t correct CoT control answer?”

**Status**: `WATCH+ / COLLISION_HEAVY`

## C4. Emergent response planning — commitment vs rewrite

ICML 2025 shows pre-output hidden states predict future response properties.

Potential question:

> is the pre-output answer plan a commitment or a provisional state that is rewritten as reasoning unfolds?

Why WATCH: latent planning/steering follow-ups are moving fast; no exact paired plan-reversal object yet.

**Status**: `WATCH+ / NEED_EXACT_TRANSITION_OBJECT`

## C5. Illusions of Confidence — stable answer vs robust belief

Perfect self-consistency can coexist with severe vulnerability to social/authority interference.

Why WATCH: nice old metacognition distinction, but natural follow-up currently looks too much like robust-vs-brittle probe/steering rather than a forced computation bottleneck.

**Status**: `WATCH+`

## C6. Temporal Forgetting — storage loss vs access loss

Same problem can be correct at an earlier checkpoint and wrong later.

Why WATCH: cross-checkpoint basis changes and open-ended CoT path changes make storage-vs-access mechanism identification assumption-heavy; adjacent forgetting work is increasingly crowded.

**Status**: `WATCH / LOWER_PRIORITY`

## C7. Description–History Gap

Classic risky-decision old question, but reasoning-vs-conversation post-training comparison changes too many factors at once.

**Status**: `WATCH / LOWER_PRIORITY`

## C8. ImplicitMemBench — non-declarative memory

Classic procedural memory / priming / conditioning framing is attractive, but current benchmark does not give a single clean representation-level storage/access dissociation. Directly adding hidden-state analysis risks becoming benchmark + mechanism-tool work.

**Status**: `WATCH / OLD_QUESTION_REFERENCE`

## C9. Agent memory experience-following — proactive interference

Retrieved similar past experiences can bias current agent behavior and propagate errors.

Potential question: corrupt current task state vs hijack action readout.

Why WATCH: agent-memory / ICL retrieval behavior crowded; generic similar-example following is a strong alternative explanation.

**Status**: `WATCH`

---

# D. RECENTLY KILLED / REFERENCE ONLY — do not recycle next round

| Candidate | Verdict | Main reason |
|---|---|---|
| BOULDER generic multi-turn degradation | KILL | Lost-in-Conversation / intent-mismatch / rolling-memory space already crowded |
| code→CoT training-order advantage | KILL | teacher forcing / gradient path / exposure bias etc. create too many competing explanations |
| AR-Bench generic information-gain mechanism | KILL | active task disambiguation / information-gain selection already directly studied |
| Reasoning Trap tool hallucination | KILL AS FOLLOW-UP | seed already provides representation-collapse + late-residual mechanism |
| EMNLP Decision Boundary / SCE | KILL | exact question crowded; hidden-state extension insufficient novelty |
| MathIF reasoning-loses-control | KILL / watch only | overlaps CoT trajectory / correctness-signal / steering work |
| IFEval++ reliability mechanism | KILL | likely paraphrase robustness + probe |
| Instruction tuning → misinformation | KILL | base→instruct changes too many factors; role/context mechanisms already studied |
| RFC-Bench reference-free misinformation | KILL | reference vs no-reference conditions differ in available information |
| Fact mutability → source routing | KILL | mutability highly confounded with relation family |
| Numeracy representation→generation | OUT OF PRIMARY POOL | seed venue outside current primary policy |
| general parametric-vs-context reconciliation | REFERENCE ONLY | NeurIPS 2025 already traces entity flow and performs intervention |
| generic metacognitive activation monitoring | REFERENCE ONLY | already directly studied with neurofeedback-style paradigms |
| LAD / MP-STRUCT | KILL | core explanation entangled with vocabulary/entropy; control tree grows |
| reversal-curse semantics follow-up | KILL | seed already argues semantics exists and order bias drives failure |
| personalization factuality mechanism | KILL AS FOLLOW-UP | seed already gives representational-entanglement account + steering |
| new-knowledge hallucination mechanism | KILL AS FOLLOW-UP | seed already gives attention mechanism + mitigation |
| general RAG context interference | KILL / CROWDED | many ACL/NeurIPS works already study conflict/interference mechanism |

---

# E. Queue discipline

Current execution order:

```text
1. SemTrace prerequisite / critical-cell G0
2. ChronoScope Gold-Context present-drift G0
3. MedEinst pair-locality + Bias-Trap G0
4. continue searching for In-context-deployment official artifact
5. DEEP_AUDIT only after the first three gates move
6. WATCH candidates do not receive large mechanism implementations
```

The goal is not a large topic list. It is to move one object from:

```text
published anomaly
→ dense local critical cell
→ identifiable computation failure
→ causal mechanism
```

without adding rescue controls at every step.