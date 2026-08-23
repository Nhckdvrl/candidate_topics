# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 保存搜索历史与完整审计过程；本文件只回答一个问题：**现在到底有哪些题还活着，下一步做什么？**
>
> 每轮搜索/审核结束后必须同步更新本文件。已经 KILL 的题不允许因为换模型、换数据集、换 probe 就偷偷复活；需要复活时必须写明新的 scientific reason。

Last updated: 2026-08-23
Source of current ranking: `ROUND_07_2026-08-23.md`

---

## A. ACTIVE — 可以直接进入 prerequisite / G0

### A1. SemTrace — lexical retrieval survives, semantic execution fails in the middle

**Seed**: ACL 2026 main, *Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding*.

**Research question**:

> **Why does context position selectively destroy operational state transition while lexical access to the same code remains available?**

**Why it is alive**:

- ACL main seed；
- 同一 model / 同一 program / 只移动 context position；
- seed 已报告巨大 semantic middle-position failure；
- lexical retrieval 可作为同一输入上的 intact control；
- 官方 reproduction package 完整；
- 7B/8B open models 可跑；
- SemTrace synthetic program 的 intermediate states 可程序化得到 exact labels；
- 不需要人工标注或付费 API。

**Frozen critical cell**:

```text
same program + same model
edge:   semantic correct
middle: semantic wrong
edge:   lexical correct
middle: lexical correct
```

只有这个 cell 有足够密度时才允许做 mechanism。

**First G0**:

1. 在一个 seed 已验证的 7B/8B open model 上复现 edge vs middle semantic drop；
2. 对相同实例验证 lexical retrieval 保持；
3. 统计 `edge-correct / middle-wrong / lexical-intact` 密度；
4. 若密度足够，再针对 exact intermediate program state 做 bounded layer-wise diagnostic / causal patching。

**Kill line**:

- critical cell 稀疏；
- lexical retrieval 与 semantic execution 同时崩；
- intermediate-state failure 无法与 generic position effect 区分；
- 最终只能得到“middle position 比较差”而没有 selective semantic-computation story。

**Collision boundary**:

不能写成 generic “state tracking mechanism” 或 generic “lost in the middle”。已有工作已经覆盖 transformer state tracking、binding circuits 和一般 positional failures。新意必须落在 **lexical access intact 时 semantic computation selective collapse**。

**Status**: `SURVIVAL_TOP / CODE_NEXT`

---

### A2. MedEinst — encoding failure or failed belief update under Einstellung?

**Seed**: ACL 2026 main, *MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis*.

**Research question**:

> **When decisive counterevidence is introduced, does the model fail to encode it, or is it encoded but unable to update the already-formed diagnostic state?**

**Why it is alive**:

- classic Einstellung / mental-set scientific question；
- 5,383 control–trap pairs；
- seed 已定义关键 failure：control correct，但 trap 仍回到 control diagnosis；
- open model 已出现 substantial Bias Trap Rate；
- diagnosis label 可自动评分；
- 不需要重新证明“Einstellung 是否存在”。

**Mandatory prerequisite**:

先验证 control→trap 是否真的是局部 discriminative-evidence edit，而不是整段 narrative 大面积重写。

Existing G0:

```text
advisor_topic_search/g0/medeinst_pair_structure.py
```

**Promotion condition**:

- pair edits 足够局部；
- control accuracy 足够高；
- Bias Trap critical cell 密度足够；
- decisive evidence span 可从公开数据/构造中稳定确定，不依赖大规模人工医学判断。

**Kill line**:

- pair edit diffuse；
- critical cell 太少；
- 需要人工逐例决定“真正关键证据”；
- 为解释结果必须不断增加 clinical confound controls。

**Status**: `SURVIVAL_TOP / RUN_PAIR_LOCALITY_G0`

---

## B. HOLD / DEEP AUDIT — 科学问题强，但 prerequisite/resource 还没过

### B1. In-context representation deployment bottleneck

**Seed**: ACL 2026 main, *Language Models Struggle to Use Representations Learned In-Context*.

**Research question**:

> **What deployment-specific routing/readout computation makes an already learned in-context representation inert under one interface but usable under another?**

**Why it is scientifically strong**:

Seed 已直接报告：

```text
novel in-context semantics are encoded in latent representations
BUT
models struggle to deploy them downstream
```

这比我们自己先做 probe 再寻找 gap 要健康得多，并且 paper 没有给出完整 causal mechanism。

**Why it is not ACTIVE yet**:

当前仍未确认完整官方 reproduction artifact / generator / exact code path。自己重建会重新同时承担 task-generation、prompt、latent measurement 和 critical-cell reproduction risk。

**Promotion gate**:

1. 找到官方 code/data；或
2. appendix 足以无 tuning 地在一个 open model 上精确重现 critical cell。

**Status**: `SCIENCE_TOP / HOLD_FOR_ARTIFACT`

---

### B2. Table DRE — structural understanding vs value referencing

**Seed**: ACL 2026 main, *When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors*.

**Research question**:

> **When table structure is understood but the model cites the wrong value, is the failure in localization, entity–value binding, or late readout substitution?**

**Why it is interesting**:

- DRE 在 1.7B–20B 模型上稳定出现；
- table 的 row / column / value identity 可自动得到 exact labels；
- seed 已构造 programmatic table/value perturbations；
- 可以寻找 `structure-correct / value-reference-wrong` 的天然 critical cell；
- 很适合 open model + causal intervention。

**Main collision**:

通用 entity binding / positional binding / counterfactual binding 已经有较成熟机制工作。因此不能把贡献写成“首次发现 binding failure”。只有 **table-specific referencing failure under intact structural understanding** 才有机会成立。

**Promotion gate**:

- 找到并核实官方 reproduction artifact；
- 在一个 7B/8B accessible open model 上确认高密度 `structure-correct / reference-wrong` cell；
- 证明该 cell 不能被 generic retrieval/binding failure 简单解释。

**Status**: `DEEP_AUDIT / ARTIFACT_AND_COLLISION_GATE`

---

### B3. Context-shaped truth geometry → source choice

**Seed**: ACL 2026 main, *How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs*.

**Research question**:

> **Do context-induced transformations of truth representations causally determine whether context or parametric memory controls behavior under conflict?**

**Why it is interesting**:

Seed 已经报告 context 对 truth-vector direction/magnitude 的系统性改变，且 conflict with parametric knowledge 会造成更大的 geometry shift。因此我们不需要自己先证明“context 会改变 truth representation”。

真正的下一问应当是：

```text
context-induced truth geometry
→ source selection
→ final behavior
```

而不是再画一遍 probe / PCA / steering 图。

**Main risks**:

- truth-vector / knowledge-conflict / steering 文献很拥挤；
- 本轮尚未找到可信官方 reproduction repo；
- 如果只能证明 truth vector 可 steer，则 novelty 不足。

**Promotion gate**:

- 找到官方 artifact；
- 在 same-fact aligned/conflicting context 中，geometry 能预测 instance-level source choice；
- intervention 能特异性改变 source choice，而非普遍改变 truthfulness/logits。

**Status**: `DEEP_AUDIT / HOLD_FOR_ARTIFACT`

---

## C. WATCH — 有价值，但当前不值得先下注

### C1. Temporal Forgetting — storage loss vs access loss

**Seed**: ACL 2026 main, *Temporal Sampling for Forgotten Reasoning in LLMs*.

**Attraction**: 同一实例 earlier checkpoint 正确、later checkpoint 错误；有公开 Qwen2.5-7B RL checkpoints 与 sampled responses。

**Why only WATCH**:

- open-ended CoT across checkpoints 不具有稳定步骤对齐；
- representation bases 跨 checkpoint 变化；
- storage-vs-access 很容易被迫依赖 alignment / Procrustes / trajectory matching / elicitation controls；
- Round 07 又发现 NeurIPS 等已有更多 example-level forgetting / retained-knowledge 工作，collision 更重。

**Status**: `WATCH / LOWER_PRIORITY`

### C2. Description–History Gap mechanism

**Seed**: ACL 2026 main / Outstanding, description–experience gap in risky decision making.

**Attraction**: 经典 decision-science old question；公开 Qwen/OLMo family choice data；post-training 与行为差异明显。

**Why only WATCH**:

reasoning-vs-conversation model comparison同时改变太多训练因素；若解释“为什么 math reasoning training 改变 DH gap”，会迅速长出大量 disentangling controls。

**Status**: `WATCH / REMOVE_FROM_ACTIVE_QUEUE`

### C3. ImplicitMemBench — non-declarative memory in LLM agents

**Seed**: ACL 2026 main / Best Resource Paper, *ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models*.

**Attraction**:

- procedural memory / priming / classical conditioning 都是经典 old questions；
- 17 模型，300-item suite；
- 报告很强的 inhibition-vs-preference asymmetry。

**Why only WATCH**:

当前 benchmark 定义的是 behavioral adaptation，并没有天然给出 representation-level storage/access dissociation。直接“上 hidden-state probe”会很像 benchmark + mechanism tool 的套壳；三个 construct 也可能没有统一机制。

**Status**: `WATCH / OLD_QUESTION_REFERENCE`

---

## D. RECENTLY KILLED / REFERENCE ONLY — 不要在下一轮重复提出

| Candidate | Verdict | Main reason |
|---|---|---|
| EMNLP Decision Boundary / SCE mechanism | KILL | exact question 已拥挤；继续加 hidden-state/patching 不形成足够 scientific novelty |
| MathIF reasoning-loses-control mechanism | KILL / watch only | 与已有 CoT trajectory / correctness-signal / steering work过近 |
| IFEval++ reliability mechanism | KILL | 下一问容易退化成 paraphrase robustness + probe |
| Instruction tuning → misinformation | KILL | base→instruct 同时改变过多因素，且 role/context-awareness mechanism 已被深入研究 |
| RFC-Bench reference-free misinformation | KILL | reference vs reference-free conditions 信息本身不同，不能干净解释成 latent knowledge vs use |
| Fact mutability → source routing | KILL / demoted | mutable/stable 与 relation family 高度混淆，probe 容易只读 relation identity |
| Numeracy representation→generation | OUT OF PRIMARY POOL | seed venue 不符合当前主 seed policy；仅作背景/方法参考 |
| Progressive Quiz Bowl reversal | HOLD-OLD | G0 脚本已存在，但当前 ranking 已被更高可行性对象替代 |
| General instruction FT → context-awareness loss | KILL / reference | seed/adjacent work 已做 attention、role bias、head steering 和 training-data attribution |
| Personality self-report vs behavior | KILL AS PRIMARY | activation/personality intervention space 已快速拥挤 |
| Refinement fluency-vs-adequacy | KILL | judge/专业人工依赖较高，且 seed 已给 projection-style解释 |
| Unlearning latent retained knowledge | KILL | 领域拥挤，评测复杂且常依赖 judge / knowledge-correlation machinery |
| METER causal reasoning mechanism | KILL | seed 已做 evidence-node saliency / information-flow analysis |
| ACL SAE causal semantic modules | REFERENCE ONLY | 已是完整 causal mechanism 方法工作，不是新的 scientific object |

---

## E. Queue discipline

当前执行顺序：

```text
1. SemTrace prerequisite / critical-cell G0
2. MedEinst pair-locality + critical-cell G0
3. 继续寻找 In-context deployment 官方 artifact
4. Table DRE artifact + critical-cell + collision audit
5. Context-shaped truth artifact + source-choice audit
6. WATCH 不主动开大实验
```

禁止同时给 5–10 个候选写大段 mechanism code。

Round 07 的一个重要结论是：**没有新发现能仅凭题味挤掉已经 survived 多轮审计的候选。** 新题必须在 scientific value 和 survival probability 两边都真正更强，才允许改排名。

目标不是保持题池丰富，而是尽快把研究风险集中到**一个已经真实存在、可识别、值得解释的 scientific object**上。