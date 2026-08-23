# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> `ROUND_*.md` 保存搜索历史与完整审计过程；本文件只回答一个问题：**现在到底有哪些题还活着，下一步做什么？**
>
> 每轮搜索/审核结束后必须同步更新本文件。已经 KILL 的题不允许因为换模型、换数据集、换 probe 就偷偷复活；需要复活时必须写明新的 scientific reason。

Last updated: 2026-08-23
Source of current ranking: `ROUND_06_2026-08-23.md`

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

## B. HOLD — 科学问题强，但 prerequisite/resource 还没过

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

当前尚未确认完整官方 reproduction artifact / generator / exact code path。自己重建会重新同时承担：

- task-generation mismatch；
- prompt mismatch；
- latent-representation measurement mismatch；
- critical-cell reproduction risk。

**Promotion gate**:

满足其一再升 ACTIVE：

1. 找到官方 code/data；
2. appendix 足以无 tuning 地在一个 open model 上精确重现 critical cell。

**Status**: `SCIENCE_TOP / HOLD_FOR_ARTIFACT`

---

## C. WATCH — 有价值，但当前不值得先下注

### C1. Temporal Forgetting — storage loss vs access loss

**Seed**: ACL 2026 main, *Temporal Sampling for Forgotten Reasoning in LLMs*.

**Attraction**: 同一实例在 earlier checkpoint 正确、later checkpoint 错误；官方释放 Qwen2.5-7B RL checkpoints 与 sampled responses。

**Why only WATCH**:

- open-ended CoT across checkpoints 不具有稳定步骤对齐；
- representation bases 跨 checkpoint 变化；
- storage-vs-access 很容易被迫依赖 cross-checkpoint alignment / Procrustes / trajectory matching / elicitation controls；
- 第一枪 identification 太复杂。

**Status**: `WATCH / REMOVE_FROM_ACTIVE_QUEUE`

### C2. Description–History Gap mechanism

**Seed**: ACL 2026 main / Outstanding, description–experience gap in risky decision making.

**Attraction**: 经典 decision-science old question；公开 Qwen/OLMo family choice data；post-training 与行为差异明显。

**Why only WATCH**:

reasoning-vs-conversation model comparison同时改变太多训练因素；如果问“为什么 math reasoning training 改变 DH gap”，会迅速长出大量 disentangling controls。

**Status**: `WATCH / REMOVE_FROM_ACTIVE_QUEUE`

---

## D. RECENTLY KILLED — 不要在下一轮重复提出

| Candidate | Verdict | Main reason |
|---|---|---|
| EMNLP Decision Boundary / SCE mechanism | KILL | exact question 已拥挤；继续加 hidden-state/patching 不形成足够 scientific novelty |
| MathIF reasoning-loses-control mechanism | KILL / watch only | 与已有 CoT trajectory / correctness-signal / steering work过近 |
| IFEval++ reliability mechanism | KILL | 下一问容易退化成 paraphrase robustness + probe |
| Instruction tuning → misinformation | KILL | base→instruct 同时改变过多因素，causal attribution 需要复杂 controls |
| RFC-Bench reference-free misinformation | KILL | reference vs reference-free conditions 信息本身不同，不能干净解释成 latent knowledge vs use |
| Fact mutability → source routing | KILL / demoted | mutable/stable 与 relation family 高度混淆，probe 容易只读 relation identity |
| Numeracy representation→generation | OUT OF PRIMARY POOL | seed venue 不符合当前主 seed policy；可作背景/方法参考，不作为主攻题 |
| Progressive Quiz Bowl reversal | HOLD-OLD / not current top | G0 脚本已存在，但当前 ranking 已被更高可行性对象替代 |

---

## E. Queue discipline

当前执行顺序固定为：

```text
1. SemTrace prerequisite / critical-cell G0
2. MedEinst pair-locality + critical-cell G0
3. 继续寻找 In-context deployment 官方 artifact
4. 只有前两条被 kill 或 #3 artifact gate 通过，才从 WATCH 中提题
```

禁止同时给 5–10 个候选写大段 mechanism code。

目标不是保持题池看起来丰富，而是尽快把研究风险集中到**一个真正存在、可识别、值得解释的 scientific object**上。
