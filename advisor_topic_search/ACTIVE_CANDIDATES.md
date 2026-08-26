# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> 当前只保留真正通过母题尺度、novelty、method runway、executable-first 初筛的题。2026-08-27 起，Topic 29 / 原 A3 会议 decision-state 题已正式归档，不再视为 ACTIVE。

Last updated: 2026-08-27

---

# 0. Current hard gates

当前选题必须同时满足：

1. **ACL-scale**：题目宽窄与 novelty package 对齐 ACL / EMNLP / NAACL Main；不是靠换模型、换语言、换数据集续命。
2. **Narrative-level novelty**：允许方法、局部现象、实验工具与已有工作重叠；但 mother question / decisive contrast / 整体叙事不能已经被完整覆盖。
3. **Method runway**：分析结果必须自然导出方法、训练目标、推理策略或系统设计，不能停在“发现现象，然后呢”。
4. **Executable-first**：第一科学检验必须便宜、直接；优先已有公开数据 / benchmark，不接受先大规模造数据再赌现象。

额外执行原则：

- 当前新题搜索只做 **NLP application**；
- 不做 RAG / retrieval-heavy；
- 不做 AI Scientist；
- 不拿一个小 error subtype 当母题；
- 不用机制/模型内部现象倒推研究问题；
- 强模型进步后仍应留下一个真实、独立的任务或约束问题；
- 一个简单 prompt 就几乎消失的问题，默认 method runway 不足，除非其更大的外部任务有独立必要性。

---

# A. ACTIVE CANDIDATES

## A1. 任务到底需要多细的词义？——任务条件化的词义分辨率

**母问题：**

> 传统 WSD 默认每个词都应该被判到固定、尽可能细的词义标签；但真实下游任务所需要的语义分辨率并不相同。LLM 时代，词义粒度能否由“任务真正需要区分什么”与“上下文实际上提供了多少证据”共同决定？

**为什么是自然问题：**

- “不同应用需要不同词义粒度”是经典 WSD 问题，不依赖某个模型存在；
- LLM 已经显著改变普通 WSD 的能力上限，因此旧的固定 inventory / 固定叶节点预测范式值得重新审视；
- 研究对象是 **task-required semantic resolution**，不是“LLM 会不会 WSD”。

**关键因素：**

- 下游任务需要粗区分还是细区分；
- 当前语境只支持粗粒度还是足以支持细粒度；
- 词义层级 / ontology depth；
- 过度消歧是否会在证据不足时降低鲁棒性或下游性能。

**第一枪：**

使用现有 WSD / coarse-WSD / lexical-semantic resources，优先构造无需人工重新标注的层级评测：同一实例在不同下游判别需求下，比较固定最细粒度预测与任务条件化粒度选择。

**方法口：**

Task-Conditioned Hierarchical WSD / Selective Sense Resolution：模型只选择“当前任务真正有用且上下文证据足以支持”的最深语义节点。

**Status:** `WATCH+ / CLASSIC-NLP REFRAMING / EXACT COLLISION + DATA CONTRACT NEEDED`.

---

## A2. 翻译什么时候应该消歧，什么时候应该保留歧义？

**母问题：**

> 当源文本本身证据不足时，忠实翻译是否应该强行选择一个解释？还是应根据上下文证据与目标语言表达能力，在“消歧 / 保留歧义 / 请求澄清”之间选择？

**为什么是自然问题：**

- ambiguity preservation 是机器翻译中的经典问题；
- 真正的翻译目标不应是无条件选择一个 sense，而应避免比原文做更多未经支持的语义承诺。

**关键因素：**

- 上下文是否足以唯一确定解释；
- 目标语言是否能够自然保留同一歧义；
- 目标语言是否被迫选择性别 / 数 / 词义等；
- 使用场景是否允许请求澄清或输出多解释。

**第一枪：**

优先复用现有 lexical-ambiguity MT benchmarks / bilingual ambiguity resources，只补最小的目标语言“是否可保留歧义”判定，不从零造大规模数据。

**方法口：**

Ambiguity-Aware Translation Policy：预测 `RESOLVE / PRESERVE / CLARIFY`，并在证据不足时约束译文不要产生额外语义承诺。

**Status:** `WATCH / MT APPLICATION / COLLISION + GOLD-POLICY DATA RISK`.

---

## A4. 文本变简单以后，“必须 / 可以 / 禁止 / 例外”有没有偷偷变掉？——规范语义保持的受约束改写

**母问题：**

> 文本简化 / 改写提高可读性时，是否会系统改变义务、许可、禁止、权利、条件与例外等规范效力？一般 semantic similarity 是否足以保证这种高风险语义维度不被破坏？

**为什么是自然问题：**

- 法律、政策、规章、合同中的 deontic semantics 是稳定外部对象；
- 法律文本简化已经有公开数据与 meaning-preservation 研究，但总体相似度并不能直接保证规范结构不变；
- 即使更强模型减少自然错误，“受约束改写必须保持规范效力”这一任务目标本身仍然存在。

**关键因素：**

- obligation / permission / prohibition / entitlement / no-obligation；
- 条件与例外是否被删除或放宽；
- 责任主体是否变化；
- 可读性提升与规范语义保持之间的 trade-off。

**已完成 G0：**

Topic 30 已用 LexDeMod gold trigger spans 做 controlled normative-force contrasts，并验证普通相似度可在规范效力已反转时仍保持极高。Lex-Simple 也有足够 deontic-eligible pairs。下一关键问题是自然 simplification drift prevalence，而不是继续证明 metric blind spot。

**方法口：**

Deontic-Structure-Constrained Simplification：显式维护 `actor → modality → action → condition → exception`，生成时约束规范结构，再优化 readability。

**Status:** `KEEP / APPLIED NLP / NATURAL-DRIFT G0 NEXT`.

---

# B. PAUSED LANES

当前用户明确要求只继续找 **NLP application** 题，因此 interpretability / mechanism lane 暂停，不为凑数量强行填候选。

---

# C. RECENTLY KILLED / ARCHIVED / DO NOT REVIVE WITHOUT A NEW MOTHER QUESTION

- **Topic 29 / 会议 decision-state preservation**：三模型 temporal-prefix 现象真实且很大，但独立母题过窄；最强 G0 依赖人为截断到最终决定之前；一句 preservation instruction 将错误从 `44–75%` 降到 `0–1.9%`，method runway 和强模型寿命不足。已正式归档。不要靠人工标注、更多模型、机制分析或“commitment inflation”大伞救题。
- 固定检索预算 / retrieval granularity：用户明确不感兴趣，且整体偏 retrieval / RAG。
- Controls Are Tests of Assumptions：AI Scientist 方向，不继续。
- When Is No Evidence Evidence：search / retrieval 味过重，不继续。
- Scope vs Verbosity in Retrieval：检索相关，不继续。
- 命题内容 vs 事实 / 承诺状态：与近期 factuality / belief / epistemic representation 工作过近，且容易成为局部 error taxonomy。
- 预设 vs 明说：近期 presupposition 行为工作密集，且不是当前 application 优先方向。
- Metaphor / idiom literal-vs-figurative mechanism：mother narrative 已被近邻工作明显覆盖。
- 实体状态更新：近期 state tracking / causal intervention 工作已覆盖母问题，KILL。
- 规则例外 rigidity：需要先构造跨场景高质量数据再赌现象，沉没成本过高。
- 语篇 anaphora accessibility mechanism：机制 lane，暂停。
- 主谓一致 attraction mechanism：语言学小现象 + mechanism lane，不继续当前版本。

---

# D. CURRENT PRIORITY

1. **只找新的 NLP application mother question**：真实任务先于模型存在、强模型时代仍有意义、不是一个小 error subtype。
2. 新题必须同时给出：exact recent collision audit、可直接拿到的 public-data contract、一个便宜直接的 first shot、2–4 个自然因素/边界、以及不靠“prompt 一句解决”的 principled method runway。
3. A4 继续保留；A1/A2 维持 WATCH，除非 exact audit 后仍有清楚独立叙事。
4. 找不到足够好的题就空着，不因数量压力降标准。
