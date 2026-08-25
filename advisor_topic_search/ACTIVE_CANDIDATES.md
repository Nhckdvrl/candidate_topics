# Active Candidates — Advisor Topic Search

> 这是 `advisor_topic_search/` 的**唯一当前候选状态表**。
>
> 2026-08-26 起，旧的 cue competition / positional imprinting / source-arbitration 等候选不再视为当前 shortlist。当前 ACTIVE 只保留用户明确认可、并通过 README 四条一等原则初筛的题。

Last updated: 2026-08-26

---

# 0. Current hard gates

当前选题必须同时满足：

1. **ACL-scale**：题目宽窄与 novelty package 对齐 ACL / EMNLP / NAACL Main；不是靠换模型、换语言、换数据集续命。
2. **Narrative-level novelty**：允许方法、局部现象、实验工具与已有工作重叠；但 mother question / decisive contrast / 整体叙事不能已经被完整覆盖。
3. **Method runway**：分析结果必须自然导出方法、训练目标、推理策略或系统设计，不能停在“发现现象，然后呢”。
4. **Executable-first**：第一科学检验必须便宜、直接、可杀。尤其可解释性题，优先要求已有公开数据 / benchmark、已知稳定行为现象，最好还有局部机制基础；不接受先重造大规模数据再赌内部机制是否存在。

此外：

- 不做 RAG / retrieval-heavy 题；
- 不做 AI Scientist 方向；
- 不为了凑数量强行加入高风险机制题；
- mechanism 只能解释一个先于模型存在的具体问题，不能从 probe / SAE / head 倒推题目。

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

**机制定位：**

可解释性不是主线。2026 已已有概念层级内部表征 / activation patching 近邻；内部分析最多作为 bonus，不能承担 novelty。

**Status:** `STRONG_KEEP / APPLICATION+CLASSIC_NLP_REFRAMING / G0_DESIGN_NEEDED`.

---

## A2. 翻译什么时候应该消歧，什么时候应该保留歧义？

**母问题：**

> 当源文本本身证据不足时，忠实翻译是否应该强行选择一个解释？还是应根据上下文证据与目标语言表达能力，在“消歧 / 保留歧义 / 请求澄清”之间选择？

**为什么是自然问题：**

- ambiguity preservation 是机器翻译中的经典问题；
- 当前大量 lexical-ambiguity MT 工作默认目标仍是“找到正确 sense 并翻出来”；
- 但真正的翻译目标应是：**不要比原文做更多未经支持的语义承诺。**

**关键因素：**

- 上下文是否足以唯一确定解释；
- 目标语言是否能够自然保留同一歧义；
- 目标语言是否被迫选择性别 / 数 / 词义等；
- 使用场景是否允许请求澄清或输出多解释。

**第一枪：**

优先复用现有 lexical-ambiguity MT benchmarks / bilingual ambiguity resources，只补最小的目标语言“是否可保留歧义”判定，不从零造大规模数据。

**方法口：**

Ambiguity-Aware Translation Policy：预测 `RESOLVE / PRESERVE / CLARIFY`，并在证据不足时约束译文不要产生额外语义承诺。

**机制定位：**

不把“内部多解释何时坍缩”作为主线；ACL 2026 已有 delayed disambiguation 的内部路径与 causal steering 近邻。

**Status:** `STRONG_KEEP / MT_APPLICATION_REFRAMING / DATA_FEASIBILITY_AUDIT_NEEDED`.

---

## A3. 会议摘要会不会把“讨论过 / 建议过 / 有条件同意”写成“已经决定”？

**母问题：**

> 会议摘要中的事实内容即使没有明显 hallucination，命题的行动状态仍可能被压缩时升级：proposal → decision、conditional → unconditional、open → resolved、rejected → accepted。LLM 是否系统性破坏这种 decision status？

**为什么是自然问题：**

- 会议理解中 proposal / agreement / resolution / action item / decision 是经典对象；
- 现有 LLM meeting summarization 主要关注 omission、hallucination、relevance、structure、personalization；
- 本题关注的是 **同一命题的决策状态在摘要过程中是否被错误升级或降级**，不是 generic factuality。

**关键因素：**

- proposal / discussion / tentative agreement / conditional commitment / rejection / reversal / final decision；
- 是否存在后续修订或撤销；
- 压缩强度；
- 摘要格式（自由摘要 vs minutes / action items）。

**第一枪：**

优先使用 AMI / QMSum 等现有会议数据与 decision-related dialogue-act / decision abstracts；先验证能否从公开标注直接构造 proposal→decision 等状态错误，不先人工造会议。

**方法口：**

Decision-State-Preserving Meeting Summarizer：生成前维护 `proposition → status → owner/speaker → condition → revision history`，再据此生成 minutes。

**Status:** `STRONG_KEEP / APPLIED_NLP / PUBLIC-DATA_G0_AUDIT`.

---

## A4. 文本变简单以后，“必须 / 可以 / 禁止 / 例外”有没有偷偷变掉？——规范语义保持的受约束改写

**母问题：**

> 文本简化 / 改写提高可读性时，是否会系统改变义务、许可、禁止、权利、条件与例外等规范效力？一般 semantic similarity 是否足以保证这种高风险语义维度不被破坏？

**为什么是自然问题：**

- 法律、政策、规章、合同中的 deontic semantics 是稳定外部对象；
- 法律文本简化已经有公开数据与 meaning-preservation 研究，但“所有语义维度同等重要”的总体相似度并不能直接保证规范结构不变；
- 题目可从 Legal NLP 扩展到公司政策、使用条款、安全说明等受约束改写。

**关键因素：**

- obligation / permission / prohibition / entitlement / no-obligation；
- 条件与例外是否被删除或放宽；
- 责任主体是否变化；
- 可读性提升与规范语义保持之间的 trade-off。

**第一枪：**

优先复用 SIMPLE-LAW、LexDeMod 等已有法律简化 / deontic modality 资源，检验现成 simplification systems 是否出现结构化规范语义漂移；不先自建大规模法律语料。

**方法口：**

Deontic-Structure-Constrained Simplification：显式抽取 `actor → modality → action → condition → exception`，生成时约束这些结构不变，再优化 readability。

**Status:** `KEEP / APPLIED_NLP / DOMAIN-BREADTH_AND_COLLISION_AUDIT_NEEDED`.

---

# B. OPEN SLOT — INTERPRETABILITY

当前**不强行填满第 5 题**。

可解释性候选只有满足以下额外条件才可进入 ACTIVE：

```text
具体、自然、模型外部存在的问题
+ 已有公开 benchmark / 成熟数据
+ 已知稳定 behavioral phenomenon / failure
+ 最好已有局部 representation / circuit 基础
+ 至少两个可区分的内部机制解释
+ causal intervention 能产生选择性预测
+ mechanism 结果能自然导出方法
```

优先寻找：

- 非语言学小现象；
- 普通人一听就明白的实体 / 决策 / 认知 / 系统行为；
- 不需要先大规模构造新数据；
- behavioral null 或 mechanism null 能低成本 kill。

---

# C. RECENTLY KILLED / DO NOT REVIVE WITHOUT A NEW MOTHER QUESTION

- 固定检索预算 / retrieval granularity：用户明确不感兴趣，且整体偏 retrieval / RAG。
- Controls Are Tests of Assumptions：AI Scientist 方向，不继续。
- When Is No Evidence Evidence：search / retrieval 味过重，不继续。
- Scope vs Verbosity in Retrieval：检索相关，不继续。
- 命题内容 vs 事实 / 承诺状态：与近期 factuality / belief / epistemic representation 工作过近，且因果识别容易混入句法和来源结构。
- 预设 vs 明说：2025–2026 presupposition 行为工作密集，内部 causal construct 难以隔离。
- Metaphor / idiom literal-vs-figurative mechanism：EACL / ACL 2026 已有 causal tracing / delayed disambiguation / steering，mother narrative 已被明显覆盖。
- 实体状态更新：ICML 2025–2026 已覆盖 state tracking、state changes、global update vs query-time rebinding、causal interventions；mother question collision，KILL。
- 规则例外 rigidity：行为问题自然，但需要先构造跨场景高质量数据再赌 mechanism；G0 沉没成本过高，不符合 executable-first。
- 语篇 anaphora accessibility mechanism：已有行为基础但共享内部 construct 是否存在风险过高；不继续当前版本。
- 主谓一致 attraction mechanism：机制可做性高，但语言学味过重，暂不进入 advisor shortlist。

---

# D. CURRENT PRIORITY

1. 对 A1–A4 继续做 **exact collision + public-data G0 feasibility + method runway** 审查；
2. 深挖一个真正满足 B 节硬门槛的可解释性题；
3. 找到合格的第 5 题之前，不因数量压力降低标准。
