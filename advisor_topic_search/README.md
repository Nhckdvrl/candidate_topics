# 导师向选题搜索：统一标准

这个目录只做一件事：

> **寻找少数真正自然、可推广、符合研究室选题风格、题目宽窄与 novelty 对齐 ACL / EMNLP / NAACL Main、在我们的资源下高概率做出来，并且最终天然留下方法口子的研究问题。**

这里不是“从最新论文里找一个 gap，再套 probe / SAE / activation patching”的地方，也不是“看到某个模型在某个 benchmark 上有怪现象，就把怪现象本身当成研究问题”的地方。

## 文件职责

- `README.md`：唯一总标准；
- `ROUND_*.md`：每一轮完整搜题历史，包括留下和杀掉的方向；
- `ACTIVE_CANDIDATES.md`：唯一当前候选状态表；
- `g0/`：first-shot / prerequisite / frozen G0；
- numbered topic：用户明确选中以后才注册的正式题。

任何 Round / ACTIVE / G0 与 README 冲突时，以 README 为准。

---

# 1. 四条并列的一等原则

以后一个题只有同时满足下面四条，才值得进入 ACTIVE。

## P1 — Construct-first：问题必须先于具体模型存在

先问：

> **如果把模型名、benchmark 名，甚至把 “LLM” 三个字删掉，这个研究对象 / 科学问题还存在吗？**

我们优先研究：

- 语言学 / 语义学 / NLP 资源中的真实问题；
- 认知、学习、记忆、信息处理中的经典 construct；
- 语言变化 / 社会语言现象；
- 表示、压缩、检索、预算中的成熟工程问题；
- 真实系统中长期存在、可定义的 failure；
- 两个成熟因素之间尚未系统回答的 interaction / trade-off。

**模型是研究对象、实验载体或测量工具，不应是题目存在的唯一原因。**

换模型以后，即使效应强弱变化，问题本身仍应有意义；cross-model variation 应成为 boundary condition，而不是让整个题目消失。

## P2 — ACL-scale：宽窄和 novelty 必须对齐 ACL / EMNLP / NAACL Main

天然问题只是必要条件，不是充分条件。

一个 Main-scale 题通常应能收成：

```text
1 个清楚的 mother question
+ 2–4 个自然 factors / competing explanations / subquestions
+ 一套统一 operationalization
+ 2–4 条能写进 abstract 的 headline findings
+ 至少 1 个有意义的 generalization / boundary-condition 轴
```

### 太宽

```text
LLM 是否会推理？
LLM 如何使用证据？
LLM 的记忆机制是什么？
```

### 太窄

```text
某个 Qwen checkpoint 为什么在某 benchmark 多掉 7 分？
某个 prompt wording 是否改变某个 bias？
某层某个 probe 为什么 AUROC 更高？
把已有现象换模型 / 换语言再做一次。
```

### 合适

```text
在固定 storage budget 下，降维与量化如何交互？
哪些因素使 subword LM 获得 character-level information？
模型是否区分 knowledge 与 existence，而不是仅依赖 familiarity？
来源数量相同但独立性不同，证据整合应如何变化？
```

## P3 — Executable-first：第一科学检验必须直接、便宜、可杀

健康路径：

```text
外部 construct
→ programmatic / public data operationalization
→ 很轻的 sanity
→ 直接 behavioral / structural G0
→ 成立后再做深化
```

危险路径：

```text
复杂复现
→ receipt A
→ seed relation B
→ eligibility C
→ layer / probe sweep
→ 才知道自己的 scientific question 能不能开始
```

Topic 25 是永久反例：昂贵 receipt 完整完成，但冻结 seed relation 失败，真正 G0 根本无法进入；失败本身又没有足够 scientific information gain。

以后必须问：

> **从 clone repo 到第一次检验“我们的科学问题”，到底隔了几步？**

## P4 — Method-runway：分析结果必须天然留下方法口子

这条是硬要求，不是 optional bonus。

我们不要求题目从第一天就写出新方法，但必须能够在分析阶段回答：

> **如果我们发现了规律 / failure / mechanism，接下来到底可以改什么？为什么这个方法是由科学发现自然导出的？**

健康结构：

```text
external construct / real problem
→ factor decomposition / diagnosis
→ 找到稳定 failure condition 或 causal factor
→ principled method / objective / data strategy / architecture / inference policy
→ method specifically targets the discovered factor
```

例如：

- source dependence 导致 false corroboration → dependency-aware evidence aggregation；
- detectability 决定 negative evidence 的意义 → search-coverage-aware stopping / belief update；
- 压缩预算下某种交互最优 → budget-aware compression allocation；
- 某类 evidence / representation 被错误忽略 → targeted training objective / routing / retrieval policy。

不健康：

```text
我们发现模型有性质 X。
然后呢？
```

或者：

```text
为了“有方法”，最后随便加一个 prompt / steering vector。
```

**方法口子必须由前面的科学结果推出，而不是论文最后硬塞。**

---

# 2. 强制研究链条：顺序不能倒

所有候选默认按下面顺序：

```text
外部存在的概念 / 规律 / 约束
→ 严格 operationalize
→ behavioral / structural test
→ 因素分解 / competing explanations
→ 必要时看 representation / mechanism
→ 导出 principled intervention / method
```

## Step A — External object first

先写清楚一个不依赖具体模型仍然存在的对象，例如 real / fictional、knowledge / existence、source dependence、storage budget、language change。

禁止从“某层有一个方向”“某 benchmark 出现 reversal”“某 token 很敏感”倒推研究问题。

## Step B — Operationalization before model analysis

必须先明确：

- independent variable；
- dependent behavioral / structural measure；
- matched controls；
- 2–4 个 competing explanations 的不同预测。

如果 construct 只能靠 hidden-state evidence 来定义，它大概率还不是成熟问题。

## Step C — Behavior first

第一阶段先回答：

> 模型是否以符合该 construct / law 的方式处理信息？边界在哪里？

Behavioral null 不能靠 probe / SAE / layer sweep 救回来。

## Step D — Mechanism only when needed

只有 behavior 已经站住，而且内部证据能区分重要 competing explanations 时，才投入 probing / patching / SAE / steering。

机制是解释外部问题，不是贡献本身。

## Step E — Method closes the loop

最后的方法最好回答：

- 哪个 failure condition 可以被消除？
- 哪个 causal factor 可以被控制？
- 哪个 system component 应该被重新设计？
- 新 objective / data construction / routing / aggregation 为什么针对前面的发现？

单纯“这个 direction 能 steer”不算完整 method runway。

---

# 3. 研究室同门给出的真实选题先验

这一节来自实验室 Slack 中 `r_han / r_hamdi / r_kisako / r_utami / r_sato / r_tsujimoto / r_yano / r_xiang` 等频道。

| 组内方向 | 真正研究的对象 | 选题启发 |
|---|---|---|
| `r_han`：Frame semantics / FrameNet | frame relation、coverage、语义资源 | 经典 NLP 资源问题在 LLM 时代仍成立，LLM 是新工具 |
| `r_hamdi`：real vs fictional | real/fictional、knowledge/existence | construct 先存在，再做 behavior → representation → causal role |
| `r_kisako`：dimensionality reduction × quantization | storage budget / compression | 两个成熟因素的 interaction 就能成为完整论文，同时天然能导出 compression 方法 |
| `r_utami`：native-language signal | 现实语言变化、L1 trace | 现实问题 + 强 measurement 可以是 Main-scale 论文 |
| `r_sato`：character-level information | tokenization / orthography / semantic / syntactic factors | 最值得模仿：现象 → 2–4 个解释 → 受控实验拆因素 |
| `r_tsujimoto`：semantic resources / induction | 经典概念体系与资源构建 | old NLP construct 可以用现代 LM 重新测量、诱导、补全 |
| `r_yano`：FrameBench | 我们到底在测什么 | novelty 可以来自更正确的 formulation / measurement |
| `r_xiang`：retrieval-agent safety（早期） | 真实系统 failure | 系统题应回答什么时候失败、为什么、怎么修 |

共同规律：

1. 问题先于模型，模型先于方法；
2. construct 最好有模型外部定义；
3. 2–4 个自然 competing explanations 比大量 settings 更重要；
4. 分析型论文完全成立，但最好能自然导向方法；
5. 标题首先体现问题，而不是 method；
6. claim 强度必须和证据强度匹配；
7. 最好的结果是一条清楚规律、因素分解、interaction、boundary condition，并能告诉我们“该怎么改”。

---

# 4. 六类优先 research shape

## Type A — 自然现象 → 因素分解 → targeted method

代表：`r_sato`。

```text
稳定 phenomenon
→ 2–4 个解释 A/B/C
→ controlled manipulation
→ 找到关键因素
→ 针对该因素设计方法 / data strategy / objective
```

## Type B — 经典 NLP / cognitive / resource 问题 × LLM 时代新 handle

代表：`r_han / r_tsujimoto / r_yano`。

不是“老 benchmark + 新模型”，而是旧问题因为 LLM 时代出现新的 measurement / automation / intervention handle，从而获得实质性新答案。

## Type C — 两个成熟因素 × interaction / budget → policy / allocation method

代表：`r_kisako`。

```text
因素 X 已成熟
因素 Y 已成熟
X × Y interaction 尚未系统回答
→ 统一 budget / constraint
→ 找规律
→ 导出最优 allocation / compression / selection strategy
```

## Type D — 自然社会 / 语言变化问题 × 可解释 proxy

代表：`r_utami`。

需要强 temporal/group controls。方法口子可以是 measurement framework、detection、adaptation、data curation，而不一定是新 neural module。

## Type E — 自然 construct → behavior → representation → causal role → intervention

代表：`r_hamdi`。

适合我们 GPU 丰富的优势，但 mechanism 必须服务于外部 construct。

## Type F — 真实系统目标 / failure → controlled diagnosis → principled mitigation

至少回答：什么时候失败、为什么失败、怎么修，其中最好三者形成一条完整因果链。

---

# 5. ACL / EMNLP / NAACL Main：宽窄与 novelty

目标 venue 的校准中心：

```text
ACL Main
EMNLP Main
NAACL Main
```

NeurIPS / ICML / AAAI / IJCAI 用作背景、方法、collision、adjacent literature，但题目尺度首先按 ACL-family Main 校准。

## 5.1 Scope verdict

每个候选必须明确写：

```text
TOO_BROAD
MAIN_SCALE
TOO_NARROW_FOR_MAIN
```

进入 ACTIVE 原则上必须是 `MAIN_SCALE`。

## 5.2 Scientific novelty 类型

认可：

1. new scientific question / reframing；
2. new factor decomposition；
3. new interaction / trade-off；
4. new empirical regularity；
5. old question, new answer；
6. new measurement / resource formulation；
7. system diagnosis + principled mitigation；
8. analysis that reveals a factor and enables a new method family。

默认不算核心 novelty：

```text
第一次在模型 X 做
第一次在语言 Y 做
第一次用 probe / SAE / activation patching 做现象 Z
多跑几个 benchmark
换更大的模型
把已有结论搬到相邻 domain
```

## 5.3 Near collision 不能靠压窄续命

如果 mother question 已被做完，不能靠 token / layer / model family / 子数据集 / probe / representation 续命。

只有仍能用独立科学问题描述，才算 survived collision。

## 5.4 ACL-paper test

注册前必须能粗写：

```text
Problem: 一个普通 NLP 研究者立即理解的问题。
Gap: 缺一个 scientific distinction，而不是“还没用我们的方法”。
Design: 一套统一 experimental framework。
Finding 1: 主规律。
Finding 2: factor decomposition / competing explanation。
Finding 3: generalization / boundary / interaction。
Method implication: 前面结果具体告诉我们该改什么。
Method: 一个由发现自然推出的 principled intervention。
```

如果最后只能写“我们在三个模型上发现 X 有时发生”，太窄。

如果方法和前面 finding 没有逻辑连接，也不算完整 contribution package。

---

# 6. 资源画像

我们的真实约束：

```text
现金少
人工标注少
GPU 相对充足
```

优先：public dataset、gold labels、released predictions / traces / checkpoints、open-weight models、synthetic / executable labels、现成语料和 metadata。

避免：大量 closed API、大规模人工 annotation、昂贵 upstream reproduction tax。

GPU 优先用于第二阶段：controlled training、cross-model confirmation、checkpoint analysis、probing / patching / SAE / steering、method training。

---

# 7. 正式候选的 9 个硬 Gate

## G1 — External Construct + Natural Question

问题先于具体模型存在，并能用一句普通话解释。

## G2 — ACL / EMNLP / NAACL Scale

必须是 `MAIN_SCALE`，而不是靠堆 settings 放大。

## G3 — Scientific Novelty

必须明确 novelty 类型和 one-liner；不能只是 first-to-apply。

## G4 — Competing Explanations / Interaction

至少 2–4 个自然解释，或者一个清楚 interaction / budget axis。

## G5 — Direct G0 + Low Prerequisite Tax

第一科学检验必须尽快碰到 mother question。

## G6 — Artifact / Resource Executability

数据、字段、scorer、模型、环境都要实际可执行。

## G7 — Interpretable Null

G0 null 仍能排除解释、回答科学问题或界定边界，不能只剩“这个模型没出现”。

## G8 — Method Runway

进入 ACTIVE 前必须写出至少一个**由预期 scientific finding 自然推出的方法方向**，并回答：

```text
Failure / factor 是什么？
我们能控制哪个变量？
方法改哪一层：data / objective / representation / retrieval / aggregation / routing / inference / system policy？
为什么这个方法针对前面的原因，而不是 generic trick？
什么实验能证明 method 真的修正了原问题？
```

如果只能写“以后可以设计一个更好的方法”，不通过 G8。

## G9 — Full-paper Runway

最好结果后还应有自然扩展：因素边界、跨模型/语言/domain generality、mechanism、method、资源或系统应用。

不能“证明了，然后呢？”

---

# 8. Mechanism 与 Method 的位置

机制不是默认贡献，也不是显得高级的装饰。

只有：

```text
external construct 已定义
+ behavior 已站住
+ competing explanations 需要内部证据
```

才值得做机制。

优先 causal intervention；probe / CKA / t-SNE 只能做证据链一部分。

方法也不能反过来定义问题。最健康的是：

```text
问题 → 规律 / failure → 原因 → 方法
```

而不是：

```text
先想一个方法 → 找 benchmark 证明它有效 → 再包装科学问题
```

---

# 9. Model-generality policy

- 发现阶段可先用一个便宜 open model；
- 晋级至少设计 2–3 个不同 family confirmation；
- 如果效应只在一个 family 上存在，除非 family difference 有科学解释，否则降级；
- model size / post-training / architecture 更适合作为 explanatory factor；
- Cross-model variation 应帮助解释问题，而不是决定问题是否存在。

---

# 10. Kill rules

以下任一明显成立，原则上 KILL / DOWNGRADE：

- 题目只能由某模型 / benchmark 怪癖定义；
- construct 在 hidden-state evidence 前无法 operationalize；
- `TOO_NARROW_FOR_MAIN` 或 `TOO_BROAD`；
- novelty 主要是换模型、语言、数据或 interpretability method；
- collision 已做完 mother question，只能继续压窄；
- first scientific test 前有昂贵 prerequisite chain；
- behavior 不成立后靠 probe / SAE / layer sweep 救故事；
- control 越写越多，claim 才勉强成立；
- G0 null 只能说明“这个模型没出现”；
- **最好结果出来后没有 principled method opening；**
- 方法只能是 generic prompt / steering / finetune，和前面的分析没有因果联系；
- benchmark 是研究对象，而不是 measurement instrument。

---

# 11. Round-by-Round 搜题日志制度

**强制保留。** 每轮都形成：

```text
advisor_topic_search/ROUND_XX_YYYY-MM-DD.md
```

Round 必须记录：

```text
1. Round objective
2. search lanes / mother constructs
3. 为什么像研究室同门的 research shape
4. old scientific / NLP anchors
5. 近年 ACL / EMNLP / NAACL / adjacent papers
6. mother question
7. width audit: TOO_BROAD / MAIN_SCALE / TOO_NARROW_FOR_MAIN
8. novelty audit
9. exact collision audit
10. competing explanations / interaction
11. artifact / prerequisite tax
12. frozen first scientific G0
13. interpretable null
14. method runway：最好结果后具体可以设计什么方法
15. method 为什么由 finding 自然推出
16. full-paper runway
17. KEEP / WATCH / KILL 及理由
18. killed lanes 与失败教训
19. 下一轮搜索方向
```

Round 不能只写 survivor。被撞掉、太窄、novelty 不够、artifact 太贵、method opening 太弱的题都必须留下 negative knowledge。

`ROUND_*.md` 只代表搜索历史；搜到五个给用户看不等于五个进 ACTIVE，更不等于自动注册 numbered topic。

---

# 12. 固定搜索流程

```text
Step 1   从组内 research shape / 经典领域问题生成 search lanes
Step 2   定义 external construct
Step 3   找 old literature / competing explanations
Step 4   operationalize variables / controls / predictions
Step 5   查 ACL / EMNLP / NAACL 近年 handle
Step 6   补 AI 顶会 collision
Step 7   写 mother question
Step 8   ACL-scale width audit
Step 9   scientific novelty audit
Step 10  model-swap thought experiment
Step 11  exact collision search
Step 12  artifact / prerequisite audit
Step 13  冻结 minimal behavioral / structural G0
Step 14  预写 factor decomposition
Step 15  写 method runway：如果 finding 成立，具体可以做什么方法
Step 16  只有 behavior 站住才考虑 mechanism
Step 17  写入 ROUND，包括 killed lanes
Step 18  只有 G1–G8 都明确通过才进入 ACTIVE
```

---

# 13. Candidate Card

每个候选进入 `ACTIVE_CANDIDATES.md` 前必须填：

```text
Title / one-line question:
External construct:
Why it matters without a specific model:
Closest lab research shape:
Old scientific / NLP anchor:
Target venue: ACL / EMNLP / NAACL Main
Scope verdict:
Why MAIN_SCALE:
Core scientific novelty type:
Novelty one-liner:
Operationalization (IV / DV / controls):
Recent ACL / EMNLP / NAACL handle:
2–4 competing explanations / interaction axis:
Expected 2–4 headline findings:
Why changing model does not destroy the question:
Closest collisions:
Why collisions do not answer mother question:
Existing artifact:
Paid API requirement:
Human annotation requirement:
Prerequisite tax:
Frozen G0:
Behavioral kill criterion:
Null-result interpretation:
Mechanism trigger:
Possible representation / causal test:
Method runway:
Concrete method family / intervention:
Why the method follows from the expected finding:
Method success criterion:
Cross-model / task / language boundary test:
Full-paper runway:
Status:
```

写不清任何关键项，说明题还没成熟。

---

# 14. 一句话版本

> **先找一个本来就值得研究、换模型也不会消失的问题；把它严格 operationalize；确认宽窄和 novelty 达到 ACL / EMNLP / NAACL Main；用低成本 G0 直接打科学问题；拆清因素 / mechanism；最后必须能由发现自然导出一个 principled method。**

而不是：

> **先找一个模型怪现象，再补分析，再硬塞一个方法。**