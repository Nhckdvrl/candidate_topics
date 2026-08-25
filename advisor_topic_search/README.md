# 导师向选题搜索：统一标准

这个目录只做一件事：

> **寻找少数真正自然、可推广、符合研究室选题风格、达到 ACL / EMNLP / NAACL 等顶会尺度，并且在我们的资源下高概率做出来的研究问题。**

这里不是“从最新论文里找一个 gap，再套 probe / SAE / activation patching”的地方，也不是“看到某个模型在某个 benchmark 上有怪现象，就把怪现象本身当成研究问题”的地方。

**README 是唯一总标准。** `ROUND_*.md` 是搜索历史，`ACTIVE_CANDIDATES.md` 是当前状态，`g0/` 是 first-shot / prerequisite 实验。三者和 README 冲突时，以 README 为准。

---

# 0. 最重要的总原则：研究对象必须先于模型存在

以后所有候选首先问：

> **如果把具体模型名、benchmark 名，甚至把 “LLM” 三个字暂时删掉，这个研究对象 / 科学问题还存在吗？**

如果答案是否定的，默认不进 ACTIVE。

我们优先研究的是：

- 一个语言学 / 语义学 / NLP 资源问题；
- 一个认知、学习、记忆、信息处理中的经典 construct；
- 一个语言变化 / 社会语言现象；
- 一个信息压缩、表示、检索、预算或系统设计问题；
- 一个真实任务中长期存在、可定义的 failure；
- 一个成熟因素之间尚未系统回答的 interaction / trade-off。

**模型应当是研究对象、测量工具或实验载体，而不是题目存在的唯一原因。**

理想情况是：换成另一个模型，问题仍然成立；模型之间是否表现不同，本身成为关于该 construct 的 boundary condition，而不是让整个题目失效。

## 0.1 强制研究链条：顺序不能倒

从现在开始，所有候选优先按下面这条链条组织：

```text
外部存在的概念 / 规律 / 约束
→ 严格 operationalize
→ 测 LLM 是否正确处理它（behavior / structural test）
→ 再问内部表示 / mechanism
→ 最后才谈 intervention / mitigation
```

五步的含义：

### Step A — External object first

先写清楚一个**不依赖具体模型仍然存在**的科学对象。例如 real / fictional、knowledge / existence、cue competition、source dependence、storage budget、language change。

禁止从“某层有一个方向”“某 benchmark 出现 reversal”“某个 token 很敏感”开始倒推研究问题。

### Step B — Operationalization before model analysis

必须先把外部 construct 变成可控制、可测量的变量：

- 哪个因素被 manipulate？
- 哪些因素必须 matched / held constant？
- 如果模型正确处理该 construct，行为上应出现什么关系？
- competing explanations 分别预测什么？

**如果 construct 无法在 hidden state 之外被定义和检验，它大概率还不是一个成熟研究问题。**

### Step C — Behavior is the first scientific test

第一阶段先回答：

> 模型是否以符合该外部规律 / construct 的方式处理信息？边界在哪里？

Behavioral null 不能靠 probe / SAE / layer sweep 救回来。若行为对象根本不稳定，优先杀题或改问题，而不是继续找内部信号。

### Step D — Mechanism is optional and subordinate

只有 behavior / structural result 已经站住，而且内部证据能区分重要 competing explanations 时，才投入 probing / patching / SAE / steering。

机制的任务是**解释外部问题**，不是把“能 decode 某个信号”本身升级成贡献。

### Step E — Intervention must close the scientific loop

Intervention 最好用于回答：

- 这个表示是否具有 causal role？
- 修正该机制是否改善对原 construct 的处理？
- 能否构造 principled mitigation / system design？

单纯证明“这个 direction 可以 steer”不够。

这条顺序对 Hamdi 类型的机制题尤其重要：real / fictional 与 knowledge / existence 的区分先于 Gemma/Qwen 存在，因此即使 representation 结果跨模型不同，研究问题仍然成立；representation difference 只会成为新的 boundary condition。

## 0.2 好问题与坏问题的区别

更像研究室同门：

```text
哪些因素让 subword LM 获得 character-level information？
FrameNet 中哪些语义关系存在系统性缺口，现代 LM 能否帮助补全？
在固定 storage budget 下，降维和量化如何交互？
LLM 普及后，academic English 中的 L1 signal 是否减弱？
模型是否区分 real / fictional status，而不是仅依赖 familiarity？
多个表面一致的报告在 source dependence 不同时，是否应具有相同证据强度？
```

不够自然：

```text
为什么 Qwen 在第 4 个 clue 后把正确答案改错？
为什么某模型在某 benchmark 的 middle position 掉 20 分？
为什么某层 probe 能读到一个信号但最终 token 没输出？
```

后者只有在能被提升为一个**独立、跨模型仍有意义的科学问题**后才考虑。

---

# 1. 研究室同门给出的真实选题先验

这一节来自实验室 Slack 中 `r_han / r_hamdi / r_kisako / r_utami / r_sato / r_tsujimoto / r_yano / r_xiang` 等频道。目的不是列履历，而是回答：

> **导师实际认可的题，研究对象长什么样？**

| 组内方向 | 真正研究的对象 | 对我们最重要的启发 |
|---|---|---|
| `r_han`：Frame semantics / FrameNet relation & induction | 显式语义资源、frame relation、coverage | 经典 NLP 资源问题在 LLM 时代仍然成立；LLM 是新工具，不是问题本身 |
| `r_hamdi`：real vs fictional | 现实/虚构这一独立 ontology construct | 做机制也要 construct-first；familiarity、genre、knowledge 等是 competing explanations |
| `r_kisako`：dimensionality reduction × quantization | 表示压缩与 storage budget | 两个成熟因素的 interaction / trade-off 就可以是完整论文，不需要 mechanism |
| `r_utami`：native-language signal in academic English | 语言变化与 L1 trace | 一个现实世界自然问题 + 可解释 measurement 可以直接成为 EMNLP 尺度研究 |
| `r_sato`：character-level information acquisition | tokenization、orthography、semantic / syntactic factors | 最值得模仿：已知现象 → 2–4 个自然解释 → 受控实验逐个拆解 |
| `r_tsujimoto`：semantic resources / induction | 经典概念体系与资源构建 | old NLP construct 可以用现代 LM 重新测量、诱导或补全 |
| `r_yano`：FrameBench / frame-semantic understanding | “我们真正想测什么”这一 measurement 问题 | novelty 可以来自更正确的问题 formulation，而不是新模型 |
| `r_xiang`：retrieval-agent safety（早期） | 真实系统失败 | 系统题也应从稳定 failure 出发，回答什么时候、为什么、怎么修 |

从组内项目抽出的共同规律：

1. **问题先于模型，模型先于方法。**
2. **研究对象最好有模型外部定义。**
3. **先 operationalize，再测 behavior，最后才允许 mechanism / intervention。**
4. **换模型后问题仍然有意义。**
5. **2–4 个自然 competing explanations 比大量 settings 更重要。**
6. **分析型论文完全成立，不强迫 mechanism。**
7. **标题和第一页应该先让人看到问题，不是 method。**
8. **claim 强度必须和证据强度匹配。**
9. **最好的结果往往是一条清楚的规律、因素分解、interaction 或 boundary condition。**

---

# 2. 我们优先寻找的六类 research shape

## Type A — 自然现象 → 因素分解

代表：`r_sato`。

```text
一个跨模型 / 跨数据仍有意义的 phenomenon
→ 2–4 个自然解释 A/B/C
→ 每个解释有直接 controlled manipulation
→ 得到因素贡献和边界条件
```

这是最高优先级之一。

## Type B — 经典 NLP / cognitive / resource 问题 × LLM 时代的新 handle

代表：`r_han / r_tsujimoto / r_yano`。

不是：

```text
老 benchmark + 新模型
```

而是：

```text
旧问题本来难以观察 / 自动处理
→ LLM 的能力、表示、生成或规模让新的 measurement / intervention 成为可能
→ 重新回答旧科学问题
```

## Type C — 两个成熟因素 × 一个自然 interaction / budget

代表：`r_kisako`。

```text
因素 X 已经成熟
因素 Y 已经成熟
X × Y 的 interaction / trade-off 还没有被系统回答
→ 用一个自然约束 / budget / task family 统一比较
```

结果必须有结构，不能只是二维表。

## Type D — 自然社会 / 语言变化问题 × 可解释 proxy

代表：`r_utami`。

问题本身来自现实世界，不来自模型 failure。重点审 proxy、时间/群体对照和 alternative explanations。

## Type E — 自然 construct → operationalization → behavior → representation → causal role

代表：`r_hamdi`。

只有下面顺序才允许做机制：

```text
construct 有独立定义
→ 用 matched controls / manipulable variables operationalize
→ behavior / structural relation 先站住
→ representation evidence
→ causal intervention 回答原 construct 问题
```

禁止：

```text
想做 probe / SAE / patching
→ 找一个模型怪现象
→ 再给内部信号起一个科学名字
```

## Type F — 真实系统目标 / failure → controlled diagnosis

系统题必须围绕一个真实目标，而不是 benchmark 分数。至少回答以下两项：什么时候失败、为什么失败、如何针对性缓解。

---

# 3. 第一硬门槛：External-Construct / Model-Invariance Gate

这个 Gate **优先于 novelty、artifact、mechanism**。

候选进入 ACTIVE 前必须回答五个问题：

### EC1. Construct independent of model

研究对象能否独立于某个模型的偶发现象被定义？

如果只能写：

> “模型 X 在任务 Y 上出现 Z。”

而不能写出更一般的问题，默认降级。

### EC2. Operationalization independent of hidden-state evidence

在看 probe / SAE / activation 之前，能否明确：

- independent variable；
- dependent behavioral / structural measure；
- matched controls；
- competing predictions。

如果不能，说明我们可能仍是在从内部现象倒推 construct。

### EC3. Model swap test

至少设想 2–3 个不同 family 的模型。

如果换模型后 phenomenon 消失：

- 这个结果能否成为有意义的 boundary condition？
- 还是整个题目直接没有了？

后者不进 ACTIVE。

### EC4. Benchmark is instrument, not object

benchmark 应当只是测量 scientific construct 的工具。

如果研究问题等价于“解释这个 benchmark 的奇怪分数”，优先级低。

### EC5. Generality must be in the claim

最终 claim 应该尽量是：

```text
某种语言 / 学习 / 信息 / 认知因素如何影响模型
```

而不是：

```text
某个 checkpoint / 某个模型有某个性质
```

**EC1 或 EC2 明显失败：不注册。EC3 明显失败：至少降为 WATCH。**

---

# 4. Venue / literature policy

主 seed 搜索范围原则上只用 CCF A/B AI / NLP 会议：

```text
ACL                 第一优先
NAACL / EMNLP       第二优先
NeurIPS / ICML      补充
AAAI / IJCAI        补充
```

低优先级 venue 可以用于背景、collision、方法和 old-question anchor，也可以作为组内 research-shape 正例，但不能仅凭漂亮 gap 把新题升进 ACTIVE。

更重要的是：**以后不再主要搜索 “surprising LLM failure”。** 搜索顺序改成：

```text
外部科学 construct / old NLP problem
→ 经典文献中的 competing explanations
→ 找到可直接 operationalize 的变量与 manipulation
→ 近 2–3 年 ACL/EMNLP/NAACL/AI 顶会如何用 LM 研究它
→ 是否出现新的 measurement / data / model handle
→ exact collision
```

---

# 5. 资源画像

我们的约束不是“算力少”，而是：

```text
现金少
人工标注资源少
GPU 相对充足
```

因此优先：

- public dataset / structured resource；
- gold labels；
- released predictions / traces / checkpoints；
- open-weight models；
- synthetic / executable / programmatic labels；
- 现成语料或元数据能直接完成 first shot。

避免：

- 大量付费闭源 API；
- 几千条新人工 annotation；
- 必须先复现一个昂贵 upstream pipeline 才碰得到自己的问题。

GPU 用于：controlled training、cross-model confirmation、checkpoint analysis、probing / patching / SAE / steering 等**第二阶段深化**，而不是 sweep 出一个偶然现象。

---

# 6. 正式候选的 8 个硬 Gate

## G1 — External Construct + Natural Question

必须同时通过第 3 节 EC gate，并能用一句普通话解释“研究的是什么”。这是最重要的 Gate。

## G2 — Scientific Anchor + Venue Scale

必须锚定至少一个：经典 scientific question、成熟 NLP resource/task、真实语言/社会现象、成熟 engineering objective、真实 system failure。

最好结果要能自然长出 ACL/EMNLP/NAACL 尺度的 2–4 条 headline findings。

## G3 — Scientific Novelty

novelty 必须来自：新的因素分解、interaction/trade-off、old question 的新答案、更合理的 measurement、新的稳定规律或新的 system diagnosis。

换模型 / 换数据 / 第一次用某 interpretability method 不算核心 novelty。

## G4 — Competing Explanations / Interaction

进入 ACTIVE 前至少有：

```text
2–4 个自然 competing explanations
```

或者一个清楚的 interaction / budget axis。

如果唯一计划是“看看 hidden state 有什么”，不进 ACTIVE。

## G5 — Direct Behavioral / Structural G0 + Low Prerequisite Tax

健康路径：

```text
公开资源 / programmatic construction
→ 一个很轻的 sanity
→ 直接 behavioral / structural test 检验科学问题
→ 成立后再考虑 mechanism
```

危险路径：

```text
复杂复现
→ receipt A
→ relation B
→ eligibility C
→ probe / layer sweep
→ 才知道 scientific object 是否存在
```

Topic 25 是永久反面教材：昂贵 prerequisite 完整跑完，冻结 seed relation 失败，G0 根本不能开始；失败本身又没有足够科学信息增益。以后这种题搜索阶段就重罚。

## G6 — Artifact / Resource Executability

“论文说代码公开”不等于可执行。注册前要核对：数据字段、instance-level metadata、模型权重、scorer、eligible support、环境要求。

## G7 — Interpretable Null

G0 如果不支持假说，必须仍能回答一个 meaningful question 或排除一个 explanation。若失败只能得到“这个模型没出现”，这是坏题信号。

## G8 — Runway

最好结果出来后还要有自然下一步：因素边界、跨语言/跨模型 generality、机制、方法、资源改进或系统 mitigation。不能是“证明了，然后呢？”

---

# 7. Model-generality policy

以后所有模型行为类候选默认遵守：

- **发现阶段**可以先用一个便宜 open model；
- **晋级阶段**至少设计 2–3 个不同 family 的 confirmation；
- 如果主效应只在一个 family 上存在，除非这种 family difference 本身对应明确科学解释，否则降级；
- 不要求每个模型 effect size 相同，但要求研究问题跨模型仍然成立；
- model size / post-training / architecture 更适合作为 explanatory factor，而不是题目本身。

换句话说：

> **Cross-model variation 应该帮助解释问题，而不是决定问题是否存在。**

---

# 8. Mechanism 的位置

机制不是默认贡献，也不是“显得高级”的装饰。

只有下面情况才值得投入 GPU：

```text
external construct 已独立定义
+ operationalization 清楚
+ behavioral / structural object 已经稳定
+ competing explanations 需要内部证据才能区分
```

优先 causal intervention；probe / CKA / t-SNE 只能作为证据链的一部分。

如果 behavior 本身不稳定，禁止用更多层、更大 coefficient sweep、更多 probe 去把故事救回来。

同样，**representation 不存在并不自动 kill 一个好问题**：如果不同模型对同一外部 construct 表现不同，这可以是有意义的 model-family boundary condition。真正危险的是 construct 本身只能靠 representation 来定义。

---

# 9. Kill rules

以下任一明显成立，原则上直接 KILL / DOWNGRADE：

- 研究问题只能依赖某个模型/benchmark 的怪癖来定义；
- construct 在看 hidden state / probe 之前无法 operationalize；
- 换一个合理模型后，问题本身失去意义；
- 主 novelty 是换模型、换语言、换数据集或换 interpretability method；
- 要靠大量 closed API 或大规模新人工标注；
- first scientific test 前有多层昂贵 prerequisite；
- behavior 不成立后试图靠 layer/probe/SAE sweep 救故事；
- 需要越来越多 control 才能解释 claim；
- exact collision 已经把 mother question 做完；
- G0 null 只能说明“这个模型没出现”；
- 最好结果出来后没有自然下一步；
- benchmark 是研究对象，而不是 measurement instrument。

---

# 10. 固定搜索流程

以后每轮必须按这个顺序：

```text
Step 1   从组内 research shape / 经典领域问题生成 search lanes
Step 2   先定义 external construct，不看模型怪现象
Step 3   找 old literature / competing explanations
Step 4   写清 operationalization：variables / controls / predictions
Step 5   查 ACL/NAACL/EMNLP/NeurIPS/ICML 近年新 handle
Step 6   写一句 mother question
Step 7   做 model-swap thought experiment
Step 8   exact collision search
Step 9   artifact / prerequisite audit
Step 10  冻结最小 behavioral / structural G0
Step 11  只有 behavior 站住后才规划 representation / mechanism
Step 12  只有 G1/G3/G5/G6 明确通过才进 ACTIVE
```

搜索日志必须同时记录：为什么留下、为什么杀掉。不能只存 survivor。

---

# 11. Candidate Card

每个候选进入 `ACTIVE_CANDIDATES.md` 前必须填：

```text
Title / one-line question:
External construct:
Why it matters without mentioning a specific model:
Closest lab research shape:
Old scientific anchor:
Operationalization (IV / DV / matched controls):
Recent NLP/AI handle:
2–4 competing explanations or interaction axis:
Why changing model does not destroy the question:
Closest collisions:
Existing artifact:
Paid API requirement:
Human annotation requirement:
Prerequisite tax:
Frozen behavioral / structural G0:
Behavioral kill criterion:
Mechanism trigger (what behavior must hold first):
Possible representation / causal test:
Intervention purpose (what scientific question it closes):
Positive-result headline:
Null-result interpretation:
Full-paper runway:
Status:
```

任何一项写不清楚，都说明题还没成熟。

---

# 12. 当前原则的一句话版本

> **先找一个本来就值得研究、换模型也不会消失的问题；把它严格 operationalize；先看模型是否正确处理，再决定是否值得进入 representation、mechanism 和 intervention。**

而不是：

> **先找一个模型怪现象或内部信号，再想办法把它包装成问题。**