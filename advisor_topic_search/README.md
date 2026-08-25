# 导师向选题搜索

这个目录用于记录**面向导师真实审查偏好、以 CCF A/B AI / NLP 会议为主搜索池、ACL 第一优先的研究选题搜索**。

这里不是“从最新论文里找一个 gap，然后想办法套一个 probe / SAE / activation patching”的目录，而是一套**候选题过滤系统**。目标不是积累最多的题，而是持续留下少数：

> **问题本身自然且值得问；研究对象已经真实存在；实验能直接区分少数几个有意义的解释；第一轮便宜、清楚、可杀；最好的结果足以形成 ACL/EMNLP/NAACL 尺度的 headline；在我们的资源下真的做得出来。**

当前候选状态统一看：

```text
advisor_topic_search/ACTIVE_CANDIDATES.md
```

`ROUND_*.md` 保存每轮搜索、collision audit、升降级和失败历史；`g0/` 只放真正决定候选生死的 prerequisite / first-shot experiments。

---

# 0. 最重要的原则

1. **先问“这是一个好科学问题吗”，再问“能不能做机制”。** mechanism 是一种可能的深化方式，不是选题起点，也不是默认贡献。
2. **组内已经做成论文/成熟项目的题，是最高优先级的选题 calibration。** 它们不是领域边界，但比“网上看起来像 ACL 的题”更能说明导师认可什么问题尺度、实验形状和 novelty。
3. **实际可行性优先于题目看起来漂亮。** 这里的可行性不仅是 GPU 能不能跑，还包括 time-to-scientific-question、prerequisite chain depth 和失败后的 information gain。
4. **导师关心“证明出来以后是否值得高兴”。** 最好结果如果只是“嗯，合理”，降级；如果一个反直觉结果本身就足够改变理解，即使没有复杂机制，也可以是好题。
5. **最好只赌一个真正的科学问题。** 不要同时赌 seed 能复现、open model 有现象、measurement 有效、critical cell 足够、mechanism 也成立。
6. **controls 应该服务于 competing explanations，而不是维持一个摇摇欲坠的 claim。** gate 和 controls 越写越复杂，通常说明题本身越来越不自然。
7. **题目/标题最好让人一眼看到研究对象和问题。** 不要把 method 名、benchmark 名或“mechanistic analysis”放在问题前面。
8. **主 seed venue 有硬约束。** ACL 第一优先；NAACL / EMNLP 其次；NeurIPS / ICML / AAAI / IJCAI 中与 NLP/LLM 直接相关的工作可进入主池。低优先级 venue 只做背景、collision、方法参考。
9. **我们的资源画像是：现金少、人工标注能力少、GPU 相对充足。** 少 API、少新标注；open-weight + controlled training / representation analysis / causal intervention 可以重 GPU。
10. **“分析型论文”完全允许。** 如果问题自然、变量清楚、结果系统且有解释力，不要求强行升级成 hidden-state causal mechanism。

---

# 1. 组内研究正例：导师真正接受的题是怎么长出来的

这一节来自实验室 Slack 中 `r_xiang / r_han / r_hamdi / r_kisako` 等研究频道，以及 `r_utami / r_sato / r_tsujimoto / r_yano` 等成熟项目。这里记录的不是成员履历，而是**选题生成规律**。

> 重要：组内正例用于校准“问题长什么样”和“导师会追问什么”，不改变本目录的主 seed venue policy。某个组内项目即使发表于本目录不作为主 seed 的 venue，也仍然可以作为 research-shape 正例。

## 1.1 `r_han` — 经典语义资源 × LLM 时代的新问题

代表性方向：**Definition Generation for Semantic Frames with In-Context Learning**，以及 FrameNet frame relation 的自动补全/估计。

真正的问题不是“LLM 能不能做一个 FrameNet task”，而是：

> 人工构建的、可解释的 frame-semantic knowledge，LLM 是否已经隐式掌握？如果要生成/补全这种知识，哪些信息与 demonstration 真正有用？

研究形状：

```text
成熟旧资源 / 旧问题（FrameNet）
→ LLM 时代出现新的生成与内化能力
→ 自然的新问题
→ random vs similar demonstrations / relation completion / hard negatives
→ frame-level generalization 与 leakage control
```

**要学的东西**：

- “old NLP problem/resource × LLM”本身可以是很强的题，不需要硬做机制；
- 关键是说明为什么这个旧问题在 LLM 时代仍然重要，而不是“因为以前有人做 FrameNet”；
- generalization unit 必须和 scientific claim 一致，例如按 frame 而不是按 example 随机切分；
- reviewer/advisor 会追问：这和已有 frame identification / WSD / resource construction 到底有什么本质区别？

## 1.2 `r_hamdi` — 一个简单自然 construct，先隔离，再做到 causal

代表性方向之一：**real vs fictional / ontological status 的内部表示**。

好题的核心不是“做一个 probe 看真实/虚构能不能分类”，而是：

> 模型是否真的把“现实存在性”作为一个独立于 familiarity / entity knowledge 的属性表示？这个表示是否因果影响最终判断？模型与人的未知实体判断是否存在系统差异？

研究形状：

```text
非常容易理解的自然问题
→ familiarity-matched / domain-matched controls
→ cross-domain generalization
→ representation evidence
→ causal intervention
→ behavior–representation dissociation
→ rare / unknown cases 作为更强的一步
```

另一个方向把“随机/任意选择”拆成可读出的 internal mode 与可写入的 control direction，再用 gated intervention 验证 specificity。

**要学的东西**：

- 如果做 mechanism，先把 construct 隔离干净；
- probe 不是终点，causal intervention 应回答原问题，而不是为了“更机制”而加；
- famous/easy cases 得到预期结果不够，导师会继续追问 rare / unseen / counterfactual cases；
- “技术上更复杂/几何上更新”不等于 payoff 更高。一个 nonlinear geometry 结果如果只是“果然比 linear 更复杂”，不一定是好题。

## 1.3 `r_kisako` — 两个成熟因素 × 一个自然统一轴，也可以成为完整论文

当前方向：**text embedding 的 dimensionality reduction × quantization interaction**。

核心不是发明新的 compression algorithm，而是把两个成熟压缩手段放到统一 storage budget 下：

```text
storage ≈ bits × dimensions
```

然后跨 classification / clustering / retrieval / STS 等任务系统回答：

> 在固定存储预算下，应该把容量花在更多维度还是更多 bit？dimension reduction 与 quantization 是否存在 task-dependent interaction？

过程中曾出现非常反直觉的低维低 bit 结果，导师的第一反应不是“赶紧写机制”，而是：**如果这个结果是真的，而且能解释，单这个现象都很有价值。** 随后优先排查 normalization / quantizer / PCA whitening 等 artifact。

**要学的东西**：

- mechanism 不是必需条件；**自然统一轴 + 系统 interaction + 反直觉 finding + 实用结论**本身可以够强；
- 标题应直接让人看到研究对象，例如“次元削減と量子化の相互作用”，而不是方法名堆砌；
- 一个怪结果首先要做 artifact audit，不能把 bug 当 phenomenon；
- 好分析题应能回答用户/研究者真实会问的 trade-off，不只是画很多曲线。

## 1.4 `r_utami` — 自然社会/语言问题 × 简单可解释 proxy

EMNLP 2026 Main 方向：**The Impact of LLMs on Native Language Signals in English Academic Writing**。

问题一句话就能理解：

> 随着 LLM 广泛进入学术写作，作者母语留下的 English writing signal 是否正在减弱/同质化？

做法并不依赖复杂机制，而是比较 pre-neural / pre-LLM / post-LLM 等时期，用 L1 detectability 作为 measurable proxy，并辅以 LLM rewriting experiment。

**要学的东西**：

- 一个非常自然、读者本来就关心的问题，可以比“最新模型内部某个 feature”更有论文价值；
- method 可以简单，但 proxy 必须能对应原问题；
- 需要主动控制 topic distribution、时代变化、数据来源等 alternative explanations；
- 导师会反复要求把标题/开头写成“问题本身”，而不是“我们做了一个分类实验”；
- story clarity 比“把所有做过的实验都塞进去”重要。

## 1.5 `r_sato` — 已知现象 → competing factors → controlled experiment

代表性方向：**How Do Language Models Acquire Character-Level Information?**

起点是一个非常明确的现象：

> subword-based LM 并没有显式 character-level input unit，却仍能保持 character-level information；这种信息到底从哪里来？

研究没有先去找 hidden-state circuit，而是系统控制三类变量：

```text
pretraining data
+ tokenizer
+ classifier train/eval data
```

再把候选因素拆成 segmentation-induced 与 segmentation-independent 两类，分别考察 merge rules、orthographic/phonological regularity、character bigram、string length、syntactic information 等。

**要学的东西**：

- 这是最值得模仿的“分析型机制”形状：**自然 phenomenon → 2–4 个 candidate explanations → 每个 explanation 都有直接 controlled experiment**；
- “机制”不等于 activation patching。控制数据生成过程、tokenizer、训练分布，本身就是 causal/diagnostic handle；
- 导师会主动把“解明 mechanism”改成更克制的“分析”，如果证据不足以支持 100% mechanism claim；
- 如果结果可能只对特定 corpus 成立，就把它写成 limitation，不要靠额外大 sweep 假装 universal；
- 组内反馈里这个项目的**研究生长过程本身**被明确评价为很值得肯定：先观察差异，再不断提出并排除具体因素。

## 1.6 `r_tsujimoto` — 旧概念体系 × 自动诱导；以及 goal-first 的实用问题

成熟方向之一是 **semantic frame induction**：

```text
FrameNet / human semantic analysis
→ LM representation / metric learning / clustering
→ 自动诱导已有与潜在新 frame
→ quantitative alignment + qualitative novel clusters
```

后续也在探索 program verification / automatic test-case generation 一类可程序验证的实用问题。

**要学的东西**：

- 旧的语义资源/认知概念可以作为 scientific anchor，而不是被“LLM 已经很强”自动淘汰；
- practical topic 的标题也应以**目标/问题**为中心，例如“为了更高效地定位 bug，如何生成测试用例”，而不是“方法 A 与 B 比较”；
- concrete examples 是研究讨论的一部分，不只是 presentation decoration。

## 1.7 `r_yano` — 把“人隐式理解什么”变成比 label prediction 更本质的评价

当前成熟方向：**FrameBench / frame-semantic language understanding**。

核心问题不是让 LLM 预测一个预定义 frame label，而是：

> 不告诉模型 FrameNet label/definition 时，LLM 能否像人一样，从上下文中隐式区分 frame-semantic distinctions？

这一步把“resource label prediction”重新定义成更接近 human implicit semantics 的自然语言理解问题，并通过人工 sanity check、generator generalization、qualitative difficult/easy cases 等确保 benchmark 真正在测目标 construct。

**要学的东西**：

- 好 novelty 经常不是“新模型/新方法”，而是**把旧 task formulation 改成更接近真正 scientific question 的 measurement**；
- benchmark paper 也可以很强，但必须让 task formulation 本身回答一个明确问题，而不是再做一个 leaderboard；
- 与已有 WSD / frame identification 的区别必须从**研究目的**解释，而不是只说 output format 不同；
- reviewer 要一个额外 generator/control 时，如果它直接验证 benchmark construct 的 robustness，就值得做；这和 Topic 25 那种昂贵 prerequisite tax 完全不同。

## 1.8 `r_xiang` — concrete agent failure / safety setting

当前方向之一是**对话型 retrieval agent 在 adversarial input 下的 safety evaluation**：normal vs adversarial tasks，观察 answer correctness、unsafe behavior / query、safe refusal preservation、retrieval steps 等。

这是早期方向，不能像已发表项目一样当“成功模板”，但它说明另一种导师可接受形状：

```text
真实系统对象（retrieval agent）
+ 明确 failure setting（adversarial input）
+ 可解释行为指标
+ controlled normal/adversarial contrast
```

如果后面能找到一个稳定且非平凡的 failure mechanism / mitigation，它可以成长；如果只是“攻击后分数下降”，则不够。

---

# 2. 从组内正例抽出来的“导师可接受题型”

以后找题不能只找 mechanism gap。至少优先考虑下面六类：

### Type A — 已知自然现象 → 因素分解

代表：`r_sato`。

```text
模型有一个已知但解释不充分的能力/现象
→ 提出少数几个可竞争解释
→ 控制训练数据 / tokenizer / input / evaluation distribution
→ 逐项验证哪些因素真的贡献
```

### Type B — 经典 NLP / cognitive / resource 问题 × LLM 时代的新能力

代表：`r_han / r_tsujimoto / r_yano`。

不是“老任务用新模型再跑一遍”，而是：

> LLM 出现后，旧问题里以前无法直接问的部分是否现在可测？旧 task formulation 是否已经不再对应真正的 scientific question？

### Type C — 两个成熟因素 × 一个自然 interaction / budget axis

代表：`r_kisako`。

```text
成熟因素 X
×
成熟因素 Y
→ 以前各自研究很多，但 interaction / trade-off 没被系统回答
→ 用一个自然预算或约束统一比较
```

这种题**不需要强行做机制**，但必须得到可解释、task-dependent、最好有反直觉的结构性 finding。

### Type D — 一个自然社会/语言变化问题 × 可解释 measurable proxy

代表：`r_utami`。

重点是问题本身重要、proxy 对应问题、alternative explanations 控得住，而不是模型技术多新。

### Type E — 自然 construct → representation → causal behavior

代表：`r_hamdi`。

只有 construct 已经通过 matched controls 被隔离，才进入 probe / steering / causal intervention。不能从“我想做 activation patching”倒推题目。

### Type F — concrete system failure → controlled failure analysis / mitigation

代表：`r_xiang` 的 agent safety 方向。

必须从真实系统 failure 出发；单纯 benchmark score drop 不够，最好能找到**什么时候失败、为什么失败、怎么有针对性修复**。

---

# 3. 导师反馈中反复出现的写题/做题偏好

## 3.1 Problem first，标题和第一页就要能想象研究对象

优先：

```text
X 为什么发生？
X 与 Y 如何交互？
LLM 是否真正拥有 Z？
随着 LLM 时代到来，旧现象 Z 是否改变？
```

而不是：

```text
A Mechanistic Analysis of ...
A Comprehensive Benchmark of ...
Using Method X for Task Y
```

## 3.2 一个 paper 最好能浓缩成 2–4 条 headline findings

常见健康形状：

```text
1. 先确认主现象
2. 拆出最重要的因素 / dissociation
3. 用一个关键 control 排除最自然 alternative explanation
4. 再走一步得到 surprising implication / practical rule / causal intervention
```

不是实验越多越好。

## 3.3 “反直觉”很值钱，但先排 artifact

如果出现违反直觉的结果，优先级上升；但必须先查：

- scoring bug；
- normalization；
- tokenizer；
- data leakage；
- sample-size / distribution；
- generator / judge bias。

确认不是 artifact 后，再考虑把它升为 paper headline。

## 3.4 Competing explanations 比“多做几个 setting”重要

好的 control 能直接区分 A/B explanation；坏的 control 只是让 appendix 更长。

如果一个 claim 需要不断追加“不是 A、不是 B、不是 C、不是 D”才能成立，通常说明题目本身不干净。

## 3.5 Claim 要和证据强度一致

可以写“analysis / factors / evidence”，不必为了显得高级就写“mechanism”。

只有在 intervention 真能区分 causal explanation 时，才把标题和主张升级到 mechanism / causal role。

## 3.6 先让读者知道“为什么这事重要”，再展示方法

尤其是 old NLP resource / cognitive question：必须解释为什么在 LLM 时代仍值得问。不要默认“以前有人做，所以现在也值得做”。

---

# 4. Venue policy

主 seed 默认优先级：

```text
ACL
>
NAACL / EMNLP
>
NeurIPS / ICML / AAAI / IJCAI 中与 NLP / LLM 直接相关的工作
```

原则上只从 **CCF A/B 的 AI / NLP 会议**晋级主 seed。

其他 venue 可以用于背景、exact collision、方法参考、old-question anchor、组内 research-shape calibration，但不能单凭一个漂亮 gap 进入 TOP_POOL。

**ACL 是第一优先级。** 优先挖 main paper 的 anomaly、error analysis、appendix、unexpected finding，而不是从 Future Work 里生造题。

---

# 5. 资源画像：现金少、标注少、GPU 多

## 5.1 现金成本

phenomenon-existence G0 **不能依赖大规模付费 closed-model API**。

优先级：

1. released predictions / traces / logits；
2. public benchmark + gold labels；
3. local open-weight models；
4. public checkpoints；
5. programmatic / executable exact labels；
6. closed API 只做少量 external check。

## 5.2 人工标注

第一枪不能要求先新标几千条数据。

允许 phenomenon 成立后人工核验少量高价值 case；不允许把大规模新 annotation 当 prerequisite。

## 5.3 GPU 是优势

现象站住后欢迎使用：

- controlled SFT / continued pretraining；
- dense checkpoint trajectories；
- hidden-state probing / decoding；
- logit/tuned lens；
- activation patching / causal tracing；
- attention / MLP ablation；
- steering / representation engineering；
- SAE / crosscoder；
- quantization / compression / training-stage analysis。

但 GPU 用来**回答已经存在的问题**，不是大 sweep 把一个不存在的 phenomenon 挖出来。

## 5.4 Artifact completeness

TOP_POOL 尽量确认：

```text
data
+ exact model/checkpoint
+ prompt / scoring recipe
+ reproduction code
```

如果缺了两项以上，除非 G0 本身极其简单，否则降级。

---

# 6. 新增硬规则：Prerequisite Tax / Information Gain

Topic 25 是必须长期保留的反面教材。

Topic 25 最终不是工程半成品，而是**完整 receipt 后的科学 stop**：

```text
Qwen3-8B-Think gold-only = 0.45746
Qwen3-8B-Think noisy      = 0.35179

frozen requirement:
noisy thinking >= thinking gold-only

result: FALSE
→ SEED_RELATION_NOT_REPRODUCED
→ G0 NOT RUN by protocol
```

工程上的 model alias / hostname 问题最后都被排除，最终 receipt 完整。因此教训不是“服务器坑”，而是：

> **真正的 scientific question 前面放了一个昂贵、脆弱、信息增益很低的 seed-reproduction gate。**

以后每个候选在注册前必须问：

### 6.1 Time-to-scientific-question

从 `git clone` 到**真正第一次检验我们自己的 scientific question**，中间要经过多少工作？

如果要先：

```text
重建 upstream environment
→ 复现复杂 baseline
→ 复现多个 cell
→ 满足一个 seed relation
→ 才能跑我们自己的 G0
```

这是高危题。

### 6.2 Prerequisite chain depth

理想：

```text
released artifact
→ one-model sanity receipt
→ our G0
```

危险：

```text
upstream reproduction A
→ relation B
→ panel C
→ eligibility D
→ our G0 E
```

每增加一个必须成立的前置关系，survival probability 都会乘法下降。

### 6.3 Failure information gain

最重要的问题：

> **如果 prerequisite 失败，我们学到了什么？**

如果答案只是：

> “seed relation 在这个环境/模型没复现，所以停止。”

而且为此已经花了很多 GPU / 工程时间，这种题要在 search 阶段重罚。

更好的 G0 即使为负，也应该能区分科学解释，例如：

```text
A 不成立，但 B 成立
→ 已经说明 failure 属于 encoding 而不是 readout
```

### 6.4 Prerequisite cost 应显著小于核心实验成本

不要求所有 receipt 都免费，但**不能让“证明有资格做 G0”本身成为项目里最贵的实验之一**。

一个健康候选最好能在单模型、单数据集、固定 prompt/scoring 下快速确认 prerequisite；如果 receipt 本身接近一篇 reproduction project，直接降级。

### 6.5 新 kill rule

出现以下任一项，默认不进 ACTIVE：

- 真正 scientific question 前有两层以上脆弱 prerequisite；
- seed 的关键 relation 没在我们将使用的 exact open model 上报告；
- receipt 比我们的 G0 更贵/更复杂；
- prerequisite 失败时只能得到 `NOT_REPRODUCED`，没有新的科学区分；
- upstream engineering / environment reconstruction 占了主要工作量；
- 必须满足一个 aggregate relation 才允许进入 instance-level question；
- 需要“先复现论文整张表”才能开始自己的研究。

---

# 7. 理想题目长什么样

更接近组内正例的健康流程：

```text
自然 phenomenon / old scientific question / real system problem
        ↓
一句话说明为什么值得问
        ↓
列出 2–4 个最自然 competing explanations / interacting factors
        ↓
确认数据、模型、标签、关键对象基本可得
        ↓
设计一个直接区分解释的 cheap G0
        ↓
得到清楚 finding
        ↓
必要时再做 mechanism / intervention / method
        ↓
形成 2–4 条 headline findings
```

而不是：

```text
最新 paper 有一个 gap
→ 先复现 paper
→ 找 latent object
→ 扫 layer
→ patch
→ 再想 headline
```

每个候选首先回答四句话：

> **1. 普通 NLP/ML 研究者为什么会自然问这个问题？**
>
> **2. 组内哪个成功/成熟项目的 research shape 最接近它？**
>
> **3. 最自然的两个 competing explanations / interacting factors 是什么？**
>
> **4. 最便宜的实验能不能直接让其中一个解释失去可信度？**

四个答不出来，不注册。

---

# 8. Venue-scale、novelty 与 interestingness

## 8.1 合适宽度

典型 ACL/EMNLP/NAACL 尺度：

```text
一个自然问题
+ 一个明确 experimental handle / measurement
+ 2–4 条主 finding
+ 少量关键 controls
+ analysis / mechanism / intervention 中至少一个自然深化
```

不要：

```text
整个 LLM reasoning 的统一理论
某 prompt 换词掉 1.3 分
新 benchmark 排一遍模型
已有 phenomenon 上第一次用 SAE
```

## 8.2 Scientific novelty，不是 configuration novelty

更好的 novelty 包括：

- 旧 task formulation 在 LLM 时代不再测真正想问的东西；
- 已知 phenomenon 的几个 plausible causes 第一次被直接 disentangle；
- 两个成熟因素存在此前未系统研究的 interaction；
- 一个本应稳定的能力在自然条件下出现反直觉 reversal；
- internal representation 与 behavior 稳定 dissociate，并且 causal test 能区分解释；
- 社会/语言现象在 LLM 时代发生可测的结构性变化；
- practical system failure 被定位到明确条件/瓶颈，而不是只报告性能下降。

## 8.3 Positive-result excitement test

先假设最理想结果出来，再问：

> **如果这是论文标题/摘要第一句，我会觉得“原来是这样”，还是“这不是废话吗”？**

“原来是这样”可以来自 causal mechanism，也可以来自一个强 interaction、一个反直觉系统规律、一个新 measurement 或一个 old-question 的清楚答案。

---

# 9. 机制题额外规则：机制是深化，不是身份证

## 9.1 Phenomenon before mechanism

只有在下列之一已成立时才进入 mechanism：

```text
稳定 behavioral anomaly
稳定 representation–behavior dissociation
明确 competing causal explanations
可程序化 intermediate state
```

禁止：

```text
先 SAE/probe
→ 看见 feature
→ 再编 scientific question
```

## 9.2 Representation ≠ causal use

linear probe 只能说明 information available。

如果 claim 是“模型使用了它”，必须考虑 natural matched intervention、patching、ablation、targeted steering 等 causal evidence。

## 9.3 Bounded search

layer / token / strength / threshold 的选择要能在 validation 上冻结；test 不允许大 sweep 后挑最好看的 cell。

## 9.4 Interpretable null

如果 intervention 无效，只能说“可能没找到正确 layer”，说明设计不可证伪。

---

# 10. 什么样的题快速杀掉

1. 从 method 出发而不是从问题出发；
2. 只是“已有 phenomenon + 新 probe/SAE/patching”；
3. 为 identification 必须造不自然 counterfactual；
4. 最好结果只是“X 会影响 accuracy”；
5. controls 越来越多才能维持 claim；
6. 主要 novelty 是换模型 / 语言 / benchmark；
7. 最强结果完全符合直觉且没有新的结构性解释；
8. phenomenon existence 依赖大量 paid API；
9. 第一批 useful data 需要大规模新人工标注；
10. 核心指标依赖昂贵 LLM judge 且无 automatic proxy；
11. 必须从头训练大 foundation model；
12. public artifact 只覆盖 toy regime，meaningful regime 靠 fishing；
13. 只有 aggregate gap，没有可解释的 instance-level object；
14. seed anomaly 只在 inaccessible closed model 出现；
15. exact question 已 crowded，只剩“再做一次 mechanism”；
16. planned causal contrast 一次改变很多因素；
17. bug/artifact 没排干净就开始写 story；
18. 题目只能写成方法名，无法用普通语言说明研究对象；
19. **prerequisite tax 高，真正 scientific question 离开工太远；**
20. **prerequisite 失败时 information gain 很低；**
21. **需要复现整篇 seed paper 才有资格跑自己的第一枪。**

特别记住：

> **Topic 25 式失败要提前发生在 search audit，而不是发生在几天工程之后。**

---

# 11. Candidate card

进入 ACTIVE/HOLD 前至少写清楚：

```text
题目：
一句话 natural research question：
最接近的组内 research shape：
Venue-scale headline：
Seed paper / old scientific question / real system problem：
Seed venue（CCF A/B?）：
前人已经证明什么：
哪一个结果自然逼出下一问：
为什么这个问题在 LLM 时代更值得/更可测：

Competing explanation A：
Competing explanation B：
（必要时 C）：
最便宜能区分 A/B 的实验：

最好正结果为什么令人兴奋：
最好负结果能学到什么：
2–4 条可能 headline findings：
最近 exact collision：
同门 / archived collision：

Artifact completeness：
- released dataset：
- exact model/checkpoint：
- prompt/scoring recipe：
- reproduction code：

Time-to-scientific-question：
Prerequisite chain depth：
Prerequisite failure information gain：
Receipt cost vs G0 cost：
Same-model prerequisite：
Critical-cell definition（若需要）：
Critical-cell reported density：

Paid API requirement：
New annotation requirement：
Open-weight availability：
Estimated GPU-hours：
Storage / multi-node requirement：
Researcher degrees of freedom：

Mechanism really necessary?：
如果需要 mechanism，哪一个 causal distinction：
Mechanism search space 如何冻结：
如果为正，下一步 intervention / method：
Kill line：
状态：
```

如果“最接近的组内 research shape”只能回答“没有，但 ACL 有一篇差不多”，要额外谨慎。

如果 competing explanations 写不出来，而唯一计划是“看看 hidden state”，不进 ACTIVE。

---

# 12. Promotion Gates

只有同时大体通过才值得认真写代码：

- **G1 Naturalness**：不用术语也能解释为什么值得问；
- **G2 Lab-shape calibration**：能说明它接近组内哪种成功研究形状，而不是只像网上某篇 paper；
- **G3 External anchor**：seed / anomaly / old problem / real system failure 明确；
- **G4 Venue-scale**：能自然长成 2–4 条 ACL/EMNLP/NAACL 级 finding；
- **G5 Scientific novelty**：不是 configuration/method novelty；
- **G6 Positive-result excitement**：最强结果真的值得高兴；
- **G7 Competing explanations**：至少有两个可区分的自然解释/因素；
- **G8 Direct G0**：第一枪直接碰 scientific question，而不是先做长 reproduction；
- **G9 Low prerequisite tax**：receipt 简单、短、失败也有信息；
- **G10 Killability**：核心结构不存在就停；
- **G11 Low control complexity**：最好 1–3 个关键 control；
- **G12 Existing object**：数据/对象基本已存在；
- **G13 Resource fit**：少 API、少标注，GPU 能发挥；
- **G14 Artifact completeness**：data + model + prompt/scoring + code 尽量齐；
- **G15 Same-model phenomenon**：需要机制时，exact failure 在目标 open model 上真实存在；
- **G16 Collision auditability**：exact / near-exact 可检索；
- **G17 Analysis/mechanism/method opening**：正结果后有自然下一步，但不强迫 mechanism；
- **G18 Bounded degrees of freedom**：不能靠 model×prompt×layer sweep fishing；
- **G19 Interpretable null**：null 也能缩小解释空间；
- **G20 Venue eligibility**：主 seed 符合当前 CCF A/B AI/NLP policy。

其中 **G1 / G7 / G8 / G9** 任何一个明显失败，原则上不注册正式 candidate。

---

# 13. 标准搜索流程

## Stage 0：先看组内正例，不要先搜 arXiv

新一轮搜索前先重新问：

```text
我们缺的是哪一种 research shape？
- phenomenon→factor decomposition？
- old question→new LLM measurement？
- interaction/budget analysis？
- longitudinal/social change？
- clean representation→causal behavior？
- real system failure？
```

## Stage 1：广泛扫 seed / old question / real phenomenon

ACL 优先，其次 NAACL/EMNLP；AI 顶会补充。还要查经典 NLP/cognitive/information-science literature、公开 benchmark error analysis、工程系统中稳定 failure。

## Stage 2：每篇只写“真正证明了什么”

先不设计 probe/patch。

## Stage 3：提出 competing explanations / interaction

如果没有两个自然解释，也没有一个自然 interaction axis，先不要写机制。

## Stage 4：collision search

查 seed 作者后续、exact wording、2025–2026 相邻工作、组内项目、本仓库 archived failures。

## Stage 5：prerequisite-tax audit

必须先写：

```text
到我们的第一条 scientific result 之前要做几步？
哪一步最可能失败？
失败后得到什么信息？
receipt 要多久？
是否比 G0 本身更贵？
```

## Stage 6：artifact + resource audit

确认数据、labels、exact model、prompt/scoring、code、GPU/storage、API、annotation。

## Stage 7：才设计 G0

G0 的目标是：

> **用最短路径改变我们对某个 scientific explanation / interaction 是否成立的信心。**

不是“证明我们有资格继续做下一层实验”。

## Stage 8：结果决定 paper shape

- 如果 factor decomposition 已经形成强 finding：继续 analysis；
- 如果存在 clean latent/behavior dissociation：再上 mechanism；
- 如果出现 robust counterintuitive interaction：先解释与扩展；
- 如果 practical rule 已很强：可以走 system/method；
- 不为了显得高级强行机制化。

---

# 14. 文件组织

```text
README.md
    唯一总筛选标准 + 组内 research priors

ACTIVE_CANDIDATES.md
    唯一当前候选状态表

ROUND_*.md
    每轮搜索与审核历史

g0/
    prerequisite / killable first-shot scripts
```

每轮结束必须：

1. 写 Round log；
2. 同步 `ACTIVE_CANDIDATES.md`；
3. 如果从新失败/组内反馈中学到新的长期规则，更新 README；
4. 不允许一个候选只因为“我已经写了很多代码”而降低 kill threshold。

---

# 15. 当前总判断

以后选题的默认顺序应该是：

```text
先找自然问题
→ 看组内成功项目里哪种 research shape 能承载它
→ 找 old scientific anchor / ACL-level external seed
→ 写 competing explanations / interaction axis
→ 做 prerequisite-tax audit
→ 用最短 G0 直接碰科学问题
→ 结果足够强时再决定走 analysis、mechanism 还是 method
```

当前最应该避免的旧习惯：

```text
ACL 有篇新 paper
→ 找 limitation/gap
→ 假设 open model 也有
→ 先复现复杂 seed
→ 再找 hidden representation
→ patch 一遍
→ 最后才问这件事到底值不值得知道
```

最理想的项目不是“什么都新”，也不是“机制最复杂”。

最理想的是：

> **一个任何 NLP/ML 研究者都能理解的自然问题；已有现象或旧科学问题把它钉住；实验直接区分少数几个解释；第一枪很快；结果无论正负都推进理解；如果需要，GPU 让我们比普通 behavioral analysis 再深一层。**
