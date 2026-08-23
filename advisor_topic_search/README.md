# 导师向选题搜索

这个目录用于记录**面向导师审查标准、并以 ACL / NAACL / EMNLP / EACL / TACL 等 NLP 顶会为主要尺度的研究选题搜索**。

这里不是一个“想到什么就记什么”的脑暴目录，而是一套候选题过滤系统。我们的目标不是积累最多的题，而是持续留下少数满足以下条件的问题：

> **问题自然、外部锚点明确、只比前人多走一步、首轮实验简单可杀、正结果值得高兴、后面还有机制或方法口子，并且在我们的现实资源下能做。**

最重要的几条原则先写在最前面：

1. **研究室同门的方向是“好题如何长出来”的参考样本，不是领域边界。**
2. **题目的宽窄、novelty 和完整度要主动对齐 ACL / NAACL / EMNLP 主会与强 Findings 的论文尺度。**
3. **我们的资源不是“算力少”，而是“现金少、人工标注能力少、GPU 算力相对充足”。** 因此少用付费 API、少造人工标注数据，但可以积极利用本地开源模型做 hidden-state、probing、activation patching、steering、causal intervention、checkpoint trajectory 等机制实验。
4. **导师不只关心最新 LLM phenomenon，也在意老的科学问题能否在 LLM 时代得到新的、以前做不了的处理方式。** 旧问题 + 新实验轴是高优先级来源。
5. **机制分析不是为了显得深。** 必须先有真实、稳定、值得解释的 phenomenon；机制工具是解释这个 phenomenon 的手段，不是题目存在的理由。

---

# 1. 我们到底要找什么样的题

理想结构通常是：

```text
一篇已经站得住的 seed paper / 一个被重复观察到的真实现象 / 一个已有几十年的科学问题
        ↓
明确前人已经证明了什么、没有证明什么
        ↓
找到一个只往旁边走一步，但 scientific question 明显不同的下一问
        ↓
确认这个问题不依赖我们自己发明一个奇怪 construct 才成立
        ↓
找到现成数据 / gold labels / open models / checkpoints / traces / measurement
        ↓
设计一个简单、决定性的 G0，先确认 phenomenon 是否真的存在
        ↓
若为正，再利用充足 GPU 做 representation / mechanism / causal intervention
        ↓
最终形成一个 ACL/NAACL/EMNLP 尺度的完整故事
```

每个候选首先要能回答：

> **是哪篇论文、哪个已知现象、或者哪个旧科学问题，逼出了这个下一问？**

如果没有外部支点，就先不要注册为候选。

---

# 2. Venue target：题目要像 ACL / NAACL / EMNLP 的题

我们不只是要求“能写论文”，而是主动用 NLP 顶会的常见题目尺度校准候选。

## 2.1 合适的宽度

一个好的候选通常应当能用**一句清楚的问题**说完，同时又足以支持：

- 2–4 个模型 / 模型家族；
- 1–3 个互补数据设置或任务；
- 一个明确主现象；
- 一组有目的的分析 / 机制实验；
- 一个自然的方法、评测或理论 implication。

典型形状是：

```text
一个明确 phenomenon
+ 一个明确 scientific question
+ 一个令人信服的 cross-setting replication
+ 足够深的 mechanism / analysis
+ 一个自然 follow-up
```

而不是：

```text
“整个 LLM reasoning 的统一理论”          # 太宽
“某个 prompt 换一个词掉了 1.3 分”         # 太窄
“我们做了一个新的 benchmark，模型强弱排序” # 通常不够
```

## 2.2 Novelty 标准

我们追求的是**scientific novelty**，不是表面 configuration novelty。

以下通常不够：

- 换模型；
- 换语言；
- 换 benchmark；
- 把已有方法搬到一个相邻任务；
- “第一次在 X 数据上用 Y probe”；
- 文献矩阵里刚好有一个空格。

更像顶会的 novelty 是：

- 已有结果暴露了一个新的 dissociation；
- 一个长期问题第一次因为 open checkpoints / hidden states / causal intervention 而可直接检验；
- 两个本应一致的能力被证明系统性分离；
- 一个已知 behavioral failure 被定位到明确的 computation / representation bottleneck；
- 一个现有评价结论因为新的 diagnostic 被重新解释。

## 2.3 Positive-result excitement test

在写代码之前先假设最干净的正结果已经得到，然后问：

> **如果这是 ACL / NAACL / EMNLP 的标题和摘要核心结论，我会觉得“这确实改变了我对模型的理解”，还是只会说“嗯，合理”？**

如果最好的结果仍然只是“reasonable / expected”，降级。

---

# 3. 我们的真实资源画像：现金少、标注少、GPU 多

这是**promotion gate**，不是备注。

## 3.1 现金成本

候选的 phenomenon-existence G0 **不能依赖大规模付费 closed-model API 调用**。

资源优先级：

1. 已发布 response traces / logits / predictions / human judgments；
2. 公开 benchmark + 已有 gold labels；
3. 本地可跑 open-weight models；
4. 公开 intermediate checkpoints；
5. 程序化生成、可 exact scoring 的数据；
6. 少量 closed API 只用于最后的 spot-check / external generalization。

一个题如果必须先花大量 GPT / Claude / Gemini API 钱才能确认 phenomenon 是否存在，直接降级。

## 3.2 人工标注成本

候选**不能要求先新标几千条语义数据**才能开始。

优先：

- benchmark gold answers；
- executable / symbolic / numeric / exact-match labels；
- 公开 human-response logs；
- Wikidata / citation graph / revision history / structured metadata；
- deterministic transformations；
- 已经发布的 expert annotations。

允许在现象成立后人工核验少量高价值案例，例如 50–200 个 case；不允许把大规模 annotation 当 G0 前提。

## 3.3 GPU 是我们的优势，不要浪费

本地 GPU 充足，因此与其为了“便宜”只做 surface-level dataset analysis，我们反而应优先搜索**开源模型可做的机制型题**。

特别欢迎：

- hidden-state decoding / probing；
- layer-wise trajectory；
- logit lens / tuned lens；
- activation patching；
- causal tracing；
- attention / MLP ablation；
- steering / representation engineering；
- sparse feature / SAE / crosscoder（前提是 phenomenon 已站住）；
- dense pretraining checkpoint dynamics；
- SFT / RL / quantization 前后的 representation change；
- lightweight controlled finetuning 做 causal test。

但 GPU 多不代表可以无限 sweep。

仍然禁止：

```text
layer × step × threshold × model × prompt × dataset
```

海量搜索后挑最好看的 cell。机制结果必须从预先明确的 phenomenon 和 hypothesis 出发。

## 3.4 最理想的资源结构

```text
现成 dataset / gold labels
+ open model/checkpoints
+ automatic scoring
+ 本地 GPU 机制分析
= 最适合我们的题
```

---

# 4. 导师偏好的另一条主线：老问题 × LLM 时代的新实验能力

导师在意的不只是“2026 又出现了什么新 benchmark”，还在意：

> **一个以前就值得问的问题，到了 LLM 时代后，是否因为新的数据规模、训练轨迹、内部状态或可干预性，终于可以被更直接地研究？**

这类题是高优先级，但必须避免退化成“LLM 像不像人”。

## 4.1 合格的 old-question transfer

应该满足：

```text
旧领域本来就有一个明确问题 / competing hypotheses
        +
LLM 提供以前没有的 experimental handle
        =
新的科学检验
```

可能的旧问题来源包括：

- cognitive science：belief revision、anchoring、content effects、confidence、recognition vs recall；
- psycholinguistics / language acquisition：development order、composition、incremental processing；
- information science：citation / evidence propagation、revision、retrieval；
- decision-making：uncertainty、abstention、aggregation、confirmation / revision；
- memory research：interference、serial position、retrieval vs storage；
- learning theory：acquisition order、forgetting、compression、generalization。

LLM 时代真正提供的新 handle 可以是：

- dense training checkpoints；
- token-by-token / layer-by-layer hidden state；
- exact causal intervention；
- billions of naturally occurring examples；
- controlled generation；
- same architecture at different training stages / scales；
- open models可以重复运行同一个实验。

## 4.2 不合格的 old-question transfer

不要做：

> “人有 X bias，LLM 有没有？”

除非已有行为结果本身很反常，且下一步可以回答更具体的问题。

更好的形式是：

> “人类研究里 X 与 Y 是两个 competing explanations；LLM 的 training trajectory / hidden states 让我们第一次能直接区分它们。”

---

# 5. 同门研究：学习“题怎么长出来”，不要照抄领域

研究室同门只是正例。

## 5.1 Hamdi 型：已知现象 → 更基础的相邻问题

例：已有工作问“模型知道这个实体吗？”，下一步不是重复 known / unknown，而是问：

> 模型熟悉一个实体，和模型认为它真实存在，是不是两个不同状态？

另一个例子：已有工作发现随机选择有系统偏差，然后追问：

> 模型内部是否存在“当前任务要求 arbitrary/random choice”的状态？

核心结构：

```text
现象已经存在
→
追问 underlying distinction / state / causal variable
```

## 5.2 Kurauchi 型：成熟 axis × 重要 scientific object

```text
Paper A：新的可靠 axis（如 diffusion time）
+
Paper B：重要对象（如 compositional semantics）
→
对象在这个 axis 上如何形成？
```

其他可用 axis：

- pretraining checkpoint；
- post-training stage；
- model scale；
- context length；
- compression / quantization；
- causal intervention strength。

## 5.3 Tsujimoto 型：旧科学问题 × model development trajectory

不是因为 checkpoint 多所以画曲线，而是旧文献本来就有发展顺序 / competing hypothesis：

```text
human development age
→
model training checkpoint
```

## 5.4 Kisako 型：单轴规律 → 有现实意义的变量 interaction

例如 dimension 已经研究得很清楚，再问同样 storage budget 下：

```text
dimension × precision
```

但 interaction 必须对应真实问题，不是为了画二维图。

## 5.5 Yano 型：“模型会做” → “模型是否真正拥有背后的能力”

一个 benchmark score 很高，不代表模型拥有 task 假定的 latent capability。

可以把标签 / shortcut 拿掉，用更直接 diagnostic 测真正结构。

## 5.6 最高优先级：dissociation / contradiction

优先搜索：

```text
模型内部知道 X
但行为不使用 X
```

```text
模型能完成每个局部步骤
但组合后失败
```

```text
更多训练 / 更多证据 / 更强模型
反而让某个明确行为变差
```

```text
verbalized reasoning 正确
final decision 却不跟随
```

这类问题通常天然适合机制分析，也最容易形成令人兴奋的 headline。

---

# 6. 机制型题目的额外标准

因为我们 GPU 资源好，后续搜索会**主动偏向 mechanism-ready seed**，但必须遵守下面的顺序。

## 6.1 Phenomenon before mechanism

禁止：

```text
先训练 SAE / probe
→
看到一个 feature
→
再编故事说它是什么 phenomenon
```

正确顺序：

```text
已有 behavioral / representational anomaly
→
简单 G0 复现
→
明确需要解释的 decision point
→
probe / patch / ablate / steer
```

## 6.2 Representation ≠ causal use

“linear probe 可以读出来”最多说明 information is available。

如果 claim 是“模型使用了这个信息”，必须考虑：

- activation patching；
- targeted steering；
- causal mediation；
- ablation；
- matched intervention。

因此最喜欢的 seed 是：

> **representation 已经被发现，但 behavior 与它分离；前人没有回答 causal use。**

## 6.3 Mechanism-level phenomenon 必须真实存在

不要再重复 archived topic 的错误：aggregate score 看起来有结构，但真正要解释的 instance-level event 根本不存在。

在做 hidden-state 分析之前，必须确认：

- critical transition / reversal / error cell 有足够密度；
- 同一模型中有 clean positive / negative instances；
- 不是 parser / sampling / threshold artifact。

---

# 7. 最优先找什么 seed

以后扫 2025–2026 论文，优先级如下。

## A. 已经发现稳定 anomaly / subgroup / reversal

例如：

- 某能力内部可读但外部不会用；
- reasoning trace 已经识别问题但 final answer 不跟随；
- stronger model 反而在某个明确条件更差；
- more evidence 造成 correct → wrong；
- 局部能力全部存在但 global decision 失败。

## B. Paper 明确留下 causal-use gap

尤其是 limitation 写着：

> probing reveals representation but not whether it is used.

这种非常适合我们的 GPU 条件。

## C. 新 measurement axis 已经成熟

例如 dense checkpoints、diffusion time、training stage、quantization trajectory，然后把重要邻接对象放到这个 axis 上。

## D. 老领域有明确 competing hypothesis

LLM 只是让旧问题第一次拥有更好的 measurement / intervention。

## E. 作者已经发布代码、数据和 open-model setting

这是极大加分项，因为我们可以先复现，再只改一个变量。

---

# 8. 什么样的题快速杀掉

1. 为 identification 必须造不自然 counterfactual。
2. 只因为文献矩阵有空格就叫 gap。
3. 最好结果只是“X 会影响 accuracy”。
4. 必须先证明“不是 A、不是 B、不是 C……”才能说主结论。
5. control 越来越多才能维持 claim。
6. 主要贡献只是换模型 / 换语言 / 换 benchmark。
7. “模型越大越好”这类完全符合直觉的结果。
8. phenomenon existence 需要大规模新人工标注。
9. G0 需要大量付费 closed API。
10. 核心指标依赖昂贵 LLM-as-a-Judge 且没有可靠 automatic proxy。
11. 必须从头训练大模型才能看到对象。
12. 只有 toy regime 成立，meaningful regime 要靠不断 model/data/config fishing。
13. 正结果之后没有 mechanism / intervention / mitigation / method opening。
14. 为了做机制，先假设一个从未被证明存在的 latent object。

特别记住：

> **如果 gate 和 kill line 越设计越复杂，通常不是我们越来越严谨，而是问题本身越来越不自然。**

---

# 9. Candidate card：进入题池前必须写清楚

```text
题目：
Venue-scale headline：
Seed paper / old scientific question：
前人已经证明什么：
哪一个结果自然逼出下一问：
我们只移动哪一步：
一句话 Research Question：
为什么问题本身自然：
为什么正结果不是“果然如此”：
为什么尺度像 ACL/NAACL/EMNLP：
最近 exact collision：
同门 collision：
已有 dataset / labels / code：
可用 open models / checkpoints：
G0 第一枪：
Kill line：
Paid API requirement for G0：
New annotation requirement for G0：
Local GPU estimate：
Mechanism-ready?：
如果为正，下一步 mechanism：
如果为正，下一步 method / intervention：
最终最强 headline：
状态：
```

如果“哪一个结果自然逼出下一问”写不出来，通常说明题还是脑补的。

---

# 10. Promotion Gate

一个题只有同时通过下面的 gate 才值得真正写代码。

### G1. Naturalness
一句话就能理解问题为什么存在。

### G2. External anchor
有 seed paper、published anomaly、旧科学问题或稳定公开现象。

### G3. One-step distance
足够靠近 seed，可复用 setting；scientific question 又明显不同。

### G4. Venue-scale
不是过窄 trick，也不是无法收束的大题；能长成 ACL/NAACL/EMNLP 尺度完整故事。

### G5. Non-triviality
正结果不能只是常识。

### G6. Positive-result excitement
如果结果完美成立，真的值得高兴。

### G7. Cheap decisive G0
第一枪不依赖大量 API / annotation，且能快速显著改变我们对题目的信心。

### G8. Killability
核心结构不存在就直接停，不靠 tuning 续命。

### G9. Low control complexity
最好 1–3 个关键对照就足够解释主结果。

### G10. Existing data
核心对象基本已经存在。

### G11. Collision auditability
exact question / contrast / mechanism 可以明确检索。

### G12. Mechanism or method opening
现象成立后自然产生 causal explanation、intervention、mitigation 或新的 learning objective。

### G13. Resource fit
现金少和标注少不会卡死项目；GPU 优势能真正转化为实验深度。

### G14. Mechanism identifiability
如果要做机制，planned intervention 必须真的能区分 causal use，而不只是多画 hidden-state 图。

---

# 11. 标准搜索流程

## Stage 1：广泛扫 seed

重点：

- ACL / EMNLP / NAACL / EACL / TACL；
- 2025–2026 主会和 Findings；
- ICLR / ICML / NeurIPS 中与语言模型机制直接相关的工作；
- old cognitive / information / learning literature；
- appendix / error analysis / limitations，而不是只看 future work。

## Stage 2：每篇只写“真正证明了什么”

先不设计方法。

## Stage 3：只允许一步移动

优先：

```text
behavior → internal representation
representation → causal use
verbalized reasoning → final decision
local competence → global composition
final model → training trajectory
phenomenon → exact failure point
old hypothesis → modern causal test
single variable → meaningful interaction
```

## Stage 4：collision search

查：

- seed 作者自己的后续；
- 引用 seed 的 2025–2026 工作；
- exact wording；
- 同门；
- 本仓库 archived failures。

## Stage 5：resource audit

先确认：

```text
公开数据？
公开标签？
open models？
公开 checkpoint？
本地 GPU 能跑？
需要多少 API？
需要多少人工标注？
```

## Stage 6：才设计 G0

G0 的目标不是漂亮，而是：

> **最快判断这个 scientific object 是否真的值得后续机制分析。**

## Stage 7：phenomenon 成立后才上机制工具

GPU 用在这里，而不是用 sweep 把不存在的 phenomenon 挖出来。

---

# 12. 每轮搜索日志格式

每轮至少记录：

```text
本轮搜索范围：
本轮使用的 venue / paper pool：
Seed papers：
每篇已证明的核心事实：
旧科学问题（若有）：
自然下一问：
一步 extension：
资源检查：
机制分析可能性：
Exact collision：
同门 / archived collision：
保留候选：
降级候选：
杀掉候选及原因：
下一轮继续搜索的分支：
```

每个保留候选必须有完整 candidate card。

---

# 13. 当前总判断

研究室没有一个固定“只能做这些领域”的列表。真正稳定的偏好是：

> **一个清楚的外部 seed / 旧科学问题 → 一个自然的一步 extension → 一个具体可测对象 → 一个简单决定性实验 → 一个值得解释的现象 → 一个明确后续口子。**

结合我们的实际资源，当前应当特别偏向：

> **NLP 顶会尺度的问题 + 现成数据 + open model + automatic labels + GPU-heavy mechanism / causal analysis。**

以后判断一个题，不再只问：

> “这个方向热门吗？”

而要同时问四件事：

> **它为什么是一个自然问题？**
>
> **为什么现在还没被直接回答？**
>
> **如果证明出来，ACL/NAACL/EMNLP 的读者为什么会在意？**
>
> **我们能否不花大量 API / 标注钱，却用自己的 GPU 把它做得比普通 behavioral paper 更深？**

这四个问题共同构成这个目录现在的选题标准。
