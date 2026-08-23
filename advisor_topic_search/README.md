# 导师向选题搜索

这个目录用于记录**面向导师审查标准、并以 ACL / NAACL / EMNLP / EACL / TACL 等 NLP 顶会为主要尺度的研究选题搜索**。

这里不是一个“想到什么就记什么”的脑暴目录，而是一套候选题过滤系统。我们的目标不是积累最多的题，而是持续留下少数满足以下条件的问题：

> **问题自然、外部锚点明确、只比前人多走一步、首轮实验简单可杀、正结果值得高兴、后面还有机制或方法口子，并且在我们的现实资源下真正做得出来。**

最重要的几条原则先写在最前面：

1. **研究室同门的方向是“好题如何长出来”的参考样本，不是领域边界。**
2. **题目的宽窄、novelty 和完整度要主动对齐 ACL / NAACL / EMNLP 主会与强 Findings 的论文尺度。**
3. **我们的资源不是“算力少”，而是“现金少、人工标注能力少、GPU 算力相对充足”。** 因此少用付费 API、少造人工标注数据，但可以积极利用本地开源模型做 hidden-state、probing、activation patching、steering、causal intervention、checkpoint trajectory 等机制实验。
4. **导师不只关心最新 LLM phenomenon，也在意老的科学问题能否在 LLM 时代得到新的、以前做不了的处理方式。** 旧问题 + 新实验轴是高优先级来源。
5. **机制分析不是为了显得深。** 必须先有真实、稳定、值得解释的 phenomenon；机制工具是解释这个 phenomenon 的手段，不是题目存在的理由。
6. **实际可行性优先于题目看起来漂亮。** TOP_POOL 优先要求 seed 已在我们能访问的 open-weight model 上报告关键现象，并且 dataset、labels、prompt/scoring recipe、reproduction code 尽量齐全。我们的研究风险应该主要押在“新的科学问题是否成立”，而不是同时押在“现象能否复现、模型能不能跑、数据能不能重建、指标能不能定义”。

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
最好确认关键 phenomenon 已经出现在我们准备分析的 open model 上
        ↓
找到现成 data / gold labels / prompts / code / checkpoints / traces / measurement
        ↓
设计一个简单、决定性的 G0，确认 critical cell 在同一模型里有足够密度
        ↓
若为正，再利用充足 GPU 做 representation / mechanism / causal intervention
        ↓
最终形成一个 ACL/NAACL/EMNLP 尺度的完整故事
```

每个候选首先要能回答：

> **是哪篇论文、哪个已知现象、或者哪个旧科学问题，逼出了这个下一问？**

然后必须再回答：

> **这个问题成立所依赖的实验对象，是否已经在我们真正能跑的模型和公开 artifact 上存在？**

如果第一个问题答不出来，题是脑补的；如果第二个问题答不出来，题的工程/现象风险仍然太高。

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
+ 已知会出现目标 phenomenon 的 open model
+ seed 的 prompt / scoring / reproduction code
+ automatic scoring
+ 本地 GPU 机制分析
= 最适合我们的题
```

## 3.5 Artifact completeness 是研究可行性的一部分

以后不能只写“dataset public / model open”就算 resource audit 通过。

TOP_POOL 应尽量确认下面四件套：

```text
data
+ exact model/checkpoint
+ prompt / scoring recipe
+ reproduction code
```

四件套越完整，我们越能把时间花在真正新的问题上。

如果 seed 的核心现象只在 GPT-5 / Claude / Gemini 上出现，而我们准备做机制的 Qwen/Llama/Gemma 是否有同样现象完全未知，那么这个题不能直接升 TOP。

## 3.6 Same-model prerequisite

机制分析必须解释**同一个模型里真实发生的 failure event**。

优先：

```text
seed 已报告 Qwen3-8B 的 anomaly
→ 官方代码可复现
→ 我们在 Qwen3-8B 上跑 G0
→ hidden-state / patching
```

避免：

```text
seed 只报告 closed frontier model
→ 假设 8B open model 也会有
→ 先写大量机制代码
→ 最后发现 prerequisite 不存在
```

## 3.7 Critical-cell density before mechanism

如果 claim 依赖下面这种 cell：

```text
内部判断正确 / 局部能力存在
BUT
最终行为错误
```

必须在 hidden-state 分析前先统计这个 cell 的真实密度。

Aggregate gap 不等于 mechanism-level phenomenon。

如果关键 cell 稀少，只能靠挑模型、挑 prompt、挑 layer、挑 subset 才出现，直接降级。

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
同一 open model 上简单 G0 复现
→
确认 critical cell 密度
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

## 6.4 机制实验也必须 bounded-search

机制题尤其容易在 `layer × token × strength` 上 winner's curse。

以后默认：

- train / validation 用于选择 probe / operating point；
- layer、token、steering strength 的选择规则在 test 前冻结；
- steering 强度尽量按 residual-stream norm 校准，而不是直接跨层/跨模型比较 raw coefficient；
- 必须有 random direction / shuffled-label / matched-norm 等少量明确 null；
- 优先 natural counterfactual patching，而不是无约束地扫 steering coefficient。

这不是为了把 control 做复杂，而是避免 measurement 自己制造 phenomenon。

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

这是现在的**强优先项**，而不只是“加分项”。

最理想的 seed 已经告诉我们：

```text
exact open model
+ exact dataset
+ exact prompt/scoring
+ exact anomaly
+ reproduction code
```

我们只比它多问一个问题。

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
15. seed 的 anomaly 只在 inaccessible closed model 上出现，而 open model prerequisite 完全未知。
16. dataset public，但 prompt/scoring/reproduction chain 缺失到需要我们先逆向重建整篇 seed。
17. 关键 dissociation 只有 aggregate score，没有可用 instance-level density。

特别记住：

> **如果 gate 和 kill line 越设计越复杂，通常不是我们越来越严谨，而是问题本身越来越不自然。**

以及：

> **如果一个题要同时赌“现象存在、模型能复现、测量有效、机制成立”，那不是一个高可行性候选。最好只保留最后一个真正的科学赌注。**

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

Artifact completeness：
- released dataset：
- exact model/checkpoint：
- prompt/scoring recipe：
- reproduction code：

Same-model prerequisite：
- seed 是否已在我们准备分析的 open model 上报告现象：
- 如果没有，最便宜 prerequisite reproduction 是什么：

已有 dataset / labels / code：
可用 open models / checkpoints：
Critical-cell definition：
Critical-cell expected / reported density：
G0 第一枪：
Kill line：
Paid API requirement for G0：
New annotation requirement for G0：
Local GPU estimate：
Mechanism-ready?：
Mechanism search space 如何冻结：
如果为正，下一步 mechanism：
如果为正，下一步 method / intervention：
如果为负，能否得到明确科学结论：
最终最强 headline：
状态：
```

如果“哪一个结果自然逼出下一问”写不出来，通常说明题还是脑补的。

如果 Artifact completeness / Same-model prerequisite / Critical-cell density 三项都含糊，不能叫 TOP_POOL。

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

### G15. Artifact completeness
TOP 候选优先要求 data + exact model + prompt/scoring + reproduction code 基本齐全。

### G16. Same-model phenomenon
要解释的 exact failure event 必须已在目标 open model 中出现，或能用极便宜的冻结 reproduction 确认。

### G17. Critical-cell density
机制 claim 所依赖的 instance-level dissociation / transition 必须有足够密度，aggregate gap 不能代替。

### G18. Bounded mechanism search
layer / token / strength 等选择必须能通过 validation 冻结，不能靠 test-set fishing 找机制。

### G19. Interpretable null
如果 intervention 无效，应该能清楚回答 scientific question 的一部分，而不是永远归因于“可能没找到正确 layer/feature”。

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

## Stage 5：artifact + resource audit

先确认：

```text
公开数据？
公开标签？
exact open model/checkpoint？
seed 是否已在这个模型上出现关键现象？
prompt / scoring recipe？
reproduction code？
公开 checkpoint？
本地 GPU 能跑？
需要多少 API？
需要多少人工标注？
```

Artifact 缺失本身就是风险，不要等写代码后才发现。

## Stage 6：critical-cell audit

在任何 probe / SAE / patching 前先确认：

```text
我们真正要解释的 instance-level event 是什么？
它在同一模型中有多少？
是否有 clean positive / negative instances？
```

如果关键 cell 不存在，直接停。

## Stage 7：才设计 G0

G0 的目标不是漂亮，而是：

> **最快判断这个 scientific object 是否真的值得后续机制分析。**

## Stage 8：phenomenon 成立后才上机制工具

GPU 用在这里，而不是用 sweep 把不存在的 phenomenon 挖出来。

机制工具的 selection 只在 train/validation 做；test 保留为锁定确认。

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
Artifact completeness：
Same-model prerequisite：
Critical-cell density：
资源检查：
机制分析可能性：
Mechanism identifiability：
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

结合我们的实际资源和前面大量失败经验，当前应当特别偏向：

> **NLP 顶会尺度的问题 + 已经站住的 open-model phenomenon + 现成 data/code + automatic labels + GPU-heavy mechanism / causal analysis。**

以后判断一个题，不再只问“这个方向热门吗”或者“这个标题酷不酷”，而要同时问：

> **它为什么是一个自然问题？**
>
> **为什么现在还没被直接回答？**
>
> **如果证明出来，ACL/NAACL/EMNLP 的读者为什么会在意？**
>
> **seed 是否已经替我们证明了最关键的 prerequisite，而不是让我们重新赌一次 phenomenon existence？**
>
> **我们能否不花大量 API / 标注钱，却用自己的 GPU 把它做得比普通 behavioral paper 更深？**

最理想的项目不是“什么都很新”。

最理想的是：

> **前人已经把实验对象钉在桌上；我们只多问一个真正重要、尚未回答，而且能被因果实验直接回答的问题。**
