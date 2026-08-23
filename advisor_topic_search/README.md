# 导师向选题搜索

这个目录用于记录**面向导师审查标准、以 CCF A/B AI / NLP 会议为主搜索池、ACL 第一优先的研究选题搜索**。

这里不是“想到什么就记什么”的脑暴目录，而是一套**候选题过滤系统**。目标不是积累最多的题，而是持续留下少数：

> **问题自然、外部锚点明确、只比前人多走一步、首轮实验简单可杀、正结果值得高兴、后面还有机制或方法口子，而且在我们的现实资源下真正做得出来。**

当前候选状态不要从 Round 日志猜，统一看：

```text
advisor_topic_search/ACTIVE_CANDIDATES.md
```

`ROUND_*.md` 只保存每轮搜索、collision audit、升降级和失败历史。

---

# 0. 最重要的原则

1. **实际可行性优先于题目看起来漂亮。**
2. **导师关心“证明出来以后是否值得高兴”。** 最好结果如果只是“嗯，合理”，降级。
3. **先有 phenomenon，再有 mechanism。** 不能先做 SAE / probe / patching 再编故事。
4. **最好只赌一个科学问题。** 不要同时赌 phenomenon 存在、open model 能复现、measurement 有效、mechanism 也成立。
5. **如果 gate 和 controls 越写越复杂，通常不是我们越来越严谨，而是问题越来越不自然。**
6. **研究室同门是“好题怎么长出来”的正例，不是领域边界。**
7. **主 seed venue 有硬约束。** ACL 第一优先；其他 CCF A/B AI/NLP 会议其次。低优先级 venue 只做背景、collision 或方法参考。
8. **我们的资源画像是：现金少、人工标注能力少、GPU 相对充足。** 这是 promotion gate，不是备注。

---

# 1. Venue policy

主 seed 默认优先级：

```text
ACL
>
NAACL / EMNLP
>
NeurIPS / ICML / AAAI / IJCAI 中与 NLP / LLM 直接相关的工作
```

原则上只从 **CCF A/B 的 AI / NLP 会议**晋级主 seed。

其他 venue 可以用于：

- 背景；
- exact collision；
- 方法参考；
- 证明某现象已有先例；

但不能单凭一个漂亮 gap 进入 TOP_POOL。

**ACL 是第一优先级。** 如果 ACL main paper 的 anomaly、appendix、error analysis、limitations 能自然逼出一步新问题，优先于从低优先级 venue 生造空白。

---

# 2. 理想题目长什么样

理想流程：

```text
已站住的 seed paper / 稳定公开现象 / 经典科学问题
        ↓
明确前人真正证明了什么
        ↓
只往旁边走一步，但 scientific question 明显不同
        ↓
确认 construct 自然，不需要我们硬造概念
        ↓
最好关键 phenomenon 已经出现在目标 open model 上
        ↓
复用 data / labels / prompts / code / checkpoints / traces
        ↓
简单、决定性的 G0 先确认 critical cell
        ↓
若为正，再用 GPU 做 mechanism / causal intervention
        ↓
形成 ACL / NAACL / EMNLP 尺度完整故事
```

每个候选首先回答两句话：

> **是哪篇论文、哪个现象、或者哪个旧科学问题，逼出了这个下一问？**

> **这个问题依赖的实验对象，是否已经在我们真正能跑的模型和公开 artifact 上存在？**

第一个答不出来，题是脑补的；第二个答不出来，工程/现象风险太高。

---

# 3. Venue-scale 与 interestingness

## 3.1 合适的宽度

一个好候选应该一句话说清楚，同时能自然支持：

- 2–4 个模型 / 模型家族；
- 1–3 个互补 setting；
- 一个明确主现象；
- 一组有目的的分析 / 机制实验；
- 一个自然的方法、干预或理论 implication。

典型形状：

```text
明确 phenomenon
+ 明确 scientific question
+ cross-setting replication
+ 深一点的 mechanism / analysis
+ 一个自然 follow-up
```

不要：

```text
整个 LLM reasoning 的统一理论          # 太宽
某个 prompt 换一个词掉 1.3 分         # 太窄
做一个新 benchmark 排模型             # 通常不够
```

## 3.2 Scientific novelty，不是 configuration novelty

以下通常不够：

- 换模型；
- 换语言；
- 换 benchmark；
- 把已有方法搬到相邻任务；
- “第一次在 X 上用 Y probe”；
- 文献矩阵里刚好有个空格。

更好的 novelty：

- 已有结果暴露新的 dissociation；
- 一个旧问题第一次因为 hidden states / checkpoints / causal intervention 可直接检验；
- 两个本应一致的能力被系统性分离；
- 已知 behavioral failure 被定位到明确 computation bottleneck；
- 新 diagnostic 迫使我们重新解释现有结论。

## 3.3 Positive-result excitement test

写代码前先假设最干净的结果已经出来，然后问：

> **如果这就是 ACL 标题和摘要核心结论，我会觉得“这改变了我对模型的理解”，还是只会说“合理”？**

如果最好结果仍只是 reasonable / expected，降级。

---

# 4. 资源与现实可行性：现金少、标注少、GPU 多

这是**总筛选标准的一部分**，不再单独维护另一套 resource policy。

## 4.1 现金成本

phenomenon-existence G0 **不能依赖大规模付费 closed-model API 调用**。

资源优先级：

1. 已发布 response traces / logits / predictions / human judgments；
2. 公开 benchmark + gold labels；
3. 本地可跑 open-weight models；
4. 公开 intermediate checkpoints；
5. 程序化生成、可 exact scoring 的数据；
6. 少量 closed API 只用于最后 spot-check / external generalization。

如果第一枪必须花大量 GPT / Claude / Gemini API 钱，直接降级。

## 4.2 人工标注成本

不能要求先新标几千条语义数据才能开始。

优先：

- benchmark gold answers；
- executable / symbolic / numeric / exact-match labels；
- public response logs；
- revision history / structured metadata；
- deterministic transformations；
- 已发布 expert annotations。

允许 phenomenon 成立后人工核验 50–200 个高价值 case；不允许把大规模 annotation 当 prerequisite。

## 4.3 GPU 是优势，不是要回避的成本

behavioral G0 成立后，欢迎利用 GPU 做：

- hidden-state decoding / probing；
- layer-wise trajectory；
- logit lens / tuned lens；
- activation patching / causal tracing；
- attention / MLP ablation；
- steering / representation engineering；
- SAE / crosscoder（前提是 phenomenon 已站住）；
- dense checkpoint analysis；
- SFT / RL / quantization 前后 representation change；
- lightweight controlled finetuning 做 causal test。

**稍微更费 GPU、但 mechanism opening 干净的题，往往优于几乎免费但只能做 surface analysis 的题。**

## 4.4 Artifact completeness

TOP_POOL 不允许只写“dataset public / model open”。尽量确认四件套：

```text
data
+ exact model/checkpoint
+ prompt / scoring recipe
+ reproduction code
```

四件套越完整，我们越能把研究风险押在真正的新问题上。

## 4.5 Same-model prerequisite

机制分析必须解释**同一个模型里真实发生的 failure event**。

优先：

```text
seed 已报告 Qwen/Llama/Gemma 某 open model 的 anomaly
→ 官方代码复现
→ 同模型 G0
→ hidden-state / causal intervention
```

避免：

```text
seed 只报告 closed frontier model
→ 假设 8B open model 也会有
→ 先写大量机制代码
→ 最后 prerequisite 不存在
```

## 4.6 Critical-cell density before mechanism

如果 claim 依赖：

```text
内部判断正确 / 局部能力存在
BUT
最终行为错误
```

必须先统计这个 exact cell 的真实密度。

Aggregate gap 不等于 mechanism-level phenomenon。

如果只有挑模型、挑 prompt、挑 subset 才出现，降级。

## 4.7 Compute abundant ≠ search free

GPU 多也不能无限 fishing。每个准备晋级的候选要记录：

- model size / family；
- checkpoint 数；
- examples / rollouts 数；
- 预计 GPU-hours；
- activation / checkpoint storage；
- 是否需要 multi-node communication；
- layer / token / threshold / prompt 等 researcher degrees of freedom 有多少。

尤其避免：

```text
model × checkpoint × layer × token × threshold × prompt × dataset
```

的大扫荡后挑最好看的 cell。

## 4.8 Resource kill rules

出现任一项就 kill 或大幅降级：

1. phenomenon existence 依赖大量付费 API；
2. 只有 closed model 有用且没有 released outputs；
3. 第一批 useful data 需要大量新人工标注；
4. 每个样本都要昂贵 LLM judge 且没有可靠 automatic proxy；
5. 必须从头训练大 foundation model 才能看到对象；
6. G0 实际上是大规模 model × prompt × sample sweep；
7. public artifact 只覆盖 toy regime，meaningful regime 要靠 fishing；
8. mechanism 需要大量灵活 layer / feature choice，正结果不可独立解释。

## 4.9 最理想的资源结构

```text
自然科学问题
+ 已发表 anomaly / old scientific problem
+ existing gold data
+ open model / public checkpoint
+ automatic scoring
+ local GPU-heavy mechanism analysis
+ little or no paid API
+ little or no new annotation
```

---

# 5. 导师偏好的主线：老问题 × LLM 时代的新实验能力

高优先级形状：

```text
旧领域本来就有明确 competing hypotheses
+
LLM 提供以前没有的 experimental handle
=
新的科学检验
```

可参考：

- cognitive science：belief revision、anchoring、confidence、recognition vs recall；
- decision-making：uncertainty、abstention、aggregation；
- memory：interference、serial position、retrieval vs storage；
- learning：acquisition order、forgetting、compression、generalization；
- psycholinguistics：incremental processing、composition、ambiguity；
- information science：evidence propagation、revision、retrieval。

LLM 时代的新 handle：

- dense training checkpoints；
- token-by-token / layer-by-layer hidden state；
- exact causal intervention；
- same architecture at different training stages / scales；
- controlled generation；
- open models 可重复实验。

不要退化成：

> “人有 X bias，LLM 有没有？”

更好的形式：

> “旧文献里 X 与 Y 是 competing explanations；LLM 的 training trajectory / hidden states 让我们第一次能直接区分。”

---

# 6. 好题常见的生长方式

## 6.1 已知现象 → 更基础 distinction

```text
现象已经存在
→
underlying distinction / state / causal variable
```

## 6.2 成熟 measurement axis × 重要 scientific object

例如：

- pretraining checkpoint；
- post-training stage；
- model scale；
- context length；
- compression / quantization；
- causal intervention strength。

但 axis 必须回答原本就重要的问题，不是为了画曲线。

## 6.3 老问题 × model development trajectory

```text
human development / learning hypothesis
→
model training checkpoint
```

## 6.4 “模型会做” → “是否真正拥有背后能力”

benchmark score 高，不代表 latent capability 真存在；可以去掉 shortcut / label clue 做更直接 diagnostic。

## 6.5 最高优先级：dissociation / contradiction

优先搜：

```text
模型内部知道 X，但行为不使用 X
```

```text
局部步骤都会，但组合后失败
```

```text
更多训练 / 更多证据 / 更强模型，反而让明确行为变差
```

```text
reasoning trace 已识别问题，但 final answer 不跟随
```

---

# 7. 机制题的额外规则

## 7.1 Phenomenon before mechanism

禁止：

```text
先 SAE / probe
→
看到 feature
→
再编 phenomenon
```

正确：

```text
已有 anomaly
→
同一 open model 简单 G0
→
critical cell density
→
明确 decision point
→
probe / patch / ablate / steer
```

## 7.2 Representation ≠ causal use

linear probe 只能说明 information available。

如果 claim 是“模型使用了它”，必须考虑：

- activation patching；
- targeted steering；
- causal mediation；
- ablation；
- natural matched intervention。

但**方法本身永远不是 novelty**。

## 7.3 Bounded mechanism search

默认：

- train / validation 用于选择 probe / operating point；
- layer、token、steering strength 的规则在 test 前冻结；
- steering 强度按 residual norm 等可比尺度校准；
- random direction / shuffled-label / matched-norm 等少量明确 null；
- 优先 natural counterfactual patching；
- test 只做锁定确认。

## 7.4 Interpretable null

intervention 无效时必须也能回答一部分科学问题。

如果 null 永远可以解释成“可能只是没找到正确 layer/feature”，说明设计不可证伪。

---

# 8. 什么样的题快速杀掉

1. 为 identification 必须造不自然 counterfactual；
2. 只因为文献矩阵有空格就叫 gap；
3. 最好结果只是“X 会影响 accuracy”；
4. 必须先证明“不是 A、不是 B、不是 C”才能说主结论；
5. controls 越来越多才能维持 claim；
6. 主要贡献只是换模型 / 语言 / benchmark；
7. 最强结果完全符合直觉；
8. phenomenon existence 需要大规模新标注；
9. G0 需要大量 paid API；
10. 核心指标依赖昂贵 LLM-as-a-Judge 且无 automatic proxy；
11. 必须从头训练大模型；
12. 只有 toy regime 成立；meaningful regime 靠 fishing；
13. 正结果之后没有 mechanism / intervention / mitigation opening；
14. 为了做机制，先假设从未证明存在的 latent object；
15. seed anomaly 只在 inaccessible closed model 上出现；
16. dataset public，但 prompt/scoring/reproduction chain 缺失到要先逆向整篇 paper；
17. 只有 aggregate gap，没有 usable instance-level critical cell；
18. seed venue 不符合当前 CCF A/B AI/NLP 主池；
19. exact question 已经 crowded，只剩“再加一个 probe/patching”；
20. planned causal contrast 同时改变多个因素，无法用 1–3 个关键 control 解释。

特别记住：

> **如果 gate 和 kill line 越设计越复杂，通常不是我们越来越严谨，而是问题本身越来越不自然。**

> **最理想的项目只剩一个真正的科学赌注。**

---

# 9. Candidate card

进入 ACTIVE/HOLD 前至少写清楚：

```text
题目：
Venue-scale headline：
Seed paper / old scientific question：
Seed venue（CCF A/B?）：
前人已经证明什么：
哪一个结果自然逼出下一问：
我们只移动哪一步：
一句话 Research Question：
为什么自然：
为什么正结果不只是“果然如此”：
为什么像 ACL/NAACL/EMNLP 尺度：
最近 exact collision：
同门 / archived collision：

Artifact completeness：
- released dataset：
- exact model/checkpoint：
- prompt/scoring recipe：
- reproduction code：

Same-model prerequisite：
Critical-cell definition：
Critical-cell expected / reported density：
G0 第一枪：
Kill line：

Paid API requirement for G0：
New annotation requirement for G0：
Open-weight model availability：
Model size / family：
Number of checkpoints：
Examples / rollouts：
Estimated GPU-hours：
Activation / checkpoint storage：
Multi-node required?：
Researcher degrees of freedom：
Can G0 run entirely from released artifacts?：

Mechanism-ready?：
Mechanism search space 如何冻结：
如果为正，下一步 mechanism：
如果为正，下一步 intervention / method：
如果为负，能否得到明确科学结论：
最终最强 headline：
状态：
```

如果“哪一个结果自然逼出下一问”写不出来，通常是脑补题。

如果 Artifact completeness / Same-model prerequisite / Critical-cell density 三项含糊，不能叫 TOP_POOL。

---

# 10. Promotion Gates

只有同时通过才值得认真写代码：

- **G1 Naturalness**：一句话理解为什么存在；
- **G2 External anchor**：seed / anomaly / old problem 明确；
- **G3 One-step distance**：靠近 seed，但 scientific question 不同；
- **G4 Venue-scale**：够成 ACL/NAACL/EMNLP 完整故事；
- **G5 Non-triviality**：不是常识；
- **G6 Positive-result excitement**：结果成立真的值得高兴；
- **G7 Cheap decisive G0**：第一枪不靠大量 API / annotation；
- **G8 Killability**：核心结构不存在就停；
- **G9 Low control complexity**：最好 1–3 个关键 control；
- **G10 Existing object**：数据/对象基本已存在；
- **G11 Collision auditability**：exact question 可检索；
- **G12 Mechanism/method opening**：正结果后自然有下一步；
- **G13 Resource fit**：现金/标注不卡死，GPU 能发挥；
- **G14 Mechanism identifiability**：干预真的区分 causal explanation；
- **G15 Artifact completeness**：data + model + prompt/scoring + code；
- **G16 Same-model phenomenon**：exact failure 已在目标 open model 出现或可极便宜复现；
- **G17 Critical-cell density**：instance-level event 足够多；
- **G18 Bounded mechanism search**：不能靠 test-set fishing；
- **G19 Interpretable null**：null 也回答问题；
- **G20 Venue eligibility**：主 seed 来自当前 CCF A/B AI/NLP 池，ACL 第一优先。

---

# 11. 标准搜索流程

## Stage 1：广泛扫 seed

- ACL 优先，其次 EMNLP / NAACL；
- NeurIPS / ICML / AAAI / IJCAI 中与 LLM/NLP/机制直接相关工作；
- 2025–2026 主会；
- old cognitive / information / learning literature；
- appendix / error analysis / limitations，而不是 Future Work 生造题。

## Stage 2：每篇只写“真正证明了什么”

先不设计方法。

## Stage 3：只允许一步移动

优先：

```text
behavior → internal representation
representation → causal use
reasoning → final decision
local competence → global composition
final model → training trajectory
phenomenon → exact failure point
old hypothesis → modern causal test
single variable → meaningful interaction
```

## Stage 4：collision search

查：

- seed 作者后续；
- exact wording / equivalent formulation；
- 2025–2026 相邻工作；
- 同门；
- 本仓库 archived failures。

如果 exact / near-exact 已回答，kill，不靠换 dataset 续命。

## Stage 5：artifact + resource audit

先确认：

```text
公开数据？
公开标签？
exact open model/checkpoint？
seed 是否已在该模型出现关键现象？
prompt / scoring recipe？
reproduction code？
公开 checkpoints / responses？
GPU-hours / storage？
需要 multi-node 吗？
需要多少 paid API？
需要多少人工标注？
```

## Stage 6：critical-cell audit

任何 probe / SAE / patching 前先确认：

```text
真正要解释的 instance-level event 是什么？
同一模型里有多少？
是否有 clean positive / negative instances？
```

没有就停。

## Stage 7：才设计 G0

目标不是漂亮，而是：

> **最快改变我们对这个 scientific object 是否值得做的信心。**

## Stage 8：phenomenon 成立后才上 mechanism

GPU 用于解释已经存在的对象，不用于 sweep 把不存在的对象挖出来。

---

# 12. 文件组织规则

```text
README.md
    唯一总筛选标准（venue + scientific + resource + mechanism）

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
3. 如果筛选标准真的发生变化，再改 README；
4. 不再维护第二份平行 resource policy。

---

# 13. 当前总判断

当前最应该偏向：

> **ACL 优先的 CCF A/B AI/NLP seed + 已站住的 open-model phenomenon + 现成 data/code + automatic labels + GPU-heavy causal mechanism。**

以后判断一个题，同时问：

> **为什么是自然问题？**
>
> **为什么现在还没被直接回答？**
>
> **证明出来为什么值得高兴？**
>
> **seed 是否已经替我们证明了最关键 prerequisite？**
>
> **第一枪能不能简单地 kill？**
>
> **我们能否少花 API / 标注钱，却用 GPU 把它做得比普通 behavioral paper 更深？**

最理想的项目不是“什么都新”。

最理想的是：

> **前人已经把实验对象钉在桌上；我们只多问一个真正重要、尚未回答，而且能被因果实验直接回答的问题。**
