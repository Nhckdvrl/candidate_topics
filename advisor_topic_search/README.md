# 导师向选题搜索：统一标准

这个目录只做一件事：

> **寻找少数真正值得做、符合导师审查习惯、达到 ACL/EMNLP/NAACL 等顶会尺度、并且在我们的资源下有较高概率做出来的研究题目。**

这里不是“从最新论文里找一个 gap，再套 probe / SAE / activation patching”的地方，也不是“先注册很多 candidate 再慢慢试”的地方。

**README 是唯一总标准。** 如果某个 `ROUND_*.md`、旧 candidate、聊天记录与这里冲突，以本 README 为准。

当前候选只看：

```text
advisor_topic_search/ACTIVE_CANDIDATES.md
```

搜索历史放在 `ROUND_*.md`；真正决定生死的 first-shot / prerequisite 实验放在 `g0/`。

---

# 1. 我们到底要什么题

一个好候选至少同时满足下面六件事。

## 1.1 问题本身自然

不用任何 mechanism / benchmark / method 名，也能用一句普通话说明：

```text
为什么会发生 X？
LLM 是否真正拥有 Y？
X 和 Y 到底如何交互？
LLM 时代到来后，旧现象 Z 是否发生变化？
一个真实系统为什么在条件 C 下稳定失败？
```

如果必须先解释三分钟技术背景，别人才能理解“为什么值得问”，优先级下降。

## 1.2 最好结果值得高兴

先假设最理想结果出来，再问：

> **如果这是论文标题或摘要第一句，我会觉得“原来是这样”，还是“嗯，这不是废话吗”？**

值得高兴的结果可以是：

- 一个反直觉现象；
- 一个清楚的因素分解；
- 一个此前没被系统回答的 interaction / trade-off；
- 一个旧科学问题在 LLM 时代得到新的答案；
- 一个 representation–behavior dissociation；
- 一个真实系统 failure 被定位到明确条件；
- 一个可直接使用的 practical rule。

**不要求每个题都做 mechanism。**

## 1.3 题目宽度对齐 NLP / AI 顶会

目标形状通常是：

```text
一个清楚问题
+ 一个自然 measurement / experimental handle
+ 2–4 条 headline findings
+ 少量关键 controls
+ analysis / mechanism / intervention / method 中至少一种自然深化
```

不要：

```text
整个 LLM reasoning 的统一理论
某 prompt 换词掉 1.3 分
再做一个 leaderboard benchmark
已有 phenomenon 上第一次用 SAE
换模型 / 换语言 / 换数据集作为主要 novelty
```

## 1.4 第一枪必须直接碰科学问题

理想路径：

```text
released artifact
→ one-model sanity check
→ 我们自己的 G0
```

不理想：

```text
重建 upstream 环境
→ 复现整篇 seed paper
→ 满足 relation A
→ 满足 panel B
→ eligibility C
→ 才能跑我们的 G0
```

**真正的 scientific question 离开工越远，题越差。**

## 1.5 实际做得出来

我们不是“总体资源少”，而是：

```text
现金少
人工标注能力少
GPU 相对充足
```

因此：

- 不依赖大规模付费 closed-model API；
- 不依赖大规模新人工标注；
- 优先公开数据、gold labels、released predictions / traces / logits；
- 优先 open-weight model、公开 checkpoint、程序化 / executable labels；
- 现象站住以后，GPU-heavy 的 controlled training、checkpoint analysis、probing、patching、SAE、steering 都可以做。

GPU 应该用来**深入一个已经存在的问题**，不是大 sweep 挖一个可能根本不存在的 phenomenon。

## 1.6 负结果也有信息

好的 G0 即使失败，也应该缩小解释空间。

例如：

```text
A 不成立，但 B 成立
→ failure 更像 encoding 而不是 readout
```

坏的失败是：

```text
seed relation 没复现
→ stop
→ 除了“没复现”什么都没学到
```

---

# 2. 组内正例：导师真正接受什么 research shape

这一节来自实验室 Slack 中 `r_han / r_hamdi / r_kisako / r_utami / r_sato / r_tsujimoto / r_yano / r_xiang` 等研究频道。

这里的目的不是列成员履历，而是校准：

> **什么问题导师觉得自然，什么实验形状能一路长成论文，导师反复会追问什么。**

| 组内方向 | 研究形状 | 最重要的启发 |
|---|---|---|
| `r_han`：Frame semantics / FrameNet generation & relation completion | 经典 NLP 资源 × LLM 时代的新能力 | old NLP problem/resource 完全可以重新成为好题；关键是新问题，不是“换 LLM 再跑” |
| `r_hamdi`：real vs fictional / internal status representation | 自然 construct → matched controls → representation → causal behavior | 如果做机制，先把 construct 隔离干净；probe 不是贡献终点 |
| `r_kisako`：dimensionality reduction × quantization | 两个成熟因素 × 一个统一 storage-budget axis | interaction / trade-off analysis 本身可以是完整论文，不需要硬上机制 |
| `r_utami`：LLM 时代 academic English 的 native-language signal | 自然社会/语言变化问题 × 可解释 proxy | 问题本身重要时，method 可以简单；重点是 proxy 和 alternative explanations |
| `r_sato`：character-level information acquisition | 已知 phenomenon → competing factors → controlled experiments | 最值得模仿：先提出 2–4 个自然解释，再直接控制 data/tokenizer/distribution 去区分 |
| `r_tsujimoto`：semantic frame induction / program verification | 旧概念体系 × 自动诱导；goal-first practical problem | old scientific construct 仍可作为 anchor；practical 题也要从目标而不是方法出发 |
| `r_yano`：FrameBench / implicit frame-semantic understanding | 旧 task formulation → 更接近真正 scientific question 的 measurement | novelty 可以来自“重新定义我们真正应该测什么”，不是新模型 |
| `r_xiang`：retrieval-agent safety（早期方向） | concrete system failure × controlled normal/adversarial contrast | 真实 failure 可以做，但“攻击后分数下降”不够，必须继续问什么时候、为什么、怎么修 |

从这些正例里抽出来的共同规律：

1. **问题先于方法。**
2. **研究对象往往本来就存在，不需要先造一个复杂 seed phenomenon。**
3. **少数 competing explanations 比大量 settings 更重要。**
4. **分析型论文完全成立，不强迫 mechanism。**
5. **反直觉 finding 很值钱，但先排 artifact。**
6. **标题/第一页应直接让人看到研究对象，而不是 method 名。**
7. **claim 强度要和证据强度一致。** 证据只支持 analysis，就不要硬写 mechanism。

---

# 3. 优先寻找的六类题

以后找题优先落到下面六种 research shape 之一。找不到对应 shape，不代表一定不能做，但要额外谨慎。

## Type A — 已知自然现象 → 因素分解

代表：`r_sato`。

```text
模型已有一个稳定能力 / failure / anomaly
→ 提出 2–4 个自然 competing explanations
→ 控制 training data / tokenizer / input / distribution / interface
→ 逐项判断哪些因素真的贡献
```

这是当前**最高优先级形状之一**。

## Type B — 经典 NLP / cognitive / resource 问题 × LLM 时代的新 measurement / capability

代表：`r_han / r_tsujimoto / r_yano`。

核心不是：

> 老任务 + 新模型。

而是：

> **LLM 出现后，以前无法直接问的问题是否现在可测？原来的 task formulation 是否已经不再对应真正想知道的 scientific question？**

导师明确在意这类“老问题在 LLM 时代能否有更好的处理方式”。

## Type C — 两个成熟因素 × 一个自然 interaction / budget axis

代表：`r_kisako`。

```text
成熟因素 X
×
成熟因素 Y
→ 以前分别研究很多
→ interaction / trade-off 没有被系统回答
→ 用一个自然预算 / 约束统一比较
```

要求：结果必须有结构，不是画二维表。

## Type D — 自然社会 / 语言变化问题 × 可解释 proxy

代表：`r_utami`。

适合问题本身重要、可观测数据已存在、proxy 可解释、alternative explanations 能控制的题。

## Type E — 自然 construct → representation → causal behavior

代表：`r_hamdi`。

只有满足下面顺序才做：

```text
construct 自然
→ matched controls 隔离 construct
→ representation evidence
→ causal intervention 回答原问题
```

禁止：

```text
我想做 SAE / probe / patching
→ 再找一个题套上去
```

## Type F — concrete system failure → controlled diagnosis / mitigation

代表：agent / retrieval / safety / tool-use 等真实系统问题。

要求至少回答三件事中的两件：

```text
什么时候失败？
为什么失败？
怎样有针对性修复？
```

单纯 score drop 不够。

---

# 4. Seed / venue policy

## 4.1 主 seed 范围

主搜索池原则上只使用 **CCF A/B 的 AI / NLP 会议**：

```text
ACL                 第一优先
NAACL / EMNLP       第二优先
NeurIPS / ICML
AAAI / IJCAI        补充与 NLP / LLM 直接相关的工作
```

低优先级 venue（包括 EACL 等）可以用于：

- 背景；
- exact collision；
- 方法参考；
- old-question anchor；
- 组内 research-shape calibration。

但**不能仅凭一个漂亮 gap 作为主 seed 升进 ACTIVE**。

## 4.2 优先挖哪里

优先看：

- main paper 的 anomaly；
- error analysis；
- appendix 中的 unexpected result；
- model-size / training-stage / task-condition dissociation；
- 作者已经公开但没有继续追的问题；
- 经典 NLP / cognitive / information-science 问题；
- 实际系统中稳定出现的 failure。

**不要主要靠 Future Work 生造 gap。**

---

# 5. 候选晋级的 8 个硬 Gate

原来的大量 gate 合并为下面八个。它们是正式 candidate 的核心审核标准。

## G1 — Natural Question

能否不用方法名，用一句普通话解释：

> **我们到底想知道什么，为什么值得知道？**

明显失败：不注册。

## G2 — Scientific Anchor + Venue Scale

必须至少有一个明确 anchor：

```text
已知 phenomenon / anomaly
经典 scientific question
成熟资源 / task 的新 measurement 问题
真实 system failure
```

并且最好结果能自然长成 ACL/EMNLP/NAACL 等尺度的 2–4 条 headline findings。

## G3 — Scientific Novelty

novelty 必须是 scientific，而不是 configuration。

优先：

- 新的因素分解；
- 新 interaction / trade-off；
- old question 在 LLM 时代的新答案；
- 更正确的 measurement / formulation；
- 稳定 dissociation / reversal；
- 明确 system bottleneck。

降级：

- 换模型；
- 换语言；
- 换 benchmark；
- 已有现象上第一次用某个 interpretability method。

## G4 — Competing Explanations / Interaction

进入 ACTIVE 前至少满足一个：

```text
有 2–4 个自然 competing explanations
或
有一个清楚、自然的 interaction / budget axis
```

如果唯一计划是“看看 hidden state 有什么”，不进 ACTIVE。

## G5 — Direct G0 + Low Prerequisite Tax

第一枪必须尽量直接碰我们的科学问题。

健康：

```text
released data/model
→ quick sanity
→ G0
```

危险：

```text
复杂 upstream reproduction
→ 多个 eligibility gate
→ 才到 G0
```

原则：

- fragile prerequisite 尽量不超过一层；
- receipt 成本应明显低于核心 G0；
- prerequisite 失败时最好仍有科学信息。

**G5 明显失败：不注册。**

## G6 — Artifact + Resource Fit

TOP 候选尽量具备：

```text
released dataset
+ exact model/checkpoint
+ prompt / scoring recipe
+ reproduction code / released outputs
```

资源硬限制：

- phenomenon-existence G0 不依赖大量 closed API；
- 第一枪不依赖大规模新 annotation；
- 核心指标最好自动 / 程序化可判；
- 有 open-weight regime；
- GPU-heavy 后续允许，但不能用 sweep 找 phenomenon。

缺两项以上 artifact，且 G0 还不简单：降级。

## G7 — Clean Identification + Killability

实验必须能清楚回答：

> **什么结果出现时，这个解释活；什么结果出现时，它死？**

要求：

- 优先 same-model / paired / matched contrast；
- controls 主要用来区分 competing explanations；
- 一般只接受 1–3 个关键 control；
- 如果需要越来越多 control 才维持 claim，优先杀题；
- layer / strength / threshold 等 search space 必须可冻结。

## G8 — Payoff + Continuation

最好结果必须有 headline；负结果最好有 information gain；正结果后还要至少有一个自然 opening：

```text
更深 analysis
mechanism
intervention
method / mitigation
practical rule
```

但 opening 不是 requirement stacking：**不能为了“以后还能做”给当前 G0 再加 prerequisite。**

### 晋级规则

以下四个 gate 任一明显失败，原则上**不注册正式 candidate**：

```text
G1 Natural Question
G3 Scientific Novelty
G5 Direct G0 / Low Prerequisite Tax
G6 Artifact / Resource Fit
```

其他 gate 如果弱，只能进 `WATCH / HOLD`，不能直接进 `ACTIVE`。

---

# 6. 可行性：我们最需要防的不是 GPU 不够，而是“题太折腾”

## 6.1 Time-to-scientific-question

每个候选必须明确回答：

> **从 clone repo 到第一次得到“属于我们自己的 scientific result”，要经过几步？**

如果主要时间都花在：

- 修 upstream environment；
- 对齐特殊 hostname / service；
- 重建复杂 pipeline；
- 复现 seed 的整张表；
- 满足脆弱 aggregate relation；

那就是高风险题。

## 6.2 Critical cell 必须真实存在

机制题尤其要求：

- exact failure 在我们准备使用的 open model 上已经报告，或能极便宜确认；
- critical cell 数量不能极稀疏；
- 不允许先扫很多 model × prompt 才找到一个能做的 regime。

## 6.3 Artifact audit 先于 story

遇到反直觉结果先查：

- scoring bug；
- normalization；
- tokenizer；
- data leakage；
- generator / judge bias；
- sample-size；
- distribution shift；
- quantizer / preprocessing / decoding 设置。

确认不是 artifact 后再升为 headline。

## 6.4 Same-model / paired contrast 优先

最理想：

```text
same model
same item / same underlying object
只改变一个科学变量
→ behavior flip / representation difference
```

比 cross-model、cross-checkpoint、cross-training-regime 的 attribution 干净得多。

跨 checkpoint / 跨模型可以做，但必须先问 representation basis、training confound、trajectory divergence 是否会让机制主张失去识别性。

---

# 7. Topic 25：永久反面教材

Topic 25 不是“工程没跑完”，而是完整 receipt 后的 scientific stop：

```text
Qwen3-8B-Think gold-only = 0.45746
Qwen3-8B-Think noisy      = 0.35179

frozen requirement:
noisy thinking >= thinking gold-only

result: FALSE
→ SEED_RELATION_NOT_REPRODUCED
→ G0 NOT RUN by protocol
```

最终工程问题已排除，所以真正教训是：

> **科学问题前放了一个昂贵、脆弱、失败后信息增益很低的 seed-reproduction gate。**

以后出现下面结构，直接重罚：

```text
upstream reproduction A
→ relation B
→ panel C
→ eligibility D
→ our G0 E
```

尤其如果失败后的唯一结论是：

```text
NOT_REPRODUCED
```

而此前已经花了大量 GPU / 工程时间，这种题应该在 search audit 阶段就死。

### Topic 25 型 kill rule

出现任一项，默认不进 ACTIVE：

- scientific question 前有两层以上脆弱 prerequisite；
- receipt 本身接近一个 reproduction project；
- receipt 比 G0 更贵；
- seed 关键 relation 没在 exact open model 上报告；
- prerequisite 失败没有新的科学区分；
- 必须先复现 seed paper 大部分结果才有资格开始自己的研究；
- aggregate relation 必须先满足，才能进入真正想研究的 instance-level question。

---

# 8. 机制题的额外规则

机制题不是默认更高级，只在科学问题需要时做。

## 8.1 什么时候值得进入 mechanism

至少已经有一个：

```text
稳定 behavioral anomaly
稳定 representation–behavior dissociation
明确 competing causal explanations
可程序化 intermediate state
```

## 8.2 Representation ≠ causal use

probe / decoder 只能说明 information available。

如果 claim 是：

> 模型“使用”了这个 representation。

则应考虑 natural matched intervention、patching、ablation、targeted steering 等 causal evidence。

## 8.3 Mechanism search space 必须有界

不能：

```text
32 layers × 20 strengths × 8 prompts
→ test 上挑最好看的结果
```

layer / token / strength / threshold 应在 validation / prior hypothesis 上冻结。

## 8.4 Null 必须可解释

如果 intervention 失败后只能说：

> 可能 layer 没找对。

说明设计不可证伪，需要降级。

---

# 9. 快速 Kill Rules

以下题型默认高危：

## 9.1 科学问题弱

- 从 method 出发，而不是问题出发；
- 最好结果只是“X 影响 accuracy”；
- 题目只能写成方法名；
- configuration novelty；
- exact question 已 crowded，只剩“再做一次 mechanism”。

## 9.2 Identification 脏

- planned contrast 一次改变很多因素；
- dataset/task identity 与目标 construct 混淆；
- 为 identification 必须造很不自然的 counterfactual；
- controls 越加越多才能保住 claim。

## 9.3 Resource / artifact 差

- 现象只在 inaccessible closed model 上；
- G0 依赖大量 paid API；
- 第一批 useful data 需要大规模新人工标注；
- 核心指标完全依赖昂贵 LLM judge；
- public artifact 只覆盖 toy regime；
- meaningful regime 必须靠 fishing。

## 9.4 Prerequisite tax 高

- 需要复杂 upstream reproduction；
- receipt 比 G0 贵；
- seed relation 脆弱；
- prerequisite 失败 information gain 低；
- 需要复现整篇 seed paper 才能开始自己的问题。

## 9.5 Researcher degrees of freedom 太大

- 需要 model × prompt × layer × threshold 大 sweep；
- 只有 aggregate gap，没有清楚 instance-level object；
- null 永远可以解释成“没找到正确 setting”。

---

# 10. 标准搜索与审核流程

每一轮都按同一个流程，不允许跳着来。

## Stage 0 — 先看组内 research shape 和 ACTIVE

先问：

```text
现在缺哪类题？
A phenomenon → factor decomposition
B old question → new LLM measurement
C interaction / budget analysis
D social / language change
E construct → representation → causal behavior
F real system failure
```

不要一上来就搜“最新 mechanistic interpretability paper”。

## Stage 1 — 广泛找 anchor

来源：

- ACL 优先；
- NAACL / EMNLP；
- NeurIPS / ICML / AAAI / IJCAI 中相关工作；
- 经典 NLP / cognitive / information-science 文献；
- benchmark error analysis；
- 真实系统 failure。

## Stage 2 — 只写“前人真正证明了什么”

每篇 seed 先只记录：

```text
他们真正证明了什么？
在哪个模型 / 数据 / setting 上成立？
关键 effect size / critical cell 是什么？
哪些解释已经被他们做掉？
```

**先不设计 SAE / probe / patch。**

## Stage 3 — 生成自然问题

写一句 natural research question，并标记属于哪类组内 research shape。

如果写不出来，不注册。

## Stage 4 — 写 competing explanations / interaction axis

至少写：

```text
Explanation A
Explanation B
```

或一个明确 X × Y interaction / budget axis。

如果写不出来，而计划只是“看看内部发生了什么”，降级。

## Stage 5 — Exact collision audit

必须查：

- seed 作者后续；
- exact question wording；
- 2025–2026 相邻工作；
- 本仓库 archived failures；
- 组内相近项目；
- 已有 mechanism follow-up。

发现核心 question 已做完：直接 kill，不靠缩窄到一个小 setting 续命。

## Stage 6 — Prerequisite / artifact / resource audit

明确写：

```text
到第一条 scientific result 有几步？
最可能失败的是哪一步？
失败后得到什么信息？
数据/模型/labels/code 是否齐？
需要多少 API / annotation / GPU？
critical cell 是否在 exact open model 上存在？
```

## Stage 7 — 才允许写 Candidate Card

通过 Gate 后才进 `ACTIVE_CANDIDATES.md`。

不要因为已经花时间读了很多 paper，就降低晋级门槛。

## Stage 8 — 才设计 G0

G0 的目标只有一个：

> **用最短路径，让我们对某个 scientific explanation / interaction 是否成立发生明显更新。**

不是“证明我们有资格继续做下一层实验”。

## Stage 9 — 根据结果决定 paper shape

结果强以后再决定：

```text
analysis
mechanism
intervention
method / mitigation
practical rule
```

不强迫所有题最后都变成 mechanism paper。

---

# 11. Candidate Card：统一模板

进入 `ACTIVE / HOLD` 前至少填写：

```text
题目：
一句话 natural research question：
组内 research shape（A–F）：
为什么普通 NLP/ML 研究者会关心：

Scientific anchor：
Seed paper / old question / system failure：
Seed venue：
前人已经证明什么：
我们真正的新问题：

Competing explanation A：
Competing explanation B：
（必要时 C）：
或 interaction / budget axis：

最便宜的 G0：
G0 正结果意味着什么：
G0 负结果意味着什么：
Kill line：

最好结果的 headline：
可能的 2–4 条 findings：
正结果后的自然 opening：

Exact collision：
同门 / archived collision：

Artifact：
- dataset：
- exact model/checkpoint：
- labels/scoring：
- code / released outputs：

Time-to-scientific-question：
Prerequisite chain：
Receipt cost vs G0 cost：
Prerequisite failure information gain：
Critical-cell definition / density：

Paid API：
New annotation：
Open-weight availability：
GPU / storage requirement：
Researcher degrees of freedom：

Mechanism 是否真的必要：
若需要，哪一个 causal distinction：
状态：ACTIVE / HOLD / WATCH / KILL
```

---

# 12. 候选排序：先过 Gate，再比较谁更值得做

通过硬 Gate 后，再用下面六项排序，每项 0–2 分：

```text
Naturalness / importance
Scientific novelty
Positive-result excitement
Feasibility / artifact completeness
Identification cleanliness
Time-to-scientific-question
```

解释：

```text
0 = 明显弱点
1 = 可以接受
2 = 明显优势
```

只有全部硬 Gate 基本通过后才打分。**总分不能救一个已经违反 hard gate 的题。**

排序时优先选择：

> **更快碰到科学问题、关键对象已经存在、失败也有信息、最好结果更值得写进标题的题。**

---

# 13. 文件组织

```text
README.md
    唯一总标准；不在 Round log 里再发明长期规则

ACTIVE_CANDIDATES.md
    唯一当前候选状态表

ROUND_*.md
    每轮搜索、collision、升降级、kill 历史

g0/
    真正决定候选生死的 first-shot / prerequisite scripts
```

每轮结束必须：

1. 写 `ROUND_*.md`；
2. 同步 `ACTIVE_CANDIDATES.md`；
3. 新的长期教训只有足够普适时才更新 README；
4. 已经写了很多代码不能成为保留候选的理由。

---

# 14. 注册 Candidate 前的最后检查

在正式注册前，逐条回答：

```text
[ ] 我能否一句话讲清问题，不提 method 名？
[ ] 这个问题是否对应组内 A–F 中至少一种成熟 research shape？
[ ] 最好结果是否真的值得高兴？
[ ] novelty 是 scientific，而不是 configuration？
[ ] 至少有两个自然解释，或一个明确 interaction axis？
[ ] 第一枪是否直接碰我们的科学问题？
[ ] receipt 是否简单、短、失败也有信息？
[ ] 数据 / 模型 / labels / scoring / code 是否基本齐？
[ ] 不需要大量 paid API 或新人工标注？
[ ] critical phenomenon 是否在可访问 open model 上存在？
[ ] 1–3 个关键 control 是否足以解释主要 confound？
[ ] exact collision 是否认真查过？
[ ] 如果 G0 为 null，我是否仍能学到东西？
[ ] 如果 G0 为正，是否自然长成 2–4 条顶会尺度 finding？
```

有明显的“否”，先不要注册。

---

# 15. 一句话总纲

以后选题的默认顺序只有这一条：

```text
自然科学问题 / old question / real failure
→ 组内 research shape 校准
→ external anchor
→ competing explanations / interaction
→ collision audit
→ prerequisite + artifact + resource audit
→ 最短 G0 直接碰科学问题
→ 结果决定走 analysis、mechanism 还是 method
```

最应该避免的是：

```text
最新 paper 有个 gap
→ 先复现复杂 seed
→ 假设 open model 也有
→ 找 hidden representation
→ patch 一遍
→ 最后才问这件事到底值不值得知道
```

我们要的不是“最复杂的题”，也不是“最新的题”。

> **我们要的是：问题天然成立、尺度够大、第一枪直接、可行性高、结果值得高兴、失败也有信息，并且能在现有资源下真正做成论文的题。**