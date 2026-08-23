# 导师向选题搜索

这个目录用于记录**面向导师与研究室风格的研究选题搜索**。

它和 [`user_interest_topic_search`](../user_interest_topic_search/) 的目标不同。这里首先考虑的不是“哪个前沿方向最让我兴奋”，而是：

> **什么样的具体研究问题，既符合笹野研真实的研究风格，又能经得住导师追问，使用可获得的数据，在半年左右成长为一篇站得住的 NLP / 语言科学 / 信息科学论文？**

这个目录不应该因为 Agentic RL、VLA、机器人、通用后训练、机制解释等方向流行，就强行把这些方向塞进来。

核心目标是：

> **从研究室已经证明可行的研究邻域出发，沿相邻问题向外搜索，但避免直接复制同门当前正在做的题。**

---

# 1. 这个目录要找什么样的题

理想的题目结构是：

```text
具体的语言 / 信息 / 科学对象
        ↓
一句话能讲清楚的自然问题
        ↓
已有数据，或低成本可构造数据
        ↓
简单、可审计的测量方式
        ↓
一个能明显支持或杀死问题的首轮实验
        ↓
能够成长为 ACL / EMNLP / NAACL / EACL / TACL 级论文的叙事
```

面向导师的标准尤其强调：

- 对象具体；
- 问题自然；
- 数据路径明确；
- 实证分析可解释；
- 一个人半年左右能做；
- 新意不能只是“换一个新模型再跑旧 benchmark”；
- 结果不应该依赖一长串隐藏假设才能解释。

这并不意味着所有题目都必须是传统语言学。LLM、开放 checkpoint、embedding、科学文档、结构化生成、模型训练动态、内部表征等都可以做，但必须依附在一个**本来就有科学意义的具体对象和自然问题上**。

---

# 2. 研究室当前活跃同门：研究方向、问题与选题来源

这一部分不是为了照抄同门题目，而是为了理解：

> **笹野研里一个题通常是怎么从已有工作自然长出来的。**

重点参考 M2 和已经形成论文级研究闭环的成员；本科生的题更多作为灵感池，不应直接拿来估计导师对修士研究的最低要求。

## 2.1 Hamdi：从“已有行为/表征”继续追问内部机制

### 方向 A：模型是否区分“真实实体”和“虚构实体”

直接种子来自 ICLR 2025 Oral 的《我知道这个实体吗？》。原工作研究模型内部是否存在“我认识这个实体 / 我能提取这个实体的事实”的知识觉知信号，并证明这种信号与拒答、幻觉行为存在因果关系。

Hamdi 没有继续重复“知道 / 不知道”，而是只横向移动一步：

```text
我知道这个实体吗？
        ↓
这个实体在模型内部被认为是真实存在，还是虚构的？
```

这里最关键的自然对照是：

- Harry Potter / Hogwarts 之类实体，模型明明非常“熟悉”；
- 但模型仍然应该知道它们不是现实存在的。

因此“熟悉度”和“现实存在性”可以天然分开。

之后导师进一步追问：如果只是 famous fictional entities，Wikipedia 本身就写着 fictional，那么“模型能区分”并不惊讶。真正值得研究的是：

> **模型到底依据什么判断一个实体是否真实？**

由此才出现更强的现象：陌生的真实实体容易失去“真实感”，而陌生的虚构实体并不会因此变成真实。研究逐渐从简单 probe 发展到 familiarity、knowledge、ontology 之间的非对称关系。

### 方向 B：随机选择时，模型内部有没有“现在要随机选”的状态

种子来自随机选择偏差与概率校准工作：LLM 在“随机选一个数字”“随便选一个颜色”“按 70%/30% 采样”时，经常存在强烈偏好。

已有工作更多问：

> **怎样让输出分布更接近目标分布？**

Hamdi 把问题改成：

> **模型内部有没有一个可读、可干预的状态，表示“当前任务要求我做 arbitrary/random choice”？**

之后形成 reader / writer / gated steering 的机制链条。

### Hamdi 的选题模板

```text
已有论文已经证明一个真实行为或内部信号存在
        ↓
不重复做 benchmark
        ↓
问一个相邻但更基础的内部问题
        ↓
probe / steering / patching 做因果验证
        ↓
最后回到行为解释或修复
```

值得学习的不是“做 mech interp”，而是：

> **机制研究之前，先有一个已经站稳的真实问题。**

---

## 2.2 Kurauchi：把 DLM 的“生成时间轴”和组合语义问题拼在一起

Kurauchi 当前已经从早期的汉字详细读法生成转向 **Diffusion Language Model（DLM）**。

现在的核心问题是：

> **在 DLM 的逐步去噪过程中，组合语义究竟什么时候形成？**

例如：

- 否定作用域到底什么时候确定；
- 时间关系到底什么时候确定；
- semantic information 是早于 lexical form 出现，还是同步出现；
- 正确生成与错误生成的语义形成轨迹是否不同。

最直接的种子是一类研究 **DLM 中 linguistic information 随 diffusion step 何时出现** 的工作。这些工作已经沿 denoising trajectory 测 POS、semantic category、exact token identity、commitment 等 token-level 信息。

Kurauchi 只往前走一步：

```text
token-level linguistic emergence
        ↓
句子级组合语义何时形成？
```

另一半灵感来自 AR LLM 的 compositional semantics / negation / temporal-order 机制工作：这些论文已经研究“组合意义分布在哪些 layer / component”，但没有 DLM 的显式生成时间轴。

因此他真正做的组合是：

```text
DLM 文献提供轴：diffusion time
+
组合语义文献提供对象：negation / temporal relation / sentence meaning
        ↓
组合语义在 diffusion time 上什么时候形成？
```

### Kurauchi 的选题模板

```text
论文 A 提供一个新的可测量轴
论文 B 提供一个重要但尚未沿该轴研究的对象
        ↓
把对象放到新轴上研究
```

这类题很适合我们继续找：**不是凭空创造现象，而是把一个已经成立的 measurement axis 移到一个邻近的重要对象上。**

---

## 2.3 Tsujimoto（M2）：Training Dynamics，把人类发展问题映射到模型训练时间

Tsujimoto 当前主线不是某一个具体词汇现象，而是：

> **模型在预训练过程中，是按什么顺序获得结构化语言能力的？**

最新题目是成对尺度形容词 / 反义词，例如：

- big / small；
- long / short；
- tall / short。

种子来自儿童语言习得研究。人类儿童往往更早掌握 positive / unmarked pole，例如 big、long，而 small、short 等 negative pole 更晚。相关心理语言学文献中本来就存在 acquisition-order 的竞争假设。

Tsujimoto 的移动非常简单：

```text
儿童年龄轴上的能力发展
        ↓
模型 checkpoint 轴上的能力发展
```

因此核心问题不是“模型会不会反义词”，而是：

1. 什么时候知道两个词属于同一个语义尺度？
2. 什么时候知道两者方向相反？
3. 什么时候能把这种关系用于组合推理？

### 选题轨迹也很有参考价值

他之前尝试过拟声拟态词 / sound symbolism 的 training dynamics，但发现语言、checkpoint、数据和测量都不够自然，于是没有继续硬救。

他保留了好的高层框架：

> **Training Dynamics**

但换掉了不合适的 phenomenon。

### Tsujimoto 的选题模板

```text
旧的人类认知 / 语言习得问题
        ↓
找到有大量公开 checkpoints 的模型
        ↓
把“年龄”换成“训练时间”
        ↓
比较能力的 emergence order
```

这是目前最值得借鉴的模板之一。

---

## 2.4 Kisako（M2）：从组内一篇 embedding 论文自然长出“维度 × 精度”的交互问题

Kisako 当前研究文本 embedding 压缩。

直接种子来自 Tsukagoshi & Sasano 关于文本 embedding 冗余、内在维度、降维的工作。原论文发现：不同下游任务对 embedding 维度的依赖程度非常不同，有些任务能极端降维而几乎不掉性能。

Kisako 看到后自然产生一个问题：

> **如果存储成本同时由“维度数”和“每维 bit 数”决定，那么在相同信息预算下，应该保留更多维度但每维更粗，还是减少维度但保留更高精度？**

因此变量不是简单的“降维”或“量化”，而是：

```text
存储预算 ≈ 维度数 × 每维 bit 数
```

之后系统比较不同模型和任务。

更值得学习的是：他后来查到“降维 + 量化”这个组合本身已经有人做过，于是没有硬 claim 方法新颖，而是把论文定位改成：

> **在统一 storage budget 下系统研究 dimensionality reduction 和 quantization 的交互。**

### Kisako 的选题模板

```text
读到一篇与自己非常接近的论文
        ↓
对其中一个被单独研究的变量产生自然的“另一个轴呢？”
        ↓
先跑小实验
        ↓
如果简单组合已被做过，就把贡献改成系统 interaction analysis
```

这说明：**好题并不一定来自方法创新，也可以来自一个干净、重要、尚未被系统研究的变量交互。**

---

## 2.5 Yano：从 FrameEOL 的“预测 frame”转向“模型是否真的具有 frame-like 语义理解”

Yano 的研究线长期围绕 Frame Semantics / FrameNet。

早期 FrameEOL 更偏方法：

> 如何利用生成式语言模型某个位置的 representation，更好地做 frame inference？

相关背景包括 prompt-based embedding 等工作。

之后研究问题逐渐倒过来：

> **如果 Frame Semantics 确实描述了人类语义理解的一部分，那么现代 LLM 是否真的获得了这些细粒度语义区分？**

于是出现 FrameBench。它不直接要求模型输出 FrameNet 标签，而是设计自然语言任务，让模型必须利用 frame-semantic distinction 才能答对。

现在遇到的主要问题是 frontier model 已经接近饱和，因此单纯“谁 accuracy 更高”不再有足够 insight。

### Yano 的选题模板

```text
上一项方法研究：能不能预测某个语言学结构？
        ↓
倒推一个更基础的问题：模型本身究竟有没有这种结构化理解？
        ↓
设计不显式暴露标签、但必须利用该结构才能完成的 diagnostic
```

同时也是一个重要警告：

> **只做 benchmark、最后得到“强模型更强”，在现在越来越不够。**

---

## 2.6 Han Yi：FrameNet / Frame Relation 的自动推断

当前主要研究：

> **LLM 能否自动推断 FrameNet 中 frame 之间的层级或语义关系？**

例如给出一个 frame，推断它在 Inheritance / Using 等关系下的 parent frame。

当前方法大致包括：

- embedding 缩小候选；
- LLM 生成 pseudo parent 或做关系推理；
- 分析模型是否真的利用 frame semantics。

一个很重要的新问题是 **训练数据泄漏 / ontology memorization**：

> 模型是在做语义层级推理，还是只是记住了 FrameNet 本身？

这条线属于研究室传统强项，题目来源更接近**长期 FrameNet 研究线内部演化**，而不是单篇 seed paper。

---

## 2.7 Guo：从多语言性能分析转向 Slide Narration Generation

Guo 过去做过 multilingual LLM performance 的分析，但当前主线已经转向：

> **根据 lecture slide / 视频生成 narration，并分析 narration 应该包含哪些信息。**

当前数据来自多种 lecture-slide / slide-video 资源，先做 slide 抽取、OCR、聚类，再把 narration 切成语义单元并做信息层级 / 信息类别标注。

这里的核心对象很具体：

> **slide 上可见信息、讲者补充信息、背景解释之间到底是什么关系？**

属于多模态教育场景中的结构化生成与信息内容分析。

---

## 2.8 Fuki Nakamura（本科生）：Typo / 字符级知识的 Training Dynamics

当前问题是：

> **模型在预训练过程中什么时候开始理解 typo？**

通过 Pythia 不同 checkpoint，对比正常词、打乱字符后的 typo、无关同长度词，观察 typo robustness 什么时候出现。

它同样属于 training dynamics，但由于本科阶段导师对研究深度要求相对低，主要把它作为**问题形式参考**，而不是修士研究 bar。

---

## 2.9 Fujiwara（本科生）：Emoji / 网络俚语语义

当前探索 emoji 的普通义、俚语义、隐喻义，例如 🔥、🤡、🧢、🐐 等在句子中的不同含义，以及 emoji 是否能替代某些词表达相同语义。

目前更像探索性选题阶段。

可作为“特殊但自然存在的语言现象”灵感来源，但不应直接拿来判断修士课题要求。

---

## 2.10 Han Ziyun：惯用语 literal / idiomatic meaning 的内部形成

当前研究模型如何区分惯用表达的：

- 字面义；
- 惯用义。

例如同一个表达在 literal context 和 idiomatic context 下，逐层看 hidden state 是否分离，并进一步考虑 activation patching。

属于：

```text
具体语言现象
+
内部表征 / 因果分析
```

比纯粹做 mechanistic interpretability 更符合研究室风格，因为研究对象先于机制方法存在。

---

## 2.11 Oshika：Citation Arrangement / Related Work 组织

研究科学论文 related work 中 citation 应如何组织，主要拆成：

- citation clustering；
- paragraph ordering；
- paragraph 内 citation ordering。

结合图方法、LLM、搜索 / 优化方法处理组合空间。

它体现的是研究室另一种典型风格：

> **先有一个真实、具体的学术写作结构问题，再选模型或算法处理它。**

---

## 2.12 当前同门给我们的总体启示

把上面几个高参考价值成员放在一起，可以看到不同题虽然表面差异很大，但结构很接近：

| 同门 | 真正的研究轴 | 不是在问什么 |
|---|---|---|
| Hamdi | 内部表征 → 因果行为 | 能不能简单 probe 出一个概念 |
| Kurauchi | diffusion time | DLM 会不会理解否定 |
| Tsujimoto | training checkpoints | 模型最终会不会反义词 |
| Kisako | 维度 × bit 的统一资源预算 | 哪个 compression method 单点更强 |
| Yano | 细粒度 frame-semantic distinction | frontier model accuracy 谁最高 |

共同规律：

> **先有一个自然问题，再找到一个很干净的“轴”，沿这个轴研究阶段、顺序、交互或机制。**

更重要的是，很多题都可以还原成：

```text
一篇很接近的 seed paper
+
只横着走一步
```

例如：

```text
Do I Know This Entity?
→ Is This Entity Real?
```

```text
DLM 中 token-level linguistic emergence
→ DLM 中 compositional semantic emergence
```

```text
儿童语言习得顺序
→ LLM training dynamics 中的习得顺序
```

```text
embedding dimensionality
→ dimensionality × quantization
```

因此后续选题搜索应重点复制这种**问题生成方式**，而不是复制他们的具体主题。

---

# 3. 研究室 / 导师真实偏好的研究风格

研究室邻域应该从真实工作中归纳，而不是从泛泛的“NLP”标签出发。

近期代表性对象包括：

- 语言模型中的字符级知识习得；
- semantic frame / frame induction / frame relation；
- 文本 embedding 的几何、冗余、内在维度和压缩；
- quiz answering、早期作答、clue 结构与难度；
- 人类与 LLM 难度差异；
- citation importance、related-work 组织；
- 科学文档信息抽取；
- slide narration 等特定结构化生成；
- 多语言性能因素；
- 模型内部语言 / 计算结构；
- 开放 checkpoint 上的具体能力习得轨迹；
- DLM 中的生成过程与语义形成。

最核心的共同点是：

> **模型出现之前，这个研究对象本身就已经有科学意义。**

例如：

- clue 顺序本来就是 quiz 的真实属性；
- semantic frame 本来就是语言资源对象；
- citation organization 本来就是科学写作结构；
- embedding compression 有意义，是因为任务语义必须在压缩后存活；
- acquisition order 有意义，是因为语言能力本身可以被直接测试；
- DLM 的 denoising time 是模型架构天然提供的新轴。

模型是实验系统或工具，而不是问题存在的唯一理由。

---

# 4. 优先搜索的题目范围

## 4.1 S 级：具体能力的 Training Dynamics / Development

研究室已经明显接受 checkpoint-rich 的能力发展研究。

优先找：

- 能力定义清楚；
- 可以自动大规模测试；
- 当前同门没有直接做；
- 重点是 acquisition trajectory，而不是 final accuracy。

有价值的现象包括：

- 突然出现；
- 非单调学习；
- 中途退化；
- 不同能力之间稳定的先后顺序；
- 某能力必须在另一 prerequisite 出现后才形成；
- 很早见过的数据模式，却很晚才真正形成能力；
- representation 先出现，但 causal use 更晚出现。

尤其推荐从**旧的人类发展 / 认知文献**中找有竞争假设的问题，再映射到 checkpoint 轴。

## 4.2 S 级：已有真实行为现象 → 内部机制

参考 Hamdi，而不是泛泛做 SAE / patching。

健康结构是：

```text
已有论文已经证明一个真实、重要的行为异常
        ↓
这个行为背后是否存在一个内部可读变量？
        ↓
这个变量是否真的因果控制行为？
```

避免：

```text
先随便找一个 hidden direction
→ 再给它补故事
```

## 4.3 S 级：新 measurement axis × 已知重要对象

参考 Kurauchi。

例如：

```text
论文 A 提供新的时间轴 / 层级轴 / 资源轴
论文 B 提供尚未沿该轴研究的重要现象
        ↓
研究该现象在这个轴上的形成、变化或失效
```

特别值得关注：

- diffusion time；
- pretraining time；
- post-training stage；
- model scale；
- context length；
- resource budget；
- interaction depth。

## 4.4 S/A 级：Embedding 结构、冗余、压缩及变量交互

优先找：

- 某一种具体语义能力在压缩下何时崩；
- 哪些任务极度脆弱，哪些极度不敏感；
- 不同语义关系是否需要不同有效维数；
- reconstructability 和 task utility 是否分离；
- 多个压缩轴之间的交互；
- 在统一资源预算下的 trade-off。

避免只做：

> “压缩后平均分下降了多少。”

## 4.5 A 级：科学文档、引用、学术写作结构

适合研究：

- citation placement；
- citation purpose；
- related-work organization；
- claim / reference 的局部对齐；
- scientific entity / method / dataset；
- 文档结构；
- citation chain 中可直接观察的结构性质。

优先使用**局部、结构化、可审计标签**。

如果核心标签需要：

```text
同一语义命题
+ 隐藏证据来源
+ 专家认识论判断
+ 穷尽文献审计
```

那么 measurement 本身很可能已经复杂到不适合个人项目。

## 4.6 A 级：Semantic Resources / Frame

可以找：

- frame coverage；
- 缺失 distinction；
- frame definition consistency；
- induction error；
- lexical-unit coverage；
- cross-resource disagreement；
- cross-lingual frame alignment；
- LLM 暴露出的系统性资源缺陷；
- 模型到底是记住 ontology 还是进行结构推理。

避免简单“让 LLM 构建一个更大 FrameNet”。

## 4.7 A 级：特殊结构化生成

重点不是普通生成质量，而是文本本身有明确结构约束的场景：

- 专利 claim；
- 法律 / 规章；
- scientific abstract / structured summary；
- definition；
- formal description；
- technical instruction；
- slide narration。

强问题通常围绕：

- 系统性结构违反；
- 信息遗漏 / 添加；
- 顺序约束；
- 指代一致性；
- 术语一致性；
- 逻辑依赖是否保留；
- 流畅性与领域有效性之间的错位。

## 4.8 A/B 级：LLM 作为旧科学问题的放大器

优先找 pre-LLM 时代因为成本高而没法大规模验证的问题，例如：

- 人工 annotation 太贵；
- taxonomy 要手工整理；
- 需要读成千上万篇文档；
- controlled stimulus 要人工编；
- 大量 human panel；
- 多资源 / 多语言比较过去成本太高。

健康结构是：

```text
旧科学问题
+ 历史上昂贵的测量方式
+ 经验证的 LLM 辅助扩展
        ↓
过去无法完成的大规模科学检验
```

不是：

> “LLM 替代标注员，准确率 92%。”

## 4.9 B 级：多语言 / 修辞 / 词汇现象

只有出现非常具体的现象才考虑，例如：

- 强跨语言反转；
- 一个明确资源 / 频率因素解释某种 gap；
- literal / figurative dissociation；
- 与具体语言结构绑定的系统错误；
- 多语言 acquisition-order 差异。

避免泛泛做“20 种语言上的 idiom benchmark”。

---

# 5. 明确不属于本目录的方向

除非出现非常具体的导师兴趣或语言 / 信息科学对象，否则不要把下面这些放进来：

- 通用 Agentic RL；
- tool-learning agent；
- web agent；
- 通用 reasoning RL；
- 泛化的 RL trace analysis；
- 机器人 / VLA 控制；
- world model / WAM；
- embodied policy mechanism；
- 泛化 continual learning；
- 脱离具体对象的 SAE / activation patching；
- 泛 mechanistic interpretability；
- 通用 RAG；
- 只比较 benchmark 分数；
- 没有具体语言 / 信息对象的优化算法论文。

这些默认放到 [`user_interest_topic_search`](../user_interest_topic_search/)。

---

# 6. 后续应该怎么找题

以后采用 **seed-paper-first + object-first**，而不是“关键词优先”。

错误流程：

```text
搜索“最新 NLP research gap”
→ 选一个流行关键词
→ 自己发明一个假设
```

正确流程：

```text
找到一篇强 seed paper
→ 精确写出它已经证明了什么
→ 找它留下的相邻问题
→ 只横着走一步
→ 再检查 2025–2026 是否已经撞车
→ 设计一个一击致命的首轮实验
```

## 6.1 三种最值得复制的题目生成方式

### 模板 A：已有现象 → 更基础的内部问题

参考 Hamdi：

```text
已有行为异常 / representation 已被证明存在
→ 它背后的内部状态是什么？
→ 这个内部状态是否因果控制最终行为？
```

### 模板 B：论文 A 的轴 × 论文 B 的对象

参考 Kurauchi：

```text
A：新的 measurement axis
B：重要的 scientific object
→ 把 B 放到 A 的轴上
```

### 模板 C：人类发展问题 → 模型 Training Dynamics

参考 Tsujimoto：

```text
心理学 / 语言习得文献中的 acquisition-order 问题
→ 开放模型 checkpoints
→ 测试模型是否出现相同 / 不同发展顺序
```

### 模板 D：单变量工作 → 自然变量交互

参考 Kisako：

```text
论文研究 X
→ 实际系统同时受 X 和 Y 影响
→ 在统一预算 / 同一目标下研究 X × Y
```

### 模板 E：上一项方法研究 → 更基础的能力问题

参考 Yano：

```text
怎么预测结构 X？
→ 模型到底有没有理解结构 X？
→ 设计不直接暴露标签的 diagnostic
```

---

# 7. 搜索与筛选流程

## 阶段 A0：Seed Paper 库

每篇强 seed 先记录：

```text
论文：
它已经证明了什么：
最强实验结果：
它提供了什么 measurement / dataset / axis：
作者没有回答的相邻问题：
为什么这个相邻问题不是简单重复：
```

## 阶段 A1：只允许“走一步”

对 seed paper 最多做以下几种变化：

- 对象换一个自然邻居；
- measurement axis 换一个已存在的新轴；
- final-model 变 training dynamics；
- behavior 变 mechanism；
- 单变量变自然 interaction；
- human developmental question 映射到 model development。

如果必须连续跨 3–4 步才能得到题，通常说明题太硬。

## 阶段 A2：自然性筛选

问：

```text
一句话能不能解释问题？
问题在没有我们的实验设计之前就存在吗？
数据是不是自然存在的？
是否必须构造奇怪的反事实样本才能识别？
导师听完会不会先问“为什么要这么构造”？
```

如果自然性不过，直接杀。

## 阶段 A3：同门碰撞审计

重点检查：

- Hamdi：实体 ontology / random-choice mechanism；
- Kurauchi：DLM compositional semantic emergence；
- Tsujimoto：adjective / antonym training dynamics；
- Kisako：embedding dimensionality × quantization；
- Yano / Han：Frame Semantics / FrameNet；
- Nakamura：typo training dynamics；
- Han Ziyun：idiom internal representation；
- Guo：slide narration generation。

不是只避免“同一个数据集”，而是避免核心 scientific question 重合。

## 阶段 A4：2025–2026 精确撞车检索

每个候选必须搜：

- exact scientific question；
- exact object；
- exact experimental contrast；
- seed paper 的后续引用工作；
- 原作者后续论文；
- 有没有已经沿同一个新轴研究过。

## 阶段 A5：首轮斩杀实验

首个实验最好是：

```text
同一个对象
只改变一个因素
一个直接结果
```

或者：

```text
开放 checkpoint
一个具体能力
看完整训练轨迹
```

或者：

```text
已有行为 phenomenon
一个 probe / intervention
看 representation 是否存在且是否 causally used
```

不要一开始就设计复杂 control stack。

---

# 8. 证据优先级

## S 级

- 已发表论文已经观察到的精确现象；
- 明确的人类 / 模型反差；
- 旧文献里明确存在的未解决问题；
- 已发布的数据 / 资源；
- 可复现的 checkpoint trajectory；
- 论文结果中直接出现但作者没解释的矛盾。

## A 级

- 一个强 published observation + 独立理由说明 extension 很重要；
- explicit limitation / future work 且数据已经存在；
- 相邻资源的明确冲突。

## B 级

- 合理的相邻空白；
- 一篇论文 + 另一个领域的类比。

只能做低成本探索，不能直接注册成主项目。

## C 级

- 纯 cross-paper bridge hypothesis；
- 从 aggregate result 猜 latent mechanism。

通常拒绝。

---

# 9. 候选题晋级硬条件

一个 lead 想从搜索日志晋级，必须同时满足：

1. **具体对象**已经明确命名。
2. 存在**外部锚点 / seed paper**。
3. **导师适配性**明确。
4. 已检查**同门撞题**。
5. 已检查 **2025–2026 最新撞题**。
6. **数据路径**已存在。
7. 核心 measurement 简单、可审计。
8. 有一个首轮实验能显著支持或杀死题目。
9. **正结果要让人兴奋**，不能只是干净。
10. 适合个人半年左右完成，不从数月 annotation 开始。
11. **null result 也能回答问题**，不是“再换模型试试”。
12. **没有复杂性臭味**：解释不能依赖越来越多 control。
13. 最好还存在自然的后续方法口，而不是“证明完了，然后呢？”

---

# 10. 现有失败题给这个目录的教训

## Topic16 / 17：问题自然，不代表 measurement 自然

引用、复现性等 meta-science 问题可以很有意义，但如果核心 label 需要穷尽式语义与 provenance 审计，就不适合个人项目。

科学文档类题应优先选择**局部可观察的结构**。

## Topic10：toy phenomenon 不够

如果真正有意义的 regime 根本构造不出来，就不要拿小规模 cute effect 硬续。

## Topic12：profile correlation 不是科学定律

两个 profile 有整体几何相似，不代表存在局部的一一对应机制。

## Topic05：如果定义对象需要越来越多排除，题可能本身不自然

当 gate / kill line 越来越复杂、需要证明“不是 A、不是 B、也不是 C”，往往说明问题已经离自然现象太远。

## Topic14 / 15：强 seed phenomenon 不等于我们的解释成立

应该先直接描述和验证现象，再加机制故事。

## Cross-lingual homograph 方向：反事实构造不能替代自然问题

如果 language × meaning 的某些格子在真实语言中本来就不存在，那么为了 identification 强行构造这些样本，会让实验本身比科学问题更难解释。

以后遇到类似情况应直接退回：

> **问题本身是否在自然数据中存在？**

---

# 11. 与个人兴趣轨道的边界

这个目录回答：

> **在导师 / 研究室环境里，什么题最自然、最稳、最有可能成长成论文？**

`user_interest_topic_search` 回答：

> **哪一个 frontier AI 问题即使不符合研究室当前风格，我个人仍然非常想做？**

不要混在一起。

| 题目形式 | 默认目录 |
|---|---|
| Agentic RL checkpoint 恢复 | user-interest |
| VLA action-chunk control | user-interest |
| reasoning-RL strategy collapse | user-interest |
| quiz clue information structure | advisor |
| semantic-frame coverage | advisor |
| embedding compression / capability | advisor |
| 具体语言能力的 acquisition trajectory | advisor |
| DLM 生成过程中的语言 / 语义形成 | advisor |
| patent / slide 等结构化生成 | advisor |
| 可结构化测量的 scientific-document phenomenon | advisor |

同一个 lead 只有在分别满足两套标准时才可以同时出现，不能靠换叙事强行跨目录。

---

# 12. 以后每轮导师向选题搜索必须记录的格式

每一轮必须包含：

```text
搜索范围：
检查的 seed papers：
每篇 seed 已经证明什么：
我们只横着走了哪一步：
检查的对象：
观察到的现象 / 历史问题：
同门精确撞题：
最新文献精确撞题：
保留的候选：
杀掉的候选及原因：
下一轮搜索分支：
```

每个保留 lead 必须记录：

```text
具体对象：
Seed paper：
Seed 已经证明什么：
我们只走哪一步：
自然问题：
为什么符合导师风格：
为什么科学上有意思：
现成数据：
核心 observable：
最近的同门工作：
精确文献撞车：
最便宜的斩杀实验：
如果成立，最强 headline：
后续方法 / 分析口：
状态：
```

搜索日志必须保留被杀掉的题，因为避免重复踩坑本身就是这个目录的目标。