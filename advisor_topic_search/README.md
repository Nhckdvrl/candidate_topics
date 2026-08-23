# 导师向选题搜索

这个目录用于记录**面向导师审查标准的研究选题搜索**。

最重要的一点先写在最前面：

> **研究室同门正在做什么，只是我们判断“什么样的问题容易成为一个好研究题”的参考样本，不是我们必须搜索的研究领域边界。**

我们不是因为 Han 做 FrameNet，就也去找 FrameNet；不是因为 Kisako 做 embedding，就继续找 embedding；也不是因为 Tsujimoto 做 training dynamics，就规定我们只能做 training dynamics。

真正需要学习的是：

> **这些题是怎么从已有论文、已有现象和已有测量中，自然地长出一个新的科学问题的。**

因此这个目录的核心原则是：

> **领域可以很宽，题型必须好。**

DLM、training dynamics、模型内部机制、知识、推理、embedding、多语言、科学文档、生成、甚至新的模型架构都可以搜索。关键不是“像不像研究室现有方向”，而是这个问题是否自然、具体、可验证、有明确的已有工作支点，并且得到正结果后真的值得高兴。

---

# 1. 我们到底要找什么样的题

理想的题目通常不是凭空 brainstorm 出来的，而是下面这种结构：

```text
一篇已经站得住的 seed paper / 一个已经被观察到的真实现象
        ↓
明确它已经证明了什么
        ↓
找到一个只往旁边走一步、但科学意义明显不同的问题
        ↓
确认这个问题本身自然存在，而不是为了实验硬造出来
        ↓
找到已有数据 / 模型 / checkpoint / measurement
        ↓
设计一个很简单、能明显支持或杀死问题的首轮实验
        ↓
如果结果为正，能形成一句让人觉得“这确实有意思”的 headline
```

这里最重要的不是方法多复杂，而是：

- **问题有没有外部来源**；
- **为什么前人做完以后自然会产生这个下一问**；
- **这个下一问是不是还没有被直接回答**；
- **结果无论正负是否都能回答一个真实问题**。

---

# 2. 同门研究只作为“好题是怎么长出来的”参考

下面记录同门，不是为了限定我们搜索的方向，而是为了抽象他们的**选题生成方式**。

## 2.1 Hamdi：从已有现象继续追问一个更基础的问题

### 例子一：从“我知道这个实体吗？”到“我认为它真实吗？”

seed paper 是 ICLR 2025 Oral《我知道这个实体吗？》。原论文已经证明：模型内部存在与“我是否认识这个实体、是否能提取其事实”相关的知识觉知信号。

Hamdi 没有重复 known / unknown，而是横着走了一步：

```text
模型知道这个实体吗？
        ↓
模型认为这个实体是真实存在，还是虚构的？
```

这个问题天然成立，因为 Hogwarts / Harry Potter 可以是：

- 模型非常熟悉；
- 但又不是真实存在。

所以 familiarity 和 ontology 可以自然分离。

随后导师又继续追问：如果 famous fictional entities 在训练数据里经常直接被标成 fictional，那么“能区分”本身并不惊讶。于是问题进一步变成：

> **模型究竟依据什么判断一个实体是否真实？**

这才长出后来更有意思的陌生真实实体 / 陌生虚构实体的不对称现象。

### 例子二：从“随机选择有偏”到“模型内部有没有随机选择状态？”

已有工作已经发现：让 LLM 随机选数字、颜色或按照指定概率分布采样时，会存在系统偏差；不少论文研究怎么校准这种行为。

Hamdi 没有再提出一个新的校准算法，而是问：

> **模型内部有没有一个状态，表示“现在这个任务要求我做 arbitrary/random choice”？**

于是才有 reader、writer、steering 等后续机制分析。

### 真正值得借鉴的不是 mech interp，而是这个结构

```text
前人已经证明一个行为 / representation 存在
        ↓
找到一个相邻但更基础的问题
        ↓
先证明问题存在
        ↓
再做内部机制或干预
```

---

## 2.2 Kurauchi：论文 A 提供“轴”，论文 B 提供“对象”

Kurauchi 当前研究 DLM 中组合语义如何沿 diffusion process 形成。

一类已有 DLM 工作已经研究：

> POS、semantic category、token identity 等 token-level 信息，在不同 diffusion step 什么时候出现？

另一类 AR LLM 工作已经研究：

> negation、temporal relation、compositional meaning 在 Transformer layer / component 中怎样形成？

Kurauchi 做的不是凭空造新任务，而是把两条线拼起来：

```text
DLM 文献提供新的时间轴：diffusion step
+
组合语义文献提供重要对象：negation / temporal relation / sentence meaning
        ↓
组合语义在 diffusion time 上什么时候形成？
```

也就是：

```text
token-level emergence over diffusion time
        ↓
sentence-level compositional emergence over diffusion time
```

### 值得借鉴的题型

> **论文 A 已经给了一个可靠的新 measurement axis，论文 B 有一个重要 scientific object，但还没人沿这个 axis 研究它。**

这种“轴 × 对象”的一格空位，比凭空猜机制可靠得多。

---

## 2.3 Tsujimoto：把一个已有的人类发展问题映射到模型训练过程

Tsujimoto 的核心不是“研究反义词”，而是 training dynamics。

当前题目的 seed 来自儿童语言习得：儿童对 big / small、long / short 等尺度形容词的掌握存在发展顺序，而且心理语言学文献本身已有竞争假设。

他做的转换很简单：

```text
儿童年龄
        ↓
模型训练 checkpoint
```

于是问题变成：

> **模型在预训练过程中，是按什么顺序学会同一语义尺度、反向关系以及组合推理的？**

更值得注意的是，他之前尝试过 onomatopoeia / sound symbolism 的 training dynamics，但发现数据、语言和 measurement 不够自然后，没有为了保题而继续硬造实验。

他保留了好的研究框架：

> training dynamics

但换掉了不自然的 phenomenon。

### 值得借鉴的题型

```text
一个已有的人类认知 / 语言发展问题
+
公开 checkpoint 让“模型的发展过程”变得可观察
        ↓
比较能力 emergence order / dependency / regression
```

这里真正重要的是**旧科学问题本来就存在**，而不是“checkpoint 很多所以想画曲线”。

---

## 2.4 Kisako：从单一变量自然推出变量交互

Kisako 的 seed 来自组内关于文本 embedding 降维、冗余和内在维度的论文。

原论文主要研究 dimension。

他看完以后自然产生：

> **embedding 的存储成本不只是维度数，还取决于每一维用多少 bit；同样 storage budget 下，到底该保留更多维度还是更高精度？**

于是从：

```text
dimension
```

变成：

```text
dimension × precision
```

后来他发现“降维 + 量化”这个组合本身已有工作，因此没有硬说“方法组合很新”，而是重新定位成：

> **在统一 bits × dimensions 预算下，系统研究两者如何交互。**

### 值得借鉴的题型

```text
前人把一个变量研究得很清楚
        ↓
发现现实系统里另一个同样基础的变量一直被固定
        ↓
问两个变量的 interaction 是否产生非平凡规律
```

关键不是“多加一个变量”，而是这个 interaction 必须对应一个真实问题。

---

## 2.5 Yano：从“能不能完成任务”倒推“模型究竟懂不懂这个对象”

Yano 早期 FrameEOL 更偏方法：如何利用 LM representation 做 frame inference。

后来的 FrameBench 则把问题倒过来：

> **如果 Frame Semantics 描述了人类语义理解中的真实 distinction，那么现代 LLM 是否真的获得了这些 distinction？**

因此不再要求模型直接输出 FrameNet label，而是设计自然语言问题，让模型必须利用这种 semantic distinction 才能回答。

这条线后来又遇到 frontier model 接近 99% 的问题，也说明：

> **“做一个 benchmark，然后强模型更强”本身不是足够有意思的科学结果。**

### 值得借鉴的题型

```text
已有任务 / 方法假定模型掌握某种能力
        ↓
把任务标签拿掉
        ↓
直接问模型是否真正拥有背后的结构化能力
```

---

# 3. 从同门案例里真正应该提炼出的六种“好题形状”

同门方向不同，但他们真正有参考价值的是下面这些题型。

## 3.1 已知现象 → 更基础的相邻问题

形式：

```text
Paper: X phenomenon exists
我们: what exactly is the underlying distinction / state / cause?
```

例如：

```text
知道 / 不知道
→
真实 / 虚构
```

适合来源：

- surprising behavioral result；
- error analysis；
- representation finding；
- 已有 mitigation 但机制未知的现象。

这是 Hamdi 型。

---

## 3.2 一个成熟 measurement axis × 一个还没沿这个轴研究的重要对象

形式：

```text
Paper A: 新轴 / 新测量方式
Paper B: 重要对象
        ↓
对象在这个轴上是什么结构？
```

例如：

```text
diffusion time
×
compositional semantics
```

其他可能的轴包括：

- pretraining checkpoint；
- model scale；
- diffusion step；
- post-training stage；
- context length；
- compression budget；
- intervention strength。

这是 Kurauchi 型。

---

## 3.3 旧科学问题 → 利用现代模型第一次大规模重新检验

形式：

```text
心理学 / 语言学 / 认知科学已经争论的问题
        ↓
过去受数据或实验成本限制
        ↓
open checkpoints / LLM 让新的实验成为可能
```

重点不是“LLM 是否像人”，而是：

> **旧文献里本来就存在一个明确 competing hypothesis。**

这是 Tsujimoto 型。

---

## 3.4 单轴规律 → 两个基础变量的 interaction

形式：

```text
变量 A 已被系统研究
变量 B 在真实系统里同样基础
        ↓
固定总资源 / 总约束，研究 A × B
```

要求 interaction 本身有实际或科学含义，而不是为了多一个二维图。

这是 Kisako 型。

---

## 3.5 “模型会做” → “模型实际上是否拥有背后的能力”

形式：

```text
某 benchmark / downstream task 上表现好
        ↓
这个成绩可能由 shortcut 得到
        ↓
设计更直接的 diagnostic，问真正的 latent capability 是否存在
```

但必须防止最后又退化成新 benchmark。

这是 Yano 型。

---

## 3.6 已经出现的矛盾 / dissociation → 直接研究为什么两件本应一致的东西分开了

这是我们后续搜索时应当**最高优先级**关注的一类。

形式例如：

```text
模型内部已经知道 X
但行为却不使用 X
```

```text
能力 A 已经形成
能力 B 理论上依赖 A
但 B 晚很多才出现
```

```text
模型规模增大后 representation 更清楚
但最终行为反而更差
```

```text
context 中已有足够信息
模型仍选择旧知识 / shortcut
```

这类题的优势是：

- 问题由已有结果直接暴露；
- 正结果往往天然反直觉；
- 不需要我们自己先假设一个复杂机制；
- 后面有明显的解释 / 方法口子。

---

# 4. 因此，我们不应该把搜索范围限制在“研究室领域”

研究室同门只是正例，不是边界。

可以继续广泛搜索：

- training dynamics；
- DLM；
- 模型知识与知识觉知；
- representation–behavior dissociation；
- planning；
- reasoning；
- post-training 前后行为变化；
- embedding / representation；
- multilingual；
- semantic / compositional phenomena；
- scientific document / structured generation；
- 新模型架构中特有的可测量过程。

但不能因为某个领域热门就直接形成题目。

必须先回答：

> **是哪篇论文、哪个结果、哪个旧科学问题，逼出了这个下一问？**

如果没有这个外部支点，就先不注册为候选。

---

# 5. 最优先寻找什么样的 seed paper

以后扫 2025–2026 论文，不要只看标题和 future work，要重点找下面几类。

## A. 论文已经发现了一个奇怪但没解释的 subgroup / failure

例如：

- 某类实体出现明显非对称；
- 大模型反而在某组更差；
- 某能力先出现但不能被行为使用；
- 某个 intervention 对不同条件方向相反。

这类最容易长成新的科学问题。

## B. 论文第一次提供了一个新的 measurement axis

例如：

- diffusion trajectory；
- dense pretraining checkpoints；
- post-training stages；
- causal feature tracking；
- model family / scale trajectory。

不要重复论文原来的对象，去找一个重要邻接对象。

## C. 论文已经把一个现象证明得很扎实，但只回答了“有没有”

然后问：

- 什么时候形成？
- 形成顺序是什么？
- 为什么在某条件下不被使用？
- 和另一个本应相关的能力是否同步？
- 是 representation 改变还是 readout 改变？

## D. 老领域存在明确竞争假设，但以前难以大规模测量

尤其适合：

- psycholinguistics；
- cognitive science；
- language acquisition；
- information science；
- human decision-making。

但必须是真的旧科学问题，不是我们自己给 LLM 编一个“像不像人”的故事。

---

# 6. 什么样的题应该快速杀掉

下面这些即使看起来新，也不要恋战。

1. **为了 identification 必须造不自然 counterfactual。**
2. **只因为文献矩阵里有个空格，就觉得是 gap。**
3. **最终最可能得到“XX 会影响性能”，然后没有下一步。**
4. **必须先证明很多“不是 A、不是 B、不是 C”，才能解释主结果。**
5. **需要越来越多 control 才能维持 claim。**
6. **主要贡献只是换模型 / 换语言 / 换 benchmark。**
7. **强模型更强、模型越大越好之类完全符合直觉的结果。**
8. **核心现象没有公开数据，必须先做几个月 annotation。**
9. **直接照搬同门，只换 lexical item / language / dataset。**
10. **问题只有方法意义，没有一个本来就值得问的科学对象。**

特别记住：

> **如果 gate 和 kill line 越设计越复杂，往往不是实验更严谨，而是问题本身正在离自然语言和真实现象越来越远。**

---

# 7. 一个候选题进入正式列表前必须回答的问题

每个候选都必须写清楚：

```text
Seed paper / 外部来源：
前人已经证明什么：
前人的结果里哪一点自然地产生下一问：
我们的“一步移动”是什么：
一句话 Research Question：
为什么问题本身自然存在：
为什么结果不是“果然如此”：
最接近的同门是谁，是否撞题：
2025–2026 最接近论文是谁，是否已经做过：
已有数据 / 模型 / checkpoint：
第一枪实验是什么：
什么结果直接 kill：
如果为正，最强 headline 是什么：
正结果之后还有什么明显方法 / 分析口子：
```

如果“前人的结果里哪一点自然地产生下一问”这一项写不出来，通常说明这个题还是脑补出来的。

---

# 8. Promotion Gate

一个题只有同时满足下面条件，才值得真正写代码。

### 1. 自然性

不需要解释很久，就能让人理解为什么这个问题存在。

### 2. External Anchor

有明确 seed paper、published anomaly、旧科学问题或公开数据支撑。

### 3. 一步距离

和 seed 足够接近，可以复用已有 measurement / data / setting；但 scientific question 明显不同。

### 4. 非平凡性

正结果不能只是“当然会这样”。

### 5. 简单首轮实验

第一枪应当能在较短时间内明显提高或降低我们对题目的信心。

### 6. 可杀性

如果核心结构不存在，可以直接停止，不靠 tuning 续命。

### 7. 低控制复杂度

核心结论最好用 1–3 个关键对照就能解释，而不是十几个控制。

### 8. 数据已经基本存在

不要一上来进入大规模人工构造。

### 9. 撞题可审计

可以明确搜索 exact question、exact axis、exact contrast。

### 10. 正结果之后还有路

现象成立以后，最好还能自然接：

- mechanism；
- causal intervention；
- mitigation；
- broader generalization；
- theoretical explanation。

不能是“证明完了，然后呢？”

---

# 9. 后续搜索的标准流程

以后每一轮都按下面做，而不是一次丢几十个脑暴题。

## Stage 1：广泛扫 seed papers

重点看：

- ACL / EMNLP / NAACL / EACL / TACL；
- ICLR / ICML / NeurIPS；
- 2025–2026 arXiv 强工作；
- 相邻心理学、认知科学、信息科学工作；
- 论文 appendix / error analysis / ablation。

## Stage 2：每篇只写“它真正证明了什么”

不要急着想方法。

## Stage 3：只允许做“一步移动”

优先尝试：

```text
有没有 → 什么时候
最终模型 → training trajectory
behavior → internal state
internal state → causal use
scale → training time
single variable → meaningful interaction
new measurement axis → adjacent important object
human developmental question → model developmental question
```

## Stage 4：先做 collision search

查：

- 同门是否正在做；
- seed paper 作者后续是否已经做；
- 引用 seed 的论文是否已经做；
- 2025–2026 exact question 是否出现。

## Stage 5：才设计第一枪

目标不是“做出漂亮结果”，而是：

> **最快判断这个问题到底有没有值得继续研究的结构。**

---

# 10. 当前从同门经验得到的最核心判断

研究室并没有一个固定的“导师只喜欢这些方向”的领域列表。

从实际同门可以看到：

- Hamdi 可以做 mechanistic interpretability；
- Kurauchi 可以做最新 DLM；
- Tsujimoto 可以做 training dynamics + psycholinguistics；
- Kisako 可以做 embedding efficiency；
- Yano 可以做 semantic benchmark / language science。

这些题在领域上差别非常大。

真正共同的是：

> **它们通常都有一个清楚的 seed，有一个自然的一步 extension，有一个具体可测的对象，而且导师可以直接追问“前人做到哪里、你到底多问了什么”。**

因此以后不要再问：

> “这个方向像不像研究室？”

而应该问：

> **“这个问题是怎么从一篇已经站稳的工作里自然长出来的？”**

以及：

> **“如果它被证明，我们到底有多值得高兴？”**

这两个问题才是这个目录真正的选题标准。

---

# 11. 每轮搜索日志格式

每一轮至少记录：

```text
本轮搜索范围：
Seed papers：
每篇已经证明的核心事实：
从结果自然产生的下一问：
一步 extension：
碰撞论文：
同门碰撞：
保留候选：
杀掉候选及原因：
下一轮继续搜索的分支：
```

每个保留候选必须额外记录：

```text
题目：
Seed：
自然问题：
为什么值得高兴：
已有数据：
最便宜的决定性实验：
Kill line：
正结果后的方法 / 机制口子：
状态：
```

这个目录的目标不是积累尽可能多的候选，而是**不断留下少数真正自然、具体、站得住的问题，并完整记录为什么其他题被杀掉。**
