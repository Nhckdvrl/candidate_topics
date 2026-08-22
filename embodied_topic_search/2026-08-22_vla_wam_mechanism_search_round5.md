# 2026-08-22：VLA / WAM 机制选题搜索日志（第五轮）

> 状态：**继续搜索。正式 Topic 仍不新增。**
>
> 本轮最重要的变化不是又找到一个“小空白”，而是重新校准候选题的**问题尺度**：实验可以窄、screening 可以狠，但研究问题本身必须达到 ICLR / ICML / NeurIPS / RSS / CoRL / ICRA / IROS 常见的自然问题尺度。如果为了绕开 collision，只能把标题不断缩成某个 dataset、operator、delay 或单一 statistic，这个题就不做。

---

# 0. 新增硬标准：题目不能靠“越压越窄”活下来

前四轮一直强调：

- phenomenon-first；
- collision-first；
- prerequisite cheap；
- mechanism 后必须有 method space；
- 不依赖复杂 gate 救题。

第五轮再加一条：

> **题目的 scope 本身必须像一篇顶会论文的问题，而不是像一个 ablation 的标题。**

参考近期实际题目：

- `Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?`
- `What Matters for Batch Online Reinforcement Learning in Robotics?` — ICLR 2026
- `HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy` — ICLR 2026
- `Self-Improving Vision-Language-Action Models with Data Generation via Residual RL` — ICLR 2026
- `Action-Constrained Imitation Learning` — ICML 2025
- `Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism` — ICML 2025
- `Action-Free Reasoning for Policy Generalization` — CoRL 2025
- `SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies` — CoRL 2025

这些题目可以很 specific，但它们都在问一个**自然存在、可独立理解的问题**：

```text
Why does X work?
What matters for Y?
Can policies do Z?
How should we learn under constraint C?
What breaks when assumption A is violated?
```

它们不是：

```text
Does metric M differ by 0.2 under subgroup S?
Does operator pair A/B increase history gain?
Does lag d=8 correlate with chunk horizon H=16?
```

后者可以是第一枪，但不能成为研究问题。

以后如果出现：

> broad version 已被做掉 -> narrow one variable -> narrow one dataset -> narrow one subset

这种连续收缩，直接视为**选题质量下降信号**，而不是“novelty 更精确”。

---

# 1. Round 4 最强线重新定标题尺度

Round 4 最后形成的是：

> `memory gain -> producer / strategy disambiguation`

如果把它直接压成：

> same-proficiency producer mixture 是否产生 extra history gain？

题目太窄。

这个实验仍然可以作为 prerequisite，但它不能定义论文。

真正尺度合格的问题应该是：

# **What Does History Actually Model in Robot Policies?**

这个问题已经单独摘到：

`topic_search_logs/candidates/what_does_history_actually_model.md`

状态仍然只是 **PROVISIONAL SEARCH CANDIDATE**，没有注册成根目录 Topic。

它是从：

- Round 3：action chunking × demonstrator timing；
- Round 4：PH-vs-MG audit / producer mixture；

连续演化出来的。

---

# 2. 继续 collision：producer-mixture non-Markovianity 本身已经不能拿

本轮找到一个非常关键的理论近邻：

## Interactive and Hybrid Imitation Learning: Provably Efficient Alternatives to Interactive Expert Feedback

NeurIPS 2025。

论文在分析 first-step mixing policy 时明确指出：

> episode 开始时随机选择一个 Markov policy，并在整条 trajectory 中持续使用它；即使 mixture 中的每个 policy 都是 Markovian，得到的 mixture policy 一般也不再 Markovian，因为过去 action/history 会携带关于 latent policy identity 的信息。

这基本就是 Round 4 的数学直觉：

```text
latent producer identity
+
trajectory-level policy mixture
-> aggregate history dependence
```

因此以下结论已经不能拿：

> “我们首次发现多个 Markov demonstrators 混合后会产生 non-Markovianity。”

理论上已经非常明确。

这进一步提高了 `What Does History Actually Model?` 的 bar。

如果它能活，贡献必须不是证明 mixture theorem，而是回答：

> **现实的 modern robot / VLA memory 到底在多大程度上实际利用这种 behavior-source information，它对 learned representation 和 closed-loop behavior 有什么后果？**

---

# 3. 更老的邻近线：history 做 system identification 也不是新概念

继续往 control / RL 邻域搜索又撞到一个经典思想：

## Preparing for the Unknown: Learning a Universal Policy with Online System Identification

RSS 2017。

这里 history 的作用非常明确：

```text
recent states + actions
-> infer latent dynamics parameters
-> condition policy
```

也就是说：

> history 不只是“记过去发生了什么”，还可以用于识别当前 episode 背后的 latent generating process。

这和 producer inference 不是同一个变量：一个是 physical dynamics context，一个是 behavior-policy context；但 computation template 很近。

另外早期 Learning-from-Demonstration 文献已经研究 policy recognition、latent intention、multi-modal demonstrations。

所以如果 Candidate A 最后只是说：

> history can infer latent context

仍然不够。

---

# 4. 为什么 Candidate A 还没有直接被砍

虽然基本数学和邻近机制都存在，但 2026 的 robot foundation model 出现了一个以前没这么强的现实组合：

```text
A. policy 获得越来越长的 memory/history
B. dataset 同时变成越来越 heterogeneous 的 behavior mixture
```

## A. Memory side

### HAMLET — ICLR 2026

把 GR00T N1.5 / N1.6 一类 pretrained VLA 改成 history-aware policy。在真正 history-dependent 的 real-world task 上 gain 很大。

### MemoryVLA — ICLR 2026 / MemoryVLA++

明确引入 working / episodic memory 和 future imagination。

### Big Picture Policies — 2026

直接发现 naive history conditioning 会 latch onto spurious correlations，因为 possible-history space coverage 爆炸。

## B. Heterogeneous-data side

### π0.7 — 2026

直接把：

- strategy；
- episode quality；
- control modality；
- subgoal image；

作为 conditioning context，用来消除 large diverse dataset 中的 behavior ambiguity；同时训练 demonstration、suboptimal autonomous data、failures、non-robot data。

这意味着：

> “history representation 的统计内容”在今天不是纯粹的 memory architecture 问题，而和 heterogeneous training-data semantics 绑定。

目前仍没有查到一篇现代 VLA 机制工作直接回答：

> 当 history-aware VLA 在 mixed behavior data 上训练时，memory 中有多少是 task/world belief，有多少是 strategy/source/context inference？这种分工是否影响 deployment generalization？

这就是 A 目前还保留的唯一理由。

但必须强调：**这是一个尚未完全 collision 的大问题，不是已经确认的 research gap。**

---

# 5. Candidate A 的升级 / kill 条件重新定义

不再要求它靠一个 robomimic number 变成论文。

## 可以升级的情况

只有当 preliminary evidence 支持一个**跨 setting 的 general statement**，例如：

> history-aware robot policies systematically allocate substantial predictive/causal capacity to behavior strategy/context, not only task state;

并进一步发现：

> this helps in-distribution imitation but creates a predictable generalization failure / ambiguity under changed producer or strategy structure.

这才是顶会尺度结果。

robomimic same-proficiency mixture 可以用来快速 screening 这个机制是否存在，但后面必须至少跨到：

- 一个 modern memory/history VLA；
- 一个不同类型的 heterogeneous source axis；
- rollout-level causal consequence。

## 应该砍的情况

如果最后的 strongest statement 只能是：

> “two human operators mixed together make an RNN more useful”

即使数字非常显著，也砍。

如果为了躲 CoRL multi-human IL、NeurIPS mixture theory、BPP history shortcut，最后只能继续限定到：

```text
same proficiency
+ fixed task
+ one dataset
+ one history length
```

也砍。

---

# 6. Autonomous self-generated data：题目尺度对，但目前不进入 shortlist

Round 4 的另一条新线是：

> robot policy 不断学习自己当前 / 上一代产生的 autonomous data，会不会像生成模型 self-consuming loop 一样，把自己的 behavior support 越学越窄？

如果题目能立，尺度应该类似：

# **Does Autonomous Robot Data Narrow Behavioral Support?**

这是一个合格的 research-question 尺度。

但本轮继续 collision 后，我暂时**不把它单独摘进 candidates/**。

原因不是题目太宽，而是已有 evidence 还不能证明我们需要的 anomaly。

---

## 6.1 What Matters for Batch Online RL 已经把 autonomous-data difficulty 做得很深

ICLR 2026：

**What Matters for Batch Online Reinforcement Learning in Robotics?**

它研究的就是：

```text
current policy
-> collect autonomous batch
-> learn from accumulated autonomous data
-> improve policy
```

关键观察包括：

- imitation / filtered imitation 容易快速停在 suboptimal point；
- Q-function 对有效利用 autonomous data 很关键；
- expressive policy 很重要；
- temporally correlated noise 提高 exploration diversity，可以继续带来 gain。

因此：

> “autonomous data 需要 diversity / exploration，否则 improvement 会停”

不能作为新问题。

---

## 6.2 PLD 已经非常接近 failure-support / recovery coverage

ICLR 2026：

**Self-Improving Vision-Language-Action Models with Data Generation via Residual RL**

PLD 不只是说 autonomous data 有用，而是专门：

```text
base VLA probes
-> residual specialist takes over around failure regions
-> collect recovery behavior
-> distill back to VLA
```

其核心动机之一就是 ordinary demonstrations / ordinary policy rollouts 对 deployment failure neighborhood 覆盖不足。

所以：

> “普通 autonomous rollout 会漏掉 recovery states，要主动 probe failure region”

也已经非常接近被做完。

---

## 6.3 因此 B 当前缺的是真实“generational ratchet”现象

从 generative-model model collapse 迁移来的真正新版本是：

```text
π0 -> D0_auto -> π1 -> D1_auto -> π2 ...
```

如果没有 external data / intervention，每代 autonomous visitation 是否会**方向性地**缩小某些 rare but useful state / behavior support，使后代更难重新发现？

这比一次性的 coverage 问题强。

但目前我们还没有找到：

- 多代 robot self-training 中明确报告 support contraction；
- alternate strategy / recovery behavior 随 generation 单调消失；
- 一个像 generative model tail-collapse 那样已存在的 robot anomaly。

因此按 phenomenon-first 原则：

**现在不进入 shortlist。**

不能因为语言模型 / image model 有 self-consuming collapse，就默认机器人也有。

---

# 7. 一个重要的反思：好的题不是“碰撞以后剩下的集合差”

前几轮很容易形成这种思维：

```text
大问题 A 已有人做
-> 减去他们做的变量 x
-> 剩下 A\x
-> 再查 collision
-> 再减一个变量
```

这样最后总能找到一个形式上没人完全做过的小区域。

但这不是科研选题。

顶会题通常反过来：

```text
一个自然的大问题 / anomaly
-> 有足够 evidence 表明它是真的
-> 一组简单实验把核心机制立住
-> mechanism 导出方法
```

而不是：

```text
literature complement set
-> 证明没人做过
-> 再想为什么重要
```

以后 novelty audit 的目的不是寻找“文献集合的缝”，而是决定一个**本来就值得问的问题**是否还有贡献空间。

---

# 8. 第五轮当前状态

目前只单独摘出一个 provisional candidate：

## **What Does History Actually Model in Robot Policies?**

来源：Round 3 temporal anomaly → Round 4 producer-mixture reinterpretation → Round 5 scope / collision audit。

它还没有过关，只是问题尺度合格且存在真实前沿 tension。

`Does Autonomous Robot Data Narrow Behavioral Support?` 暂不进入 candidates，因为目前缺少独立的 generational anomaly，且 batch-online RL / PLD 已经覆盖相邻一级问题。

其余 Round 1–4 已砍掉的 architecture / chunking / grounding / cross-embodiment / WAM optimism / phase / recovery 等方向不重新包装。

---

# 9. 下一轮搜索重点

下一轮对 Candidate A 只做两件事：

### 9.1 Collision-to-kill

继续查有没有工作已经直接：

- disentangle task memory vs demonstrator/strategy memory；
- 给 modern VLA memory 做 source/style intervention；
- 证明 history-aware robot policy 用 history 识别 latent behavior mode；
- 研究 heterogeneous-data metadata 与 memory 的替代关系。

一旦有人已经把这条闭环，直接砍。

### 9.2 Existence evidence

不急着写大工程 harness。优先找公开数据 / checkpoint 中是否已有现成迹象：

- multi-source data 上 history gain 明显大于 homogeneous data；
- strategy metadata 与 history 有可替代性；
- history intervention 会在 current scene 不变时切换 behavior mode；
- modern memory VLA 在 source/style shift 下出现可预言 failure。

如果连存在性证据都找不到，不因为标题漂亮就继续。

同时继续从相邻领域、开源整模型、作者博客和数据文档找新的**自然大问题**，而不是围着 Candidate A 无限雕刻。
