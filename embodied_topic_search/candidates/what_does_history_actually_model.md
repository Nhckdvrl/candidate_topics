# What Does History Actually Model in Robot Policies?

> Status: **PROVISIONAL SEARCH CANDIDATE — not a registered Topic.**
>
> 这个文件只是把已经从搜索日志中审出来、值得继续 collision 的问题单独摘出来。它不进入根目录正式 Topic，也不因为进入这里就默认成立。

---

## 1. 问题

> **What does history actually model in robot policies?**

现代机器人 policy 越来越多地加入 observation history、episodic memory、moment tokens、long-context transformer 或 memory bank。通常的解释是：机器人任务本身具有 temporal dependency，policy 需要记住过去的世界状态、交互结果和任务进度。

这个解释当然成立于很多任务，但它可能不是 history gain 的全部来源。

机器人训练数据同时越来越 heterogeneous：

- 不同 human operators；
- 不同 proficiency / motion styles；
- teleoperation 与 autonomous rollouts；
- 多个 RL / specialist checkpoints；
- synthetic / MimicGen trajectories；
- success / failure / recovery trajectories；
- 不同 strategy、speed、control modality。

因此 history 还可能在做另一件事：利用过去轨迹推断**当前 episode 正在遵循哪一种 behavior mode / strategy / producer**。

问题不是“history 能不能分类 demonstrator”，而是：

> **现代 robot policy 中实际观察到的 history / memory benefit，到底主要在表示什么？它是 world/task state 的记忆，还是有一部分来自 training-data behavior structure？**

这是本文档保留的顶层 research question。

---

## 2. 这个题从哪里挑出来的

### Round 3：action chunking × demonstrator timing

来源：

`topic_search_logs/2026-08-22_vla_wam_mechanism_search_round3.md`

Round 3 从 2026-08-03 的：

**Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?**

出发。该工作发现，在 human demonstrations 中，过去 observation 有时比当前 observation 更容易预测当前 action。

最初的问题是：

> 这种 non-Markovianity 来自 task 本身，还是来自 human reaction / teleoperation / control pipeline？

### Round 4：PH-vs-MG natural contrast 被重新审计

来源：

`topic_search_logs/2026-08-22_vla_wam_mechanism_search_round4.md`

Round 4 发现原先计划使用的 robomimic PH-vs-MG contrast 并不干净：MG 本身是多个 SAC checkpoints 的 rollout mixture。

这暴露了一个更一般的机制：

即使每一个 producer 单独都是 Markov policy，多个 producer 混合后，如果 producer identity 没有被显式提供，history 也可能通过识别当前 trajectory 的 behavior mode 来提高 action prediction。

于是问题从：

```text
human reaction delay
```

升级成：

```text
what information is robot policy memory actually using?
```

因此这个候选是从 **Round 3 的 temporal anomaly** 和 **Round 4 的 producer-mixture audit** 连续演化出来的。

---

## 3. 为什么这个问题尺度合格

我们以后不再为了 novelty 把题目压成某个微型统计 effect。

参考最近顶会的题目尺度：

- `Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?`
- `What Matters for Batch Online Reinforcement Learning in Robotics?`（ICLR 2026）
- `HAMLET: Switch Your Vision-Language-Action Model into a History-Aware Policy`（ICLR 2026）
- `Long-Context Robot Imitation Learning by Focusing on Key History Frames`
- `Action-Constrained Imitation Learning`（ICML 2025）

这些工作的共同点是：**主问题是一个自然、独立、领域级别的问题；具体 ablation / decomposition 是验证手段，而不是题目本身。**

所以如果这个候选最后只能写成：

> “same-proficiency two-operator mixture increases RNN gain”

那就应该砍，而不是把它包装成论文。

我们真正要求它最后能站住的是：

> **What does history actually model in robot policies?**

---

## 4. 为什么这个问题现在特别 relevant

### MemoryVLA / MemoryVLA++

- https://github.com/Somprat/myMemoryVLA
- https://arxiv.org/abs/2606.09827

这条工作线把 memory 明确解释为 robot temporal dependency：working memory、episodic memory、future imagination。

### HAMLET — ICLR 2026

- https://iclr.cc/virtual/2026/poster/10010110
- https://myungkyukoo.github.io/hamlet/

HAMLET 给 pretrained VLA 加 history-aware memory，并在真正需要历史的 long-horizon tasks 上获得很大提升。

### Big Picture Policies

- https://bigpicturepolicies.github.io/
- https://arxiv.org/abs/2602.15010

BPP 发现 naive raw-history conditioning 会 latch onto spurious history correlations，根因是训练时 history-space coverage 不足。

这已经说明：

> history 有信息，不等于 policy 学到的是我们希望它学的 task memory。

### π0.7

- https://www.pi.website/download/pi07.pdf

π0.7 直接面对大规模 heterogeneous robot data，并显式提供 strategy、episode quality、control modality、subgoal image 等 context 来消除 demonstration ambiguity。

因此 2026 的趋势实际上是同时发生：

```text
policy sees longer history
+
training data contains richer and more heterogeneous behavior modes
```

“history 到底在利用什么”因此不是一个旧式 RNN analysis，而是 foundation robot policy 的数据—表示问题。

---

## 5. 已经 collision 的东西

这个题要活下来，必须明确不声称以下内容是新发现。

### Multi-human heterogeneity 已经是老问题

`Eliciting Compatible Demonstrations for Multi-Human Imitation Learning`（CoRL 2022）已经说明不同 human demonstrators 会采取互相不兼容但都合理的策略。

https://proceedings.mlr.press/v205/gandhi23a.html

`Learning to Discern` 也直接研究 heterogeneous human demonstrations 的 quality / style。

https://arxiv.org/abs/2310.14196

所以：

> multiple demonstrators have different styles

不是贡献。

### Hidden behavior variable / history inference 有理论近邻

causal imitation / offline RL 已经研究 hidden confounder、multiple behavior policies、trajectory history 等问题。

因此：

> history can statistically reveal a latent behavior source

本身也不能成为论文结论。

### History spurious correlation 已经被 BPP 正面做

所以不能只得到：

> history model learns shortcuts.

BPP 已经把 long-context history coverage 与 spurious correlations 讲得很清楚。

---

## 6. 真正需要回答的层级

如果这个题值得做，至少应该回答一个比上述已有事实更大的问题：

### Question A — Representation

Robot policy 的 history representation 中，task/world state 与 behavior/strategy information 各占什么地位？

### Question B — Causality

改变 history 中的 behavior/strategy evidence，是否会在当前 observation 不变时改变 action mode？

### Question C — Generalization

这种 history dependence 在训练时有帮助，但 deployment 时是否会变成 source-specific shortcut，尤其当部署 trajectory 不再对应训练 producer/style 时？

### Question D — Modern VLA relevance

这个机制是否不仅存在于 robomimic RNN，而能在至少一个现代 history-aware VLA / memory policy 上复现？

如果只能回答 Question A 的一个小 probe，不够。

---

## 7. 最便宜的 prerequisite 仍然可以很窄

**题目要宽，第一枪可以窄。**

robomimic Multi-Human 是一个很好的 screening environment，因为它公开保留六个具体 operator masks，同 proficiency 内也有两个独立 operator。

可以先检查：

```text
single operator
vs
same-proficiency mixed operators
```

history model 的额外收益是否明显改变。

如果连这种最干净的 producer variation 都没有产生任何额外 history dependence，那么“behavior-source information 是 memory 的重要组成”这条解释大概率不值得继续。

如果 separation 存在，再考虑：

- explicit strategy/source context；
- history swap；
- matched current-state counterfactual；
- modern VLA reproduction。

这些都是**验证大问题的实验**，不是最终题目本身。

---

## 8. Kill standard

以下情况直接把本候选移出 shortlist：

1. 核心现象只存在于 robomimic MH，没有跨数据/model evidence；
2. 只能解释 mixed-human quality/style，而不能说明现代 robot-policy memory 的一般问题；
3. collision audit 找到已有工作已经直接分解 task memory vs behavior-source memory；
4. producer/strategy effect 在 rollout behavior 上没有 causal consequence；
5. 为了维持 novelty，标题必须不断缩成某个 operator / latency / dataset-specific effect。

第 5 条尤其重要：**如果只能靠越来越窄来躲 collision，就不做。**

---

## 9. 当前判断

**保留，继续审计，不注册。**

它现在最大的优点不是某个 robomimic effect，而是它能形成一个自然的顶会尺度问题：

> **What Does History Actually Model in Robot Policies?**

最大的风险也很清楚：multi-demonstrator heterogeneity、history shortcut、hidden-variable inference 三个邻域都有大量已有工作。

所以它只有在我们能把这些已有事实统一成一个**现代 history-aware robot policy 的机制结论，并证明 behavioral consequence**时才值得正式注册。
