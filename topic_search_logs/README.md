# 选题搜索日志

这个目录保存**尚未注册成正式 Topic 的选题搜索过程**。

目的不是积累论文列表，而是把选题形成过程中真正有用的信息留下来：

- 从哪一个已知现象出发；
- 为什么一开始觉得它可能值得做；
- 查到了哪些直接或邻近工作；
- collision 到什么程度；
- 为什么继续、降级或砍掉；
- 如果机制成立，是否自然留下 method 空间；
- 第一个验证实验到底依赖什么前提、实际成本多大。

这里允许记录未成熟想法，但必须明确区分：

- **已知事实 / 文献直接结果**；
- **我们的机制猜测**；
- **待查 collision**；
- **待验证 prerequisite**。

正式注册到根目录编号 Topic 之前，候选最好先在这里经过一轮筛选。

另外，从 Round 5 开始增加一个硬标准：

> **实验可以窄，但研究问题本身不能靠不断缩小 scope 来躲 collision。**

如果 broad question 已经被做掉，只能靠限定某个 dataset / operator / delay / statistic 才显得新，则优先砍掉，而不是继续包装。

Round 8 又补充了一条校准：

> **顶会级问题不要求“零邻近工作”。真正要求的是：问题尺度够大、叙事角度相对新、第一枪实验相对新且干净，并且正反结果都有解释价值。**

## 当前日志

1. [Round 1：VLA / WAM 机制选题搜索](./2026-08-22_vla_wam_mechanism_search.md)
2. [Round 2：collision audit 与 data → learned computation](./2026-08-22_vla_wam_mechanism_search_round2.md)
3. [Round 3：二阶矛盾、action chunking timing 与 RL emergent behavior](./2026-08-22_vla_wam_mechanism_search_round3.md)
4. [Round 4：data producer、temporal support、memory 与 autonomous self-consuming data](./2026-08-22_vla_wam_mechanism_search_round4.md)
5. [Round 5：顶会问题尺度校准与继续 collision audit](./2026-08-22_vla_wam_mechanism_search_round5.md)
6. [Round 6：广泛大问题搜索与 action-generalization mechanism](./2026-08-22_vla_wam_mechanism_search_round6.md)
7. [Round 7：foundation scaling、WAM controllability 与 heterogeneous-data collision audit](./2026-08-22_vla_wam_mechanism_search_round7.md)
8. [Round 8：重新广搜大问题、hidden dynamics、physical feasibility、active perception 与 hierarchy mechanism](./2026-08-22_vla_wam_mechanism_search_round8.md)

## 当前 active provisional shortlist

这里不是正式 Topic 表，只是把搜索日志中已经值得进一步审计的问题单独拿出来，方便继续碰撞。候选一旦后续被撞死，直接保留失败记录，不移动到根目录包装。

### B. [How Do Robot Foundation Policies Generalize Actions?](./candidates/how_do_robot_foundation_policies_generalize_actions.md)

来源：Round 6，并在 Round 7 被 foundation scaling / compositional-generalization 新证据进一步加强。

核心不是再做 OOD benchmark，而是区分 foundation policy 的 action-side generalization mechanism：

```text
retrieval
vs interpolation
vs composition
vs extrapolation / synthesis
```

当前更准确的母问题是：

> **What qualitative transition, if any, does robot foundation pretraining induce in action generation?**

邻近工作已经覆盖 retrieval、motion primitives、action composition、mechanistic features 等局部结构，但仍没有系统回答：**随着 foundation pretraining 的 scale / diversity 增长，successful OOD behavior 的默认生成机制是否发生系统性转变。**

因此 B 保留。后续实验应尽量加入 pretraining scale / diversity 轴，而不只是 model-family 对比；如果最终只能缩成某 benchmark 的 trajectory-similarity statistic，则降级。

### C. [Why Does Task Decomposition Help Robot Foundation Policies?](./candidates/why_does_task_decomposition_help_robot_foundation_policies.md)

来源：Round 8。

核心问题：

> **When a hierarchical VLA greatly outperforms a flat VLA, how much of the gain comes from genuine high-level planning/reasoning, and how much comes from repeatedly translating the global task back into atomic instructions that the low-level policy already knows how to execute?**

更尖锐地说：

> **Does the planner reason better, or does it keep the controller on-support?**

RoboHiMan 中同一个 π0.5 加 rule-based atomic sub-instruction switching 就能得到很大长任务提升；Hi-VLA systematic study 又显示 low-level language steerability 对 hierarchy 成败关键；compositional-generalization diagnosis 进一步说明不少失败是已有低层能力无法被 instruction 正确 steer 出来，而不是 skill 本身不存在。

因此第一枪可以非常干净：固定 low-level VLA 和 task states，比较 flat global instruction、oracle atomic decomposition、learned planner、wording-support control 与 boundary-only reset，从而拆分：

```text
planning / sequencing
vs
controller-support matching
vs
temporal handoff / reset
```

这条当前作为 provisional C 保留，不注册为正式 Topic。

## 已移出 active shortlist

### A. [What Does History Actually Model in Robot Policies?](./candidates/what_does_history_actually_model.md)

来源：Round 3–5。

**Round 8 后移出 active shortlist。** 原问题围绕 history 到底建模 world/task state，还是 producer / strategy / latent behavior structure。

后续 collision 已经明显升高：IntentVLA 直接做 history → short-horizon intent；BPP 研究 history 中 spurious correlation；RoboMME / μVLA / memory-VLA 系列持续扩张；`Present but Not Remembered: Auditing How Frozen VLAs Encode, Deploy, and Steer Visual History` 又已经对 frozen VLA 做 layer-wise probing + causal interchange，直接审计 history 是否被编码、是否被 action readout 使用、是否可 steering。

A 剩余 novelty 越来越依赖 `world-state memory vs producer/strategy memory` 这一窄 decomposition。按“不能靠不断缩 scope 躲 collision”的规则，当前不再占 active candidate 位，但保留文件和失败/降级记录。

## 仍留在搜索日志、但未摘出的方向

### Does Autonomous Robot Data Narrow Behavioral Support?

题目尺度合格，但 batch-online RL / PLD 已经覆盖 diversity、failure-region probing、recovery coverage 等一级问题；目前还缺独立的 multi-generation support-contraction anomaly，不满足 phenomenon-first。

### Does Fine-Tuning Collapse the Behavioral Repertoire of Robot Foundation Policies?

有真实旁证，尤其 Qwen-RobotManip 的 `VLA-to-VA degradation`，但与 2026 robot-policy memorization / generalization / task-vector / SAE 研究线过近。当前不单独摘出，除非后续出现一个明显不同于 generic memorization / VLM catastrophic forgetting 的大机制。

### What Makes Heterogeneous Robot Experience Compatible?

Qwen-RobotManip、Rethinking VLA Scaling、π0.7、VLAFlow、JoyAI-RA 都证明这是一个真实大问题：naive pooling 可以 negative transfer，而 alignment / richer context / future-latent constraints 能让 heterogeneous experience 共存。但 broad question 本身已经被 2026 scaling/alignment literature 正面占据，再做很容易被迫缩成 normalization / coordinate frame / mixture ratio，因此不摘成候选。

### Can Robot Foundation Policies Adapt to Hidden Physical Dynamics from Their Own Consequences?

Round 8 仍认为问题自然：视觉几乎不变但 mass / friction / damping / compliance / controller response 改变后，policy 是否能通过自己动作造成的后果进行 online system identification 并更新后续控制。但目前还缺一个足够强、足够现代的 anomaly 作为 phenomenon-first 起点，因此继续作为搜索线，不摘候选。

### Do World Action Models Actually Use Their Predicted Futures?

题目非常自然，但已被 Fast-WAM、Faster-WAM、RIFT 连续占据：training-time world modeling、inference-time future conditioning、iterative rollout necessity 都已有直接 controlled intervention / ablation。保留为搜索教训，不再占候选位。
