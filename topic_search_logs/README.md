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

## 当前日志

1. [Round 1：VLA / WAM 机制选题搜索](./2026-08-22_vla_wam_mechanism_search.md)
2. [Round 2：collision audit 与 data → learned computation](./2026-08-22_vla_wam_mechanism_search_round2.md)
3. [Round 3：二阶矛盾、action chunking timing 与 RL emergent behavior](./2026-08-22_vla_wam_mechanism_search_round3.md)
4. [Round 4：data producer、temporal support、memory 与 autonomous self-consuming data](./2026-08-22_vla_wam_mechanism_search_round4.md)
5. [Round 5：顶会问题尺度校准与继续 collision audit](./2026-08-22_vla_wam_mechanism_search_round5.md)
6. [Round 6：广泛大问题搜索与 action-generalization mechanism](./2026-08-22_vla_wam_mechanism_search_round6.md)
7. [Round 7：foundation scaling、WAM controllability 与 heterogeneous-data collision audit](./2026-08-22_vla_wam_mechanism_search_round7.md)

## 单独摘出的未注册候选

这里不是正式 Topic 表，只是把搜索日志中已经值得进一步审计的问题单独拿出来，方便继续碰撞。候选一旦后续被撞死，直接保留失败记录，不移动到根目录包装。

### A. [What Does History Actually Model in Robot Policies?](./candidates/what_does_history_actually_model.md)

来源：Round 3 的 action-chunking temporal anomaly → Round 4 的 producer-mixture reinterpretation → Round 5 的 scope / collision audit。

**当前状态：保留，但 collision risk 已升高。** Round 7 找到 `IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation`，已经直接把“相似当前观测对应不同 short-horizon intent，history 用来维持 episode 内行为模式”做成现代 VLA 问题。因此 A 不能再把 `history -> latent intent / behavior mode inference` 本身作为 novelty。

A 若继续，必须回答更强的问题：现代 history-aware robot policy 的 memory 到底在利用 task/world state，还是 training-data behavior source / strategy structure；并且必须证明这种 dependence 有 causal rollout consequence。若再找到直接做这层 decomposition 的工作，应立即移出 shortlist。

### B. [How Do Robot Foundation Policies Generalize Actions?](./candidates/how_do_robot_foundation_policies_generalize_actions.md)

来源：Round 6。主要从 ICLR 2026 `Demystifying Robot Diffusion Policies` 的 action-memorization / interpolation audit，与 2026 foundation-VLA 对 zero-shot、recovery、composition、cross-embodiment generalization 的强 claim 之间的张力长出来。

核心不是再做 OOD benchmark，而是区分 foundation policy 的 action-side generalization mechanism：

```text
retrieval
vs interpolation
vs composition
vs extrapolation / synthesis
```

**Round 7 后 B 明显加强，当前优先级高于 A。** Dyna-2 已经把 human-video pretraining 推到 1M 小时并报告 cross-embodiment scaling law；π0.7 报告 compositional generalization；Qwen-RobotManip 报告大规模 aligned pretraining 后的 OOD / recovery / cross-embodiment 能力。但这些工作仍没有系统回答：success 随 scale 上升时，motor generation 到底发生了什么 qualitative change。

Round 7 后 B 更准确的机制表述是：

> **What qualitative transition, if any, does robot foundation pretraining induce in action generation?**

但标题仍保留更自然的 `How Do Robot Foundation Policies Generalize Actions?`。

第一枪后续应尽量加入 pretraining scale / diversity 轴，而不只是 model-family 对比。如果最后仍只能缩成某个 benchmark 的 trajectory-similarity analysis，则直接砍。

## 仍留在搜索日志、但未摘出的方向

### Does Autonomous Robot Data Narrow Behavioral Support?

题目尺度合格，但 batch-online RL / PLD 已经覆盖 diversity、failure-region probing、recovery coverage 等一级问题；目前还缺独立的 multi-generation support-contraction anomaly，不满足 phenomenon-first。

### Does Fine-Tuning Collapse the Behavioral Repertoire of Robot Foundation Policies?

有真实旁证，尤其 Qwen-RobotManip 的 `VLA-to-VA degradation`，但与 2026 robot-policy memorization / generalization / task-vector / SAE 研究线过近。当前不单独摘出，除非后续出现一个明显不同于 generic memorization / VLM catastrophic forgetting 的大机制。

### What Makes Heterogeneous Robot Experience Compatible?

Round 7 重新认真审过。Qwen-RobotManip、Rethinking VLA Scaling、π0.7、VLAFlow、JoyAI-RA 都证明这是一个真实大问题：naive pooling 可以 negative transfer，而 alignment / richer context / future-latent constraints 能让 heterogeneous experience 共存。但 broad question 本身已经被 2026 scaling/alignment literature 正面占据，再做很容易被迫缩成 normalization / coordinate frame / mixture ratio，因此不摘成候选。

### Do World Action Models Actually Use Their Predicted Futures?

题目非常自然，但已被 Fast-WAM、Faster-WAM、RIFT 连续占据：training-time world modeling、inference-time future conditioning、iterative rollout necessity 都已有直接 controlled intervention / ablation。保留为搜索教训，不再占候选位。
