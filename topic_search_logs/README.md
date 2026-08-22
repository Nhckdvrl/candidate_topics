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

## 单独摘出的未注册候选

这里不是正式 Topic 表，只是把搜索日志中已经值得进一步审计的问题单独拿出来，方便继续碰撞。候选一旦后续被撞死，直接保留失败记录，不移动到根目录包装。

- [What Does History Actually Model in Robot Policies?](./candidates/what_does_history_actually_model.md)  
  来源：Round 3 的 action-chunking temporal anomaly → Round 4 的 producer-mixture reinterpretation → Round 5 的 scope / collision audit。当前仍是 provisional candidate；NeurIPS 2025 已经明确给出 mixture-of-Markov-policies 可形成 non-Markovian mixture 的理论近邻，因此不能靠 producer-mixture 本身作为 novelty。

`Does Autonomous Robot Data Narrow Behavioral Support?` 当前**没有**单独摘出：题目尺度合格，但 batch-online RL / PLD 已经覆盖 diversity、failure-region probing、recovery coverage 等一级问题；目前还缺独立的 multi-generation support-contraction anomaly，不满足 phenomenon-first。
