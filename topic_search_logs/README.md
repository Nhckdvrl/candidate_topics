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

## 当前日志

1. [Round 1：VLA / WAM 机制选题搜索](./2026-08-22_vla_wam_mechanism_search.md)
2. [Round 2：collision audit 与 data → learned computation](./2026-08-22_vla_wam_mechanism_search_round2.md)
3. [Round 3：二阶矛盾、action chunking timing 与 RL emergent behavior](./2026-08-22_vla_wam_mechanism_search_round3.md)
4. [Round 4：data producer、temporal support、memory 与 autonomous self-consuming data](./2026-08-22_vla_wam_mechanism_search_round4.md)

## Round 4 后当前最强未注册候选

> **When a robot policy uses history, is it remembering the world — or identifying the data producer?**

当前不直接注册。先用 robomimic Multi-Human 的逐 operator masks 做 matched-data、within-proficiency prerequisite：检查 producer mixture 是否产生额外 history gain，以及 true producer ID 是否能解释这部分 gain。若核心 separation 不存在，直接砍，不增加复杂 gate 救题。
