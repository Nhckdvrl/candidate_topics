# 选题搜索日志

这个目录保存**尚未注册成正式 Topic 的选题搜索过程**，以及候选被升格、降级、撞死后的状态记录。

目的不是积累论文列表，而是把选题形成过程中真正有用的信息留下来：

- 从哪一个已知现象出发；
- 为什么一开始觉得它可能值得做；
- 查到了哪些直接或邻近工作；
- collision 到什么程度；
- 为什么继续、降级或砍掉；
- 如果机制成立，是否自然留下 method 空间；
- 第一个验证实验到底依赖什么前提、实际成本多大。

这里必须区分：

- **已知事实 / 文献直接结果**；
- **我们的机制猜测**；
- **待查 collision**；
- **待验证 prerequisite**。

正式注册到根目录编号 Topic 之前，候选最好先在这里经过一轮筛选。

从 Round 5 开始的硬标准：

> **实验可以窄，但研究问题本身不能靠不断缩小 scope 来躲 collision。**

Round 8 又补充：

> **顶会级问题不要求“零邻近工作”。真正要求的是：问题尺度够大、叙事角度相对新、第一枪实验相对新且干净，并且正反结果都有解释价值。**

Topic 23 归档后新增一条更强的识别规则：

> **如果一个 intervention 声称“拿掉了某个能力/路线/程序”，先证明 canonical behavior 确实因这个 intervention 丢失了那个 causal program。仅仅 nominally clamp 某组 actuator 不够。统计显著性无法补救 construct invalidity。**

Round 13（Topic 24 注册）把这条规则又推进了一步，而且这次是对着**我们自己的 instrument**：

> **一个 intervention 在未扰动条件下完美复现原系统，并不证明它切开的 seam 承载了任何 feedback。必须单独证明 seam liveness：同 command、同 controller 内部状态、同 clock，只换 observation，输出是否确定性地改变。**

原因是 P0 的“完美”本身就是它的上限：世界没有被扰动时，状态从未离开录制轨迹，所以任何 feedback 都不需要动作。一个纯前馈 controller 会给出一模一样的 10/10。这就是为什么 Topic 24 在 P0 之后又加了一道 P0b 才注册。

## 搜索日志

1. [Round 1：VLA / WAM 机制选题搜索](./2026-08-22_vla_wam_mechanism_search.md)
2. [Round 2：collision audit 与 data → learned computation](./2026-08-22_vla_wam_mechanism_search_round2.md)
3. [Round 3：二阶矛盾、action chunking timing 与 RL emergent behavior](./2026-08-22_vla_wam_mechanism_search_round3.md)
4. [Round 4：data producer、temporal support、memory 与 autonomous self-consuming data](./2026-08-22_vla_wam_mechanism_search_round4.md)
5. [Round 5：顶会问题尺度校准与继续 collision audit](./2026-08-22_vla_wam_mechanism_search_round5.md)
6. [Round 6：广泛大问题搜索与 action-generalization mechanism](./2026-08-22_vla_wam_mechanism_search_round6.md)
7. [Round 7：foundation scaling、WAM controllability 与 heterogeneous-data collision audit](./2026-08-22_vla_wam_mechanism_search_round7.md)
8. [Round 8：重新广搜大问题、hidden dynamics、physical feasibility、active perception 与 hierarchy mechanism](./2026-08-22_vla_wam_mechanism_search_round8.md)
9. [Round 9：whole-body motor equivalence、physical prompting、padding missingness 与新模型异常](./2026-08-22_vla_wam_mechanism_search_round9.md)
10. [Round 10：capability asymmetry、event transition、camera-role、MoE 与 human-video transfer audit](./2026-08-22_vla_wam_mechanism_search_round10.md)
11. [Round 11：feedback law / task-space correction 搜索启动与今晚收尾](./2026-08-23_vla_wam_mechanism_search_round11.md)
12. [Round 12：Topic 23 失败复盘与 feedback-source attribution 候选](./2026-08-24_vla_wam_mechanism_search_round12.md)
13. [Round 13：E 通过 P0 / P0b，注册为 Topic 24](./2026-08-24_vla_wam_mechanism_search_round13.md)
14. [2026-08-24 B/C/D promotion audit + Topic 23 outcome](./2026-08-24_candidate_promotion_audit.md)

## 当前 active provisional shortlist

### E. [Where Does Closed-Loop Robustness Live in Hierarchical Robot Foundation Policies?](./candidates/where_does_closed_loop_robustness_live.md)

**状态：已注册为 [Topic 24](../24_hierarchical_feedback_attribution/)（2026-08-24）。**

先 P0、后注册的流程走完了：P0 replay fidelity `10/10` 三条件全过、轨迹偏离 `0.000 rad`；
P0b WBC seam liveness `D = 2.4e-02..4.6e-02 rad`，repeatability floor 精确为 0。
physical-disturbance G0 已冻结，尚未运行。详见
[Round 13](./2026-08-24_vla_wam_mechanism_search_round13.md)。

### B. [How Do Robot Foundation Policies Generalize Actions?](./candidates/how_do_robot_foundation_policies_generalize_actions.md)

**状态：KEEP PROVISIONAL。**

核心问题：

> **What qualitative transition, if any, does robot foundation pretraining induce in action generation?**

希望区分：

```text
retrieval
vs interpolation
vs composition
vs extrapolation / synthesis
```

已有工作已经直接覆盖 retrieval / memorization、ACT-style interpolation，以及 GR00T 的更强 OOD robustness。真正还没干净解决的是 foundation-scale pretraining 是否把 action generation 推向 composition / synthesis，以及怎样在 behavior level 定义这个转变而不依赖任意 trajectory-similarity metric。

所以 B 暂时保留，但**不注册**。注册前必须先有一个简洁、可行为识别的 composition / synthesis contrast。

## 已移出 active shortlist

### C. [Why Does Task Decomposition Help Robot Foundation Policies?](./candidates/why_does_task_decomposition_help_robot_foundation_policies.md)

**状态：DOWNGRADED / REMOVE FROM ACTIVE SHORTLIST。**

原问题是：

```text
planner reasoning
vs
controller-support matching
vs
temporal handoff/reset
```

但 2026 年的 Hi-VLA / Cortex / hierarchical post-training 工作已经直接进入 planner-executor compatibility、low-level steerability、subgoal distribution alignment。继续做会越来越依赖窄化 scope 来躲 collision，不再占 active 位。

### D. [Do Robot Foundation Policies Learn Motor Equivalence Classes?](./candidates/do_robot_foundation_policies_learn_motor_equivalence_classes.md)

**状态：PROMOTED AS TOPIC 23 → ARCHIVED。**

正式项目：[`../23_motor_equivalence_classes/`](../23_motor_equivalence_classes/)

最终标签：

`ARCHIVED / ORIGINAL PANEL FALSE-POSITIVE + FROZEN PANEL PREREQUISITE FAILURE`

最重要的结果不是普通 null，而是原设计被实证证明会产生强假阳性：CloseDoor 下 `right_disabled=29/30`、`full_hold=0/30`、paired diff `0.967`、95% CI `[0.90,1.00]`、29 个 substitution events，但机器人其实主要靠 locomotion 把垂在身侧的手撞到门上，并没有 canonical right-arm motor program 被移除。

修正版 panel 加入 `right_frozen` / `both_arms_disabled` 后：

```text
CloseDoor:
  canonical 30/30
  canonical-right_frozen = 0.033
  both_arms_disabled = 30/30
  -> PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM

OpenFaucet:
  canonical 11/30 = 0.367
  -> PREREQUISITE_FAIL_CANONICAL
```

因此冻结的两个 task 都无法承载 motor-equivalence identification。Broad question 未被证伪，但不能为了救题再挑第三个 task。

详见：

- [`../23_motor_equivalence_classes/G0_RESULTS.md`](../23_motor_equivalence_classes/G0_RESULTS.md)
- [`../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md`](../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md)

### A. [What Does History Actually Model in Robot Policies?](./candidates/what_does_history_actually_model.md)

**状态：Round 8 后移出 active shortlist。**

IntentVLA、Big Picture Policies、RoboMME / μVLA / memory-VLA，以及 frozen-VLA visual-history auditing 已经把 history encoding / deployment / steering 逐步覆盖。剩余 novelty 越来越依赖 `world-state memory vs producer/strategy memory` 这一窄 decomposition，不再保留 active 位。

## 仍留在日志但未摘成候选的方向

### Does Autonomous Robot Data Narrow Behavioral Support?

题目尺度合格，但 batch-online RL / PLD 已经覆盖 diversity、failure-region probing、recovery coverage 等一级问题；目前缺独立 multi-generation support-contraction anomaly。

### Does Fine-Tuning Collapse the Behavioral Repertoire of Robot Foundation Policies?

有真实旁证，但与 2026 robot-policy memorization / generalization / task-vector / SAE 线过近。除非出现明显不同于 generic memorization / catastrophic forgetting 的大机制，否则不摘。

### What Makes Heterogeneous Robot Experience Compatible?

这是现实大问题，但 broad question 已被 2026 scaling/alignment literature 正面占据。继续做容易退化成 normalization / coordinate frame / mixture ratio。

### Can Robot Foundation Policies Adapt to Hidden Physical Dynamics from Their Own Consequences?

问题自然，但仍缺一个足够强、足够现代的 anomaly 作为 phenomenon-first 起点。

### Does Action-Space Missingness Become an Embodiment Shortcut?

Padding / validity-mask 设计差异是真实的，但还缺 behavioral evidence 证明模型真的利用 missingness 作为 embodiment shortcut。否则容易退化成 masking implementation paper。

### What Does a Physical Prompt Specify?

`goal / strategy / object / effector / timing` 的 factorized binding 仍值得审，但 generic “demo is goal or trajectory?” 已和 BPP / ICRT / ICIL 太近。

### Do Continuous Action Heads Miss Sparse Mode-Switch Events?

有 `reach/align works, grasp event never fires` 现象，但 hybrid action-space / gripper expert 已有直接工作。只有未来形成跨 architecture / event type 的更基础机制才值得升格。

### Do Robot Foundation Policies Learn Reusable Task-Space Feedback Laws?

这个方向后来注册为 Topic 19，并因 task-space identification failure 归档。不要从旧 joint-space proxy 继续救；若未来重做，必须从 contact/object/outcome space 重新注册。

### Do World Action Models Actually Use Their Predicted Futures?

Fast-WAM、Faster-WAM、RIFT 已经覆盖 training-time world modeling、inference-time future conditioning、iterative rollout necessity 等 controlled intervention。保留为搜索教训，不再占候选位。

## 当前结论

截至 2026-08-24，**E 已经注册为 [Topic 24](../24_hierarchical_feedback_attribution/)**，是这一轮 embodied search 第一个通过“先证明 instrument、后注册”流程的候选。B 仍处于 active provisional 状态；C 已降级；D 完成一次正式 Topic 生命周期并归档。

后续继续找题时，优先要求：

```text
真实 anomaly / tension
+
one-clean-contrast
+
canonical behavior 确实依赖被干预的 causal object
+
被切开的 seam 确实承载 feedback（seam liveness，不只是 replay fidelity）
+
正反结果都能直接解释
```

不要用更复杂的 gate 去拯救一个本身没有干净 experimental object 的题。