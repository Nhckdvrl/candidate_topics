# 2026-08-24: Embodied Topic Search Round 12 — 从失败题反推“可识别性优先”的新候选

> 状态：**找到一个新的最高优先级 provisional candidate E，但暂不注册根目录 Topic。先过无扰动 replay-fidelity P0。**

## 0. 为什么前面的题一直死

这轮不是继续列论文，而是把具身方向所有已跑失败压成一个共同错误模式。

### Topic 08 — uncertainty geometry

问题不是代码没跑通，而是最初 measurement 本身容易循环定义；重建后虽然看到 action/outcome decoupling，但真实 deployed entropy monitor 只是 weak-but-informative，没有达到值得做机制论文的 operational bar。

**教训：不要让 metric 自己把想要的解释编码进去。**

### Topic 09 — VLA self-knowledge

识别逻辑本身成立，但依赖同一 state 上不同 checkpoint 产生大量双向 winner reversal。大量 rollout 后才发现 joint instance-level support 几乎不存在。

**教训：不要把题建立在一个未经证实的自然 crossover population 上。**

### Topic 15 — predictive policy state

future supervision 确实让 representation 更 predictive，但这个 state 没有改善 action path；恢复 action capacity 后也没有救回来。

**教训：encoded / decodable information 与 causal use 是两件事。优先直接切 causal path。**

### Topic 19 — task-structured feedback

joint-axis response 无法识别 redundant system 的 task-space correction；kinematically large EE perturbation 也不自动等于 task-relevant perturbation。

**教训：主 endpoint 要尽量留在 task/outcome space，少发明 proxy geometry。**

### Topic 23 — motor equivalence

这是最关键的校准。原四条件 CloseDoor 能给出：

```text
right_disabled = 29/30
full_hold = 0/30
paired diff = 0.967
95% CI = [0.90, 1.00]
29 apparent substitution events
```

但解释完全错：canonical 行为本来就没有需要被替代的右臂 motor program。修正版 `right_frozen` / `both_arms_disabled` 一加，马上发现 arm articulation 根本不重要。

**教训：一个 intervention 叫“remove X”不代表 X 在 canonical computation 里真的被移除了。先验证 treatment semantics，再看 outcome。**

## 1. 共同根因

前几题有一个反复出现的结构：

```text
先提出一个二阶抽象机制区别
    uncertainty geometry / self-knowledge / predictive mediation /
    task-space feedback / motor equivalence
然后再去公开模型里寻找刚好能承载这个区别的 experimental object
```

这顺序太危险。

正确顺序应该倒过来：

```text
先找到公开系统里已经存在、可直接切开的 causal seam / anomaly
再问一个只跨一步的因果问题
```

这轮因此新增硬规则：

> **候选优先来自“已经实证看到的系统张力”，而不是先想一个优雅抽象概念。**

## 2. 继续扫过但不再押的方向

### B: How Do Robot Foundation Policies Generalize Actions?

仍是大问题，但 retrieval/interpolation 已被 ICLR 2026 直接审计；composition/synthesis 仍缺不依赖 arbitrary trajectory metric 的行为定义。继续保留 B，不升。

### feedback-law / task-null geometry

Round 11 方向自然，但若继续依赖 null-space / task-relevant manifold，很容易重走 Topic 19。近期 recovery / geometry-aware imitation / perturbation control 邻域也越来越密。暂不押。

### cross-embodiment transfer vs routing

问题现实，但 heterogeneous co-training、embodiment alignment、validity mask、cross-embodiment scaling 已非常拥挤；很容易退化成 metadata/padding implementation audit。

### hidden dynamics / online system ID

问题好，但 2026 已有 in-context world modeling / test-time adaptation / system identification 线，缺我们自己的强 anomaly。

### sparse mode-switch event

`reach works, grasp never fires` 是真实现象，但 hybrid action / gripper expert / event-aware VLA 已经成赛道。

## 3. 新现象来自 Topic 23 自己

Topic 23 在真实 Psi0 + SIMPLE stack 中发现：

> **在 VLA 输出处修改命令，不等于在物理机器人上删除那条行为，因为下游 WBC 会读取当前 proprio 并重新求解。**

这不是 nuisance；它说明现代 humanoid VLA 的“robust closed-loop behavior”本身是一个多层 feedback 系统产物。

于是出现新的自然问题：

# Where Does Closed-Loop Robustness Actually Live in Hierarchical Robot Foundation Policies?

> **机器人被扰动以后还能完成任务，到底是 VLA 重新规划了，还是低层 WBC/RL controller 自己救回来的？**

## 4. 为什么这次 identification 更短

Psi0/SIMPLE upstream 已明确暴露两个真实软件 seam：

```text
VLA -> WBC: vla_cmd
WBC -> actuator: target_q / hand targets
```

因此无需 probe、SAE、trajectory manifold、task-null direction、自然 crossover。

只需要录制 nominal command tape，然后在同一个 physical perturbation 下依次 replay：

```text
fresh feedback
VLA-command replay + live WBC
post-WBC actuator-reference replay
```

直接用 official task success 做 endpoint。

## 5. 为什么暂时不注册 Topic 24

这正是吸取 Topic 23：

> **先验证 replay intervention 的语义，再注册科学 claim。**

所以先做无扰动 P0：CloseDoor 10 configs，要求 live / VLA replay / actuator replay 都至少 0.90 success，且 replay 相对 live drop 不超过 0.10。

P0 只验证 instrument，不施加外部扰动，也不看 attribution 结果。

如果过，再正式注册 Topic 24 并冻结 G0 force panel。

## 6. 当前判断

这是目前 embodied search 中我最愿意押的一题，因为它同时满足：

```text
真实 observation 已存在          ✓  Topic23 亲自暴露 WBC absorption
causal object 先验存在            ✓  两个 software seams 明确存在
第一枪短                          ✓  record/replay，不训练
endpoint 直接                     ✓  official task success
无需 rare natural support         ✓
无需 latent metric                ✓
正反结果都可解释                  ✓
method opening 明显               ✓
已有强现代系统作为 motivation      ✓
直接 collision 暂未找到            ✓
```

候选详见：

`candidates/where_does_closed_loop_robustness_live.md`

P0 原型：

`prototypes/feedback_source_attribution/`
