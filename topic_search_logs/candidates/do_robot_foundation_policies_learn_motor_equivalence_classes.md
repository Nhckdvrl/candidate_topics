# Do Robot Foundation Policies Learn Motor Equivalence Classes?

> Status: **PROVISIONAL SEARCH CANDIDATE — not a registered Topic.**
>
> 来源：Round 9。这个文件只摘出经过本轮广泛搜索后仍值得进一步做 prerequisite experiment 的问题；不进入根目录正式 Topic，也不默认命题成立。

---

## 1. 问题

> # **Do Robot Foundation Policies Learn Motor Equivalence Classes?**
>
> **Do they learn the task constraint, or the particular motor realization chosen by the demonstrator?**

更具体地说：

> **如果训练 demonstration 总是用一种身体实现完成任务，但同一个任务实际上允许很多不同的身体解；测试时只拿掉 demonstration 中那条 canonical motor solution，foundation policy 会不会找到另一个 goal-equivalent solution？**

例如：

```text
目标：把抽屉关上
训练示范：一直用右手
测试约束：右手正在拿东西 / 右臂某些 DoF 不可用
替代解：左手、髋部、身体其他接触点仍可完成
```

我们真正关心的不是：

> 加点 noise 后 success 会不会下降？

而是：

> **policy 是否理解“关上抽屉”这个 task effect，还是主要学会了“用这条右手运动去关抽屉”？**

---

## 2. 为什么这个问题现在自然

2026 的 whole-body / omni-bodied robot foundation models 已经开始展示过去 specialist policy 很少出现的行为。

### Figure Helix 02

官方：

https://www.figure.ai/news/helix-02

Figure 在 full-body kitchen task 里明确展示：

- hands occupied 时用 **hip** 关 drawer；
- 用 **foot** 抬 dishwasher door；
- 官方称之为 `using the entire body as a tool`。

这不是普通 walking + manipulation coordination。

它意味着：

```text
task effect preserved
motor realization changed
```

### Skild omni-bodied brain

官方：

https://www.skild.ai/blogs/omni-bodied

Skild 又从 locomotion / morphology 侧展示：

- limb length / DoF 被改变后快速换 gait；
- knee joint 被锁后重新分配重心；
- wheel jam 后从 rolling 切换 walking；
- stilts 改变身体几何后重新调整 timing / placement。

这些现象不能直接证明 foundation policy 学到了 motor-equivalence representation，但它们给了一个非常强的 phenomenon-first motivation：

> **大规模、多身体 physical pretraining 是否改变了 policy 抽象运动的层级？**

---

## 3. 正确的理论语言：Motor Equivalence / Goal-Equivalent Manifold

这个概念本身一点也不新。

神经科学 / motor control 很早就研究：

> 同一个 task-level goal 可以由大量不同 motor solutions 实现。

用 Goal Equivalent Manifold（GEM）表示：

\[
\mathcal{G}=\{x\mid f(x)=0\}
\]

其中：

- `x`：body / motor variables；
- `f(x)`：task-level error；
- `\mathcal{G}`：所有 task-equivalent motor solutions。

参考：

- https://pmc.ncbi.nlm.nih.gov/articles/PMC3858478/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11230222/

因此我们不能 claim：

> “机器人有冗余”“不同动作能完成同一目标”是新发现。

真正的新问题是：

> **现代 robot foundation policy 在大规模 imitation / physical pretraining 后，是否自然学到了这种 task-level equivalence，还是仍然主要绑定 demonstrator 轨迹？**

这是一个 foundation-policy mechanism audit。

---

## 4. 为什么这题和普通 robustness 不一样

普通 robustness：

```text
scene / object / camera / dynamics 改了
-> policy 是否还能成功
```

本题：

```text
scene 不变
task effect 不变
object 不变
robot embodiment 不变

只把 demonstrator 常用的 motor solution 从 feasible set 中移除
-> policy 是否转到另一个 goal-equivalent solution
```

所以这里直接识别的是：

```text
motion-template dependence
vs
task-effect abstraction
```

这比 generic perturbation robustness 更具体。

---

## 5. 为什么不是 cross-embodiment transfer

cross-embodiment 常问：

> 在 robot A 学到的东西能不能迁移到 robot B？

这里固定：

> **same robot, same task, same world.**

只有 available motor realization 改变。

因此我们不用解决：

- morphology correspondence；
- action-space alignment；
- camera/domain gap；
- embodiment normalization；
- controller mismatch。

这让机制 identification 反而更干净。

---

## 6. 为什么不是 fault-tolerant control

已有很多 fault-tolerant controller 会：

- 显式随机 actuator failure；
- 学 fault estimator；
- 训练 adaptation module；
- 设计 fault-aware reward / policy。

那些工作问：

> **怎么训练一个故障以后还能工作的 controller？**

本题问：

> **一个已经训练好的 generic foundation policy，在没有专门针对这个 intervention 训练的情况下，是否已经形成 spontaneous motor-equivalence behavior？**

所以第一阶段不设计 failure method。

我们的 intervention 甚至不一定叫“故障”：

- 一只手被物体占用；
- 某只手当前不可达；
- canonical contact point 被阻挡；
- 某个 joint range 被限制。

只要 alternative solution 物理存在即可。

---

## 7. 为什么不是 B 的换皮

B：

> **How Do Robot Foundation Policies Generalize Actions?**

问 action generation 的 retrieval / interpolation / composition / synthesis。

D 固定 action task 和 embodiment，专门问：

> **policy 是否把不同 body-space realizations 视为同一个 task solution class。**

形式上：

```text
B: new behavior comes from where?
D: which motor differences does the policy treat as irrelevant to the task effect?
```

两者可以互相支持，但研究问题不同。

---

## 8. 最漂亮的开源实验平台：Ψ₀ + SIMPLE

### Ψ₀

**Ψ₀: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation**  
RSS 2026

https://github.com/physical-superintelligence-lab/Psi0

公开：

- model；
- data；
- training / fine-tuning；
- SIMPLE evaluation；
- whole-body action policy / low-level controller integration。

### SIMPLE

https://github.com/physical-superintelligence-lab/SIMPLE

有 50+ whole-body tasks，包括：

- OpenFaucet；
- CloseDoor；
- PushOfficeChair；
- OpenOven；
- OpenTrashCan；
- BendPick / Handover 等。

最关键的是：**现有 task definition 已经天然把 task effect 和 motor realization 分开。**

---

## 9. OpenFaucet 给出的天然 identification

源码：

https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/src/simple/tasks/g1_wholebody_open_faucet_teleop.py

task success 只检查 faucet joint：

```python
faucet_joint0_qpos > 0.7 or faucet_joint0_qpos < -0.7
```

也就是 task 真正要求：

> **把 faucet 打开。**

但 automated demonstration decomposition 却硬编码：

```python
hand_uid="dex3_right"
lock_links=["left_hand_palm_link"]
```

因此 dataset 只展示：

```text
一个 task effect
+
一个任意选中的 right-hand realization
```

这是本题几乎理想的 setting。

---

## 10. CloseDoor 同样如此

源码：

https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/src/simple/tasks/g1_wholebody_close_door_teleop.py

success 只根据 door joint 判断。

但 decomposition 同样固定 `dex3_right` 并锁左手。

因此我们完全可以问：

> 如果 right-hand solution 不再可用，但 left-hand / body solution 仍可达到同一个 door-state success，policy 怎么办？

---

## 11. 第一枪：只改变可用 motor solution

第一轮建议只选 2–3 个 effect 定义非常清楚的 tasks。

优先：

1. OpenFaucet；
2. CloseDoor；
3. PushOfficeChair / OpenOven 中挑一个经过 feasibility audit 最干净的。

### Condition A — Canonical

原始 environment + policy。

先确认 baseline success 足够高。

### Condition B — Same-effector redundancy

限制 canonical arm 的部分 joint / workspace，但**同一只手仍可通过另一套关节配置完成**。

测 local kinematic substitution。

### Condition C — Cross-effector substitution

让 canonical right hand 不可用，但保持 left hand 可行。

测真正的 cross-effector motor equivalence。

### Condition D — Functional occupation

右手不是“神秘失灵”，而是拿着一个物体 / 被占用。

这最接近 Figure 的自然 whole-body phenomenon。

### Condition E — Impossible negative control

让所有 alternative solutions 都不可行。

确认模型不会因为偶然乱动被误判成 substitution。

### Condition F — Alternative-solution oracle

scripted / teleop / motion planner 证明 intervention 后任务仍然可解。

这是 prerequisite。

如果 alternative oracle 都做不到，那个 intervention 不能进入实验。

---

## 12. 不要只测 success

至少测：

### 12.1 Task-effect success

例如：

- faucet joint；
- door angle；
- object displacement。

这是最重要的 task-level metric。

### 12.2 Canonical-effector retry count

如果 policy 不断尝试已经不可用的右手路径：

> strong trajectory binding。

### 12.3 Body-part activation shift

从 action expert output 直接看：

```text
right-arm dominant
-> left-arm / torso / whole-body redistribution?
```

### 12.4 Adaptation latency

干预后多少 closed-loop steps 才出现替代策略。

### 12.5 Motor deviation vs effect preservation

我们真正想看到：

```text
motor-space deviation ↑
while
task-effect error remains low
```

这比单纯 trajectory similarity 更接近 motor-equivalence 定义。

---

## 13. prerequisite / kill line

第一枪之前先冻结以下 prerequisite。

### P1. baseline

canonical task 必须有足够 success。

否则无法审计能力退化。

### P2. alternative feasibility

每个 intervention 都必须有 oracle alternative solution。

### P3. intervention specificity

限制 canonical motor solution 时，不能同时破坏：

- camera visibility；
- task language；
- object dynamics；
- success evaluator。

### P4. policy observability

policy 至少要能从 proprio / vision 看到 motor constraint 的后果。

否则“它不知道自己手被锁”是 identification confound。

### P5. phenomenon effect size

如果所有公开 foundation policies 在最简单的 left/right substitution 上都是 `~0%`，题目**不一定死**：这本身可能就是强负结果。

真正的 kill 是：

> 干预后 success 的变化完全可以由 reachability / controller failure / observation corruption 解释，无法识别 task abstraction。

如果必须堆大量复杂控制才能排除这些解释，就不要继续。

---

## 14. 预期结果类型

### Result A — spontaneous motor equivalence

canonical solution 被拿掉后，policy 自动迁移到 alternative body solution。

结论：

> **foundation physical pretraining can support task-level motor abstractions that are not tied to the demonstrator's exact body trajectory.**

### Result B — trajectory fixation

policy 重复尝试 canonical path，即使 alternative solution 明确可行。

结论：

> **current foundation policies may still model demonstrations more strongly than task-equivalent physical effects.**

这会直接质疑一些“general physical intelligence”式 whole-body claim 的含义。

### Result C — morphology diversity changes the regime

普通 VLA 不 substitution，但 omni-body / cross-morphology pretrained model 会。

结论：

> **morphology diversity may change the abstraction level of learned control, rather than merely increasing robustness.**

这个结果尤其像 ICLR / ICML / RSS 叙事。

---

## 15. 后续 method 空间

如果问题成立，方法不用现在设计死，但空间非常自然：

- effect-space supervision；
- effector / DoF dropout；
- motor-equivalence augmentation；
- task-null / goal-equivalent perturbation training；
- object-effect prediction；
- contrastive learning across different body solutions of the same task；
- intentional collection of diverse motor realizations；
- planner/controller jointly搜索 goal-equivalent solution set，而非单 trajectory。

---

## 16. 顶会叙事为什么够

论文标题不需要出现 Ψ₀、SIMPLE 或某个 benchmark：

> **Do Robot Foundation Policies Learn Motor Equivalence Classes?**

它连接：

- foundation robot policy；
- imitation learning；
- whole-body / humanoid control；
- motor control / redundancy；
- representation abstraction；
- robustness / adaptation。

AI venue 可以讲：

> 大规模 physical pretraining 改变了什么 abstraction？

robotics venue 可以讲：

> learned policy 是否利用身体冗余达到 task-level resilience？

并且最关键的是：**正反结果都值得写。**

如果成立，我们第一次 controlled 地看到 effect-level motor abstraction；如果不成立，则说明现有 foundation policy 的“whole-body generality”仍然高度依赖 demonstration coverage。

---

## 17. 当前评级

**Novelty：中高。** 经典 motor equivalence 不是新概念，但在现代 foundation-VLA 上做 causal mechanism audit 仍未见直接同题工作。

**Significance：高。** 直接触及“foundation physical intelligence 到底抽象了什么”。

**Experiment cleanliness：中高。** Ψ₀ + SIMPLE 已提供 task-effect evaluator 和 whole-body policy；最大工作是构造并验证 motor-solution interventions。

**Engineering risk：中。** 要确保 alternative solution 真的可行，并区分 high-level VLA substitution 与 low-level controller compensation。

**Collision risk：中。** whole-body / fault tolerance / cross-embodiment 邻域很热，但核心 causal question 仍相对独立。

**当前决定：保留为 provisional candidate D，进入后续 prerequisite / code-design audit；不注册正式 Topic。**
