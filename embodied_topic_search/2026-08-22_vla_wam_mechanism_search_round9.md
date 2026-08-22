# 2026-08-22：VLA / WAM 机制选题搜索日志（第九轮）

> 状态：**继续搜索。本轮主体继续找新的大问题，不反复重审 B/C。新增 provisional candidate D；其余方向保留真实 collision / kill 记录，不为数量硬凑候选。**
>
> 目标 venue 仍按 ICLR / ICML / NeurIPS / RSS / ICRA / IROS 的叙事尺度校准：题目本身必须是自然、一般、值得回答的 computation / representation / learning question；允许有邻近工作，但核心叙事与第一枪实验必须有相对新意。

---

# 0. 本轮结论先行

Round 9 扫了几块与 B/C 基本正交的新区域：

- 多机器人协作里是否存在 partner / other-agent model；
- tactile 到底改变 high-level intent 还是只提供 local reflex；
- whole-body foundation policy 是否会做 **motor equivalence / motor substitution**；
- body schema 与在线身体适应；
- action uncertainty 是否混淆 epistemic uncertainty 与合法 multimodality；
- human egocentric video 中 camera/head motion 是否构成 hidden supervision；
- heterogeneous action padding / missingness 是否变成 embodiment shortcut；
- one-shot physical prompting / in-context imitation 到底从一条 demonstration 里绑定了什么。

其中大多数要么已有非常接近的研究线，要么仍然太像 data / engineering artifact。

本轮真正越搜越强、且第一枪已经能落在公开 foundation policy + simulation 上的问题是：

> ## D — **Do Robot Foundation Policies Learn Motor Equivalence Classes?**
>
> **Do they learn the task constraint, or the particular motor realization chosen by the demonstrator?**

更直白地说：

> **如果示范里一直用右手完成任务，但任务本身用左手、身体、脚等也完全可解；测试时只把“右手这条解”拿掉，foundation policy 会换一个 goal-equivalent motor solution，还是继续执着地 replay demonstrator 的身体轨迹？**

这个问题有两个独立的最新 phenomenon-first 入口：Figure Helix 02 的 whole-body functional substitution，以及 Skild omni-bodied brain 在 limb / wheel / joint 改变后的快速重新分配。

同时，SIMPLE + Ψ₀ 给出了一个非常漂亮的开源 identification：**任务成功条件本身只定义 environment effect，但 demonstration generator 却硬编码了特定右手 solution。** 这意味着不需要先发明 benchmark，现有环境已经天然把 `task effect` 与 `demonstrated body realization` 分开。

因此本轮将 D 单独摘出为 provisional candidate。

---

# 1. 多机器人协作：很大，但 broad question 已经被占

Figure Helix 02 的双机器人 bedroom / living-room demo 一度引出：

> **Does a robot foundation policy form a model of another agent's latent intent?**

因为多个机器人可以在没有显式 central planner / message passing 的情况下，通过视觉观察对方动作完成协作。

这个题很自然，但继续查到：

- **Latent Theory of Mind**（CoRL 2025 Oral）已经直接从对方动作中推断 latent state / intent；
- **CHORUS**（2026）已经把 decentralized, communication-free multi-robot cooperation 接到 foundation-VLA setting。

所以：

```text
other-agent modeling / partner intent inference
```

作为 broad question 已经太近。

**结论：不摘。**

---

# 2. whole-body demo 里真正奇怪的现象：机器人会换身体部位完成同一物理效果

## 2.1 Figure Helix 02：hands occupied → hip / foot become tools

Figure 官方 Helix 02 页面：

https://www.figure.ai/news/helix-02

它明确报告 whole-body kitchen task 中：

- 双手被占用时，用 **hip** 关 drawer；
- 用 **foot** 抬 dishwasher door；
- 官方将其描述为 `using the entire body as a tool rather than relying solely on the hands`。

重要的不是“humanoid 很灵活”。

真正异常的是：

```text
任务效果：关上抽屉
常规实现：手推
当前约束：手被占用
实际实现：髋部推
```

也就是说 task effect 保持不变，但 motor realization 换了。

这在经典 motor-control 里是一个非常老、非常基本的现象：**motor equivalence**。

---

## 2.2 Skild：身体本身变了，任务级行为仍能重新实现

Skild 官方：

https://www.skild.ai/blogs/omni-bodied

它报告同一个 omni-bodied model 在没有针对测试故障 fine-tuning 的情况下：

- calf 被截短、少 4 DoF 后，约 7–8 秒重新调整 thigh swing 并恢复 locomotion；
- knees 被锁后，约 2–3 秒重新分配重心并形成三腿 gait；
- wheels 被 jam 后，发现 wheel command 不再产生前进效果，转成 walking gait；
- wheels 恢复后又切回 rolling；
- 加 stilts 后重新调整 timing / foot placement。

这和 Figure 的 manipulation substitution 来自不同公司、不同任务族，却共同暗示：

> **大规模 physical pretraining 可能让模型学到“任务目标”与“具体身体实现”之间的某种解耦。**

当然，这些公司 demo 不能直接作为科学结论；但它们足够作为 phenomenon-first 起点，告诉我们这里值得做 controlled audit。

---

# 3. 从神经科学 / motor control 借来的正确问题：Motor Equivalence / Goal-Equivalent Manifold

经典 motor-control 的基本事实是：身体通常是冗余的。

同一个 task effect 可以由大量不同 body states / trajectories 实现。

用 Goal Equivalent Manifold（GEM）语言：

\[
\mathcal{G}=\{x\mid f(x)=0\}
\]

其中：

- `x` 是 body / motor space；
- `f(x)` 是 task-level error；
- `\mathcal{G}` 是所有能把 task error 做到 0 的 motor solutions。

参考：

- Goal-equivalent manifold overview / motor variability: https://pmc.ncbi.nlm.nih.gov/articles/PMC3858478/
- task-level equifinality discussion: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230222/

这给 foundation robot policy 一个非常自然的问题：

> **模型学到的是一条 demonstration trajectory，还是任务真正允许的 goal-equivalent solution class？**

这和普通 robustness 不一样。

robustness 常问：

```text
scene/object perturb 以后还能不能成功？
```

这里问：

```text
世界和任务都不变，
只拿掉 demonstrator 选择的 motor solution，
policy 会不会主动迁移到另一个 task-equivalent solution？
```

这是一种非常直接的 **task abstraction vs motion imitation** identification。

---

# 4. 为什么这不是 B 的换皮

B 问：

> **How Do Robot Foundation Policies Generalize Actions?**

主要分 retrieval / interpolation / composition / synthesis，并关心 foundation scaling 如何改变 action generation mechanism。

D 不是再问 action support。

D 固定：

- 同一个 robot embodiment；
- 同一个 task；
- 同一个 environment effect；
- 甚至可以固定 initial state。

唯一改变的是：

> **示范里反复出现的那条 body-space realization 突然不可用了，但 task-level feasible set 仍非空。**

D 因此问的是：

```text
trajectory imitation
vs
task-effect / motor-equivalence abstraction
```

它可以作为 B 的一个未来实验轴，但作为 research question 本身是独立的。

---

# 5. 最关键的开源 identification：SIMPLE 的 task definition 与 demonstration realization 天然分离

这一轮非常重要的发现来自：

**SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation**  
https://github.com/physical-superintelligence-lab/SIMPLE

SIMPLE 已公开：

- 50+ whole-body humanoid tasks；
- MuJoCo / IsaacSim；
- teleoperation + automated data generation；
- Ψ₀ / GR00T / π0.5 / DreamZero 等接口。

代表任务包括：

- `G1WholebodyOpenTrashCanTeleop-v0`
- `G1WholebodyPushOfficeChairTeleop-v0`
- `G1WholebodyOpenFaucetTeleop-v0`
- `G1WholebodyOpenOvenTeleop-v0`
- `G1WholebodyCloseDoorTeleop-v0`
- BendPick / Handover 等。

## 5.1 OpenFaucet：task success 只看 faucet effect

源码：

https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/src/simple/tasks/g1_wholebody_open_faucet_teleop.py

成功条件本质只看：

```python
faucet_joint0_qpos > 0.7 or faucet_joint0_qpos < -0.7
```

也就是说任务定义是：

> **faucet 被打开。**

它不要求：

> 必须右手打开。

但是同一个文件的 automated decomposition 却写死：

```python
hand_uid="dex3_right"
lock_links=["left_hand_palm_link"]
```

所以 demonstration pipeline 实际只展示了一个 arbitrary motor realization。

---

## 5.2 CloseDoor：同样的结构

源码：

https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/src/simple/tasks/g1_wholebody_close_door_teleop.py

成功条件只看 door joint 是否达到 closed region。

但是 decomposition 同样使用：

```python
hand_uid="dex3_right"
lock_links=["left_hand_palm_link"]
```

于是存在一个非常漂亮的统计结构：

```text
Task specification:
    environment effect only

Training demonstration:
    one particular right-hand realization
```

这恰好是我们想识别的问题。

---

# 6. 可用的公开 foundation policy：Ψ₀

**Ψ₀: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation**  
RSS 2026

repo：

https://github.com/physical-superintelligence-lab/Psi0

它公开：

- model；
- data；
- training / finetuning；
- SIMPLE simulation integration；
- SONIC whole-body controller integration。

架构上：

```text
Qwen3-VL backbone
   -> multimodal diffusion / flow action expert
   -> whole-body action chunks
   -> RL tracking controller
```

因此第一枪不需要等待 Figure / Skild 开模型，也不需要自己训练一个巨大 humanoid foundation model。

优先平台可以直接：

> **Ψ₀ + SIMPLE**

然后再视结果扩展到 HEX / FRoM-W1 / 其他公开 whole-body policy。

---

# 7. D 的最干净第一枪

核心原则：

> **不改变 task effect，只改变 available motor realization。**

选 2–3 个 task-effect 非常明确、同时身体上有冗余的 SIMPLE task，例如：

- OpenFaucet；
- CloseDoor；
- PushOfficeChair / OpenOven（先做 feasibility audit 再冻结）。

同一个 policy、同一批 initial states，做以下条件。

## A. Canonical

完全原始 setting。

验证 model 在标准 task 上有足够 baseline success。

## B. Local redundancy intervention

限制 demonstrator 常用的某个 joint / DoF，但保持**同一只手仍然物理可完成任务**。

问：

> policy 会不会利用同一 effector 内部的 kinematic redundancy？

## C. Cross-effector intervention

让 canonical right-hand solution 不可用，但保证 left hand / alternative body part 仍能完成 task effect。

问：

> policy 会不会真正跨 effector substitution？

## D. Functional occupation

不是“软件把手锁死”，而是让右手真实拿着一个物体 / 被任务占用。

这更接近 Figure 的自然 phenomenon，也避免模型把 actuator fault 当纯 OOD bug。

## E. Impossible control

让 canonical + alternative solution 都不可达。

这是 negative sanity control：不能把乱动偶然成功解释成 motor equivalence。

## F. Alternative-solution oracle

用 scripted / teleop / motion planner 证明当前 intervention 下 task **确实存在 alternative feasible solution**。

这一条必须有，否则失败无法区分：

```text
policy 不会 substitution
```

和

```text
我们把任务物理做死了
```

---

# 8. 第一枪测什么

不要一开始上 SAE。

行为层面已经足够。

至少记录：

1. **task-effect success**；
2. **effect-space error**（door/faucet/object state）；
3. **body-part activation shift**；
4. **canonical-effector retry count**；
5. **adaptation latency**；
6. **motor-space distance from demonstrations**；
7. **large motor change with small task-effect error** 的比例。

后者可以直接对应 motor equivalence：

```text
body trajectory 明显变了
but
task effect 仍保持
```

如果 task geometry 足够简单，还可以借 GEM / uncontrolled-manifold 思路，把 perturbation 分成：

- task-relevant direction；
- approximately task-null / goal-equivalent direction。

但这不是 prerequisite，不要把第一枪复杂化。

---

# 9. 强结果 / 反结果都意味着什么

## 9.1 强 motor equivalence

如果 foundation policy 在 canonical effector 被拿掉以后，会主动切换到 alternative body solution，并维持 task effect：

> **foundation physical pretraining may induce task-level motor abstractions that are less tied to the demonstrated body trajectory than standard behavioral cloning suggests.**

这会把 Figure / Skild 的“emergent flexibility”第一次变成 controlled mechanism evidence。

后续 method 空间自然是：

- effect-space supervision；
- effector dropout；
- motor-equivalence augmentation；
- goal-equivalent contrastive objective；
- task-space / null-space regularization；
- deliberately diversify body realizations in demonstrations。

## 9.2 仍然执着 canonical trajectory

如果 policy 明明看到了 intervention，却反复尝试右手原路径，而 alternative solution 是可行的：

> **current robot foundation policies may still behave primarily as large demonstration-trajectory models rather than effect-level controllers.**

这个结论同样很大。

它会重新解释：

- whole-body data scaling；
- human-video transfer；
- cross-embodiment claims；
- “emergent recovery” demo。

也会说明未来需要从：

```text
cover more trajectories
```

转向：

```text
learn equivalence classes of physical effects
```

## 9.3 只有 specialized omni-body model 能做到

如果 Ψ₀ /普通 VLA 不行，Skild-like morphology-diverse model 才行：

> **morphology diversity may not merely improve robustness; it may change the abstraction level at which motor behavior is represented.**

这又会形成非常漂亮的 scaling / representation story。

---

# 10. collision audit：邻域很多，但没有找到 exact same question

## 10.1 Cross-embodiment transfer

大量工作研究：

> 一个 robot 学的 skill 能不能迁移到另一个 morphology？

例如 morphology-aware IL、cross-robot correspondence、HEX / Qwen / GR00T 等。

D 固定的是**同一个身体**，只干预可用 motor solution，因此 identification 不同。

## 10.2 Fault-tolerant control

FT-WBC / adaptive control / body-schema work 会专门：

- 训练 fault estimator；
- inject actuator failures；
- 设计 adaptation module / fault-aware controller。

D 不提出 fault method，而问：

> **一个已经训练好的 generic foundation policy，在没有专门 failure training 的情况下，是否已经形成 spontaneous task-equivalent substitution？**

## 10.3 Whole-body control / skill blending

WholeBodyVLA、SkillBlender、OmniContact、SONIC 等证明 whole-body coordination 很重要，但通常主问题是：

> 怎么学一个稳定、可控的 whole-body policy？

不是：

> demonstrated solution 被拿掉后，policy 学到的是不是 task equivalence class？

## 10.4 Classical redundancy / PbD

经典 robotics 早已知道：

- redundant body admits multiple solutions；
- task space 与 joint space 不同；
- imitation 不一定等于 exact mimicking。

因此我们绝不能 claim “motor equivalence”概念本身新。

真正的新问题在于：

> **现代大规模 foundation robot policy 是否自然跨越了 motion mimicking → task-effect abstraction 这一步？**

这正是 2026 的 foundation-policy claim 需要被审计的层级。

---

# 11. heterogeneous action padding / missingness：真实结构，但本轮不升候选

Round 9 又深挖了一次大型 heterogeneous VLA 的 action bookkeeping。

真实例子：

- LingBot-VLA 2.0 把多种 robot 统一到固定高维 action vector，未使用 DoF padding；
- GR00T N1.7 有公开 issue 证明 padded action dimensions / timesteps 在 final loss mask 前已经进入 noise、action encoder 和 DiT self-attention，因此能影响 valid output；
- LingBot 又有 `action_is_pad` 未真正进入 loss masking 的公开 issue；
- Qwen-VLA 则显式使用 per-channel / per-step validity mask；
- Green-VLA 明确强调 mask 可避免 padding 的 spurious gradients；
- Galaxea G0.5 只生成 active motion groups，直接避免无意义 padding。

这支持一个真实问题：

> **Does action-space missingness itself become an embodiment shortcut?**

即模型是否通过“哪些维度存在 / 被 pad / 被 mask”来识别 data source / embodiment，而不是真正从物理行为理解 embodiment。

但目前还缺一个强 behavioral anomaly 证明这是 foundation pretraining 的主要瓶颈。

继续下去很容易退成：

> 哪种 mask 实现最好？

因此按当前顶会尺度标准：

**记录，但不摘候选。**

---

# 12. tactile：`touch changes intent or only execution` 被架构工作占得太近

一度考虑：

> **When vision and touch disagree, does a VLA update its belief / intent, or use touch merely as a local reflex signal?**

但 2026 已经有：

- VLA-Touch：同时把 tactile 放进 high-level planning 与 low-level refinement；
- TouchWorld：predictive tactile subgoal + fast tactile residual；
- 多种 tactile-VLA / force-VLA / tactile-WAM。

因此 broad decomposition 已经进入现有 method design。

**结论：不摘。**

---

# 13. human-video camera motion：被 ActiveMimic / EgoMI 正面做掉

曾考虑：

> human egocentric pretraining 的收益会不会大量来自 head / camera motion 本身泄露 action / attention / intent？

但：

- ActiveMimic 已明确把 camera motion 当成 active-perception action，而不是 nuisance；
- EgoMI 直接利用 head-hand coordination / pre-action fixation；
- VITRA 等恢复 camera + hand trajectory。

所以：

> camera motion is hidden supervision

已经不能作为新母题。

**结论：砍。**

---

# 14. uncertainty：multimodality vs ignorance 已有直接历史

另一个自然问题：

> diffusion / flow action variance 大，到底表示“模型不会”，还是“这里有多种都正确的动作”？

如果混淆，confidence gating / human intervention / replanning 都可能误判。

但 Diff-DAgger 早已明确指出 ensemble disagreement 会把合法 demonstration multimodality 当 uncertainty；2026 又已有 conformal / confidence / uncertainty gating 工作。

**结论：不摘。**

---

# 15. body schema：broad concept 已被 2026 work 正面占据

“模型有没有一个可更新的身体图式”本身非常大，但 2026 已出现：

- diffusion-based body schema learning under muscle rupture / actuator jam；
- embodied self-model / morphology-adaptation framework。

所以 D 不能包装成 generic body-schema paper。

D 的价值正是更具体：

> **same body, same task effect, alternative motor solution exists — does a generic foundation policy exploit the equivalence without dedicated body-schema training?**

---

# 16. 最新前沿：GEN-1.5 / physical prompting 很重要，但本轮不再硬提 E

2026-08-19 Generalist 公布 GEN-1.5。当前主要是公司披露 / 媒体转述，尚不应把所有 claim 当 peer-reviewed evidence。

公开报道包括：

- https://ai.nexsight.co/articles/2026/08/20/generalist-gen15-one-shot-robotics/
- https://www.eweek.com/news/generalist-gen-1-5-robot-one-shot-learning/

公司声称：

- 3–12 秒 single demonstration 可放入 context；
- 无 gradient update 做 physical prompting；
- 10 个短 manipulation tasks 平均约 59% success；
- 少量 few-shot gradient steps 可进一步提高；
- 能 recovery / improvise；
- 两个 demonstrations 可以 composition，并补出两段 prompt 都没有显式展示的 intermediate motion。

这是非常强的新现象源。

但 broad question：

> **Can robots learn from a demonstration in context?**

早已被：

- ICRT；
- Behavior Prompting Policy (BPP)；
- MimicDroid；
- RoboSSM；
- visual-reasoning ICIL 等

系统占据。

## 16.1 曾考虑的机制题

> **What Does a Physical Prompt Specify?**

一次 demonstration 同时纠缠：

- goal / desired effect；
- object correspondence；
- strategy；
- path；
- timing / speed；
- effector / embodiment realization。

很自然可以问模型哪些会 transfer、哪些会丢掉。

但 BPP 已经明确把 behavior prompt 定义成同时提供 `what + spatial/temporal how`，并直接研究：

- observations / actions / proprioception 的 prompt representation；
- prompt observation frequency；
- temporal alignment；
- drawing trajectory；
- folding / grasp strategy；
- task diversity 对 prompting 的作用。

ICRT 又把 task 拆成 motion primitive + interacted object，新的 visual-reasoning ICIL 专门增强 task intent inference。

所以这一方向虽然仍可能做更强的 factorized causal swap，但它离现有 ICIL 主线比 D 更近。

**当前处理：作为强搜索线记录，不升 E。**

未来如果出现可直接审计 GEN-1.5 类大模型的 checkpoint / API，再考虑做：

```text
goal fixed, strategy swapped
strategy fixed, object/effect swapped
effector changed, task effect fixed
```

的 prompt binding audit。

---

# 17. Round 9 shortlist

当前 active provisional shortlist：

## B — How Do Robot Foundation Policies Generalize Actions?

```text
retrieval
vs interpolation
vs composition
vs synthesis
```

重点是 foundation scale / diversity 是否改变 action-generation mechanism。

## C — Why Does Task Decomposition Help Robot Foundation Policies?

```text
planning / sequencing
vs controller-support matching
vs temporal reset / handoff
```

重点是 hierarchy gain 的真正来源。

## D — Do Robot Foundation Policies Learn Motor Equivalence Classes?

```text
demonstration trajectory
vs task-effect abstraction
```

重点是 canonical motor solution 被拿掉、但 task 仍可解时，policy 是否找到 goal-equivalent body solution。

A 继续保持 archived / inactive。

---

# 18. 下一轮继续搜什么

不要因为出现 D 就停止找题。

Round 10 可以继续沿完全不同的区域：

1. **effect-level abstraction beyond body redundancy**：同一物理目标下 object/tool/interaction path 等实现自由度是否被正确抽象；
2. **physical prompting 的 emerging mechanism**：等更强的一手资料 / open checkpoint 出来后再审，不抢当前 method paper 的题；
3. **foundation policy 的 internal intervention model**：不是 WAM generic controllability，而是 policy 在 closed-loop 中是否区分 self-caused vs externally-caused state change；
4. **最新 2026-08 model reports / repo issues**：继续找作者为把模型跑通而修掉、但尚未上升成研究问题的异常；
5. **跨领域迁移**：motor control、causal representation、meta-learning、control theory、human motor adaptation 继续提供问题结构，而不是只提供 method。

Round 9 的原则仍然是：**不凑题，不追求零 collision，但只把真正有大问题尺度、相对新叙事和干净实验的方向摘出来。**
