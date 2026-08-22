# 2026-08-22：VLA / WAM 机制选题搜索日志（第八轮）

> 状态：**继续搜索。本轮主体重新回到“找新的大问题”，不再把主要时间消耗在反复审 B。A 移出 active shortlist；B 保留；新增 provisional candidate C。**
>
> 本轮材料继续混合：2025–2026 顶会 / arXiv、最新开源整模型与 model card、公司 technical report / project page、真实机器人系统 paper、practitioner implementation，以及相邻的 hierarchical RL、system identification、active perception、causal / predictive representation 等领域。

---

# 0. 本轮重新校准搜索标准

上一轮之后需要明确一件事：

> **我们不是在找“世界上从来没有任何邻近工作”的题。**

ICLR / ICML / NeurIPS / RSS / ICRA / IROS 级问题本来就应该长在一个活跃领域里。

真正的筛选标准是：

1. **问题尺度够大。** 标题本身就是自然 research question，而不是某个 benchmark statistic；
2. **叙事角度相对新。** 已有工作可以覆盖邻边，但不能已经把我们的核心 causal decomposition 做完；
3. **第一枪实验相对新且干净。** 不靠 SAE / latent manifold / 大量控制变量才能勉强识别；
4. **正反结果都有解释价值。** 不能只有一个方向显著才值得写；
5. **成立后自然留下方法空间。** 但当前先找问题，不急着造 method。

Round 8 因此不再围绕 B 做 endless collision audit，而是大范围寻找新的母题。

---

# 1. 先记录一个整模型里很好的工程异常：history 作为 implicit embodiment identifier

## Qwen-RobotManip

- report: https://arxiv.org/abs/2606.17846
- blog: https://qwen.ai/blog?id=qwen-robotmanip

Qwen-RobotManip 在大规模 heterogeneous robot data 中强调 recent observation / state / action history 的作用。

作者给出的一个很有意思的解释是：history 不只提供 task progress，它还暴露了当前 embodiment / controller 的真实行为响应，因此可以充当某种 **implicit embodiment identifier / behavioral profile**。

直觉上：

```text
same desired command
    + different recent realized motion
        -> infer how this body/controller actually responds
```

这很容易长成一个大问题：

> **Can a robot foundation policy identify its own body / controller online from interaction history?**

甚至更一般：

> **Does a foundation policy infer how its current system works before deciding how to control it?**

但继续搜索后，这条不能升候选。

### direct / near-direct collision

**In-Context World Modeling for Robotic Control**  
https://arxiv.org/abs/2606.26025

这篇已经非常直接地把 system identification 变成 in-context adaptation：机器人先执行短段 task-agnostic self-interaction，从 history 中推断 camera / morphology / system variables，再无参数更新地适应新配置。

另外 history-conditioned cross-embodiment policy、DexFormer、adaptive control / universal policy 也都占据邻域。

### 结论

**不作为新候选。**

但这个异常留下一个值得继续盯的方向：现代 VLA 的 context 可能已经不只是 memory，而是在做 online system identification。

---

# 2. 新母题：机器人 foundation policy 是否真正理解“physical time”？——有意思，但最终仍太像数据 / action-space 问题

大规模机器人数据里存在一个很基础的结构：

- 5 Hz / 10 Hz / 20 Hz / 50 Hz；
- 不同 action chunk horizon；
- 不同 camera / proprio rate；
- 不同 execution latency；
- human / robot / sim 数据的时间离散方式不同。

但很多 policy 仍然使用：

```text
step 1, step 2, ..., step H
```

而不是显式 physical duration。

这自然引出：

> **Does a robot foundation policy learn behavior in physical time, or in dataset-specific step units?**

一个真正 physical behavior representation 理论上应该对合理的 time reparameterization 有某种结构稳定性；否则所谓跨 embodiment / dataset transfer 可能仍然绑在数据采样率上。

### 为什么没有升候选

搜索后发现这一邻域已经有很多直接处理：

- ACE-Ego-0 用 physical-time-aligned action horizon 解决 heterogeneous source rate mismatch；
- IROS 2026 trajectory standardization / ISR 直接去 human demonstration 的 speed / pause / temporal density variation；
- TempoVLA 等开始显式做 temporal adaptation；
- action-space design 工作也已经系统比较 delta / absolute / horizon / frequency。

继续做很容易退化成：

> 哪个 temporal normalization 更好？

这不够我们当前要求的尺度。

**结论：不摘。**

---

# 3. 新母题：Do Robot Policies Act on What They Know?——很强，但已经形成研究线

这一轮一个非常有吸引力的跨论文现象是：

> VLA representation 里常常已经存在 task-relevant information，但默认 action generation path 未必真的利用它。

代表性证据包括：

- frozen OpenVLA / π0.5 representation 可以线性读出明显的 future-success / value-like structure；
- VLA representation 中可以 decode task progress；
- semantic re-binding 工作发现模型内部保留正确 task semantics，但 action path 仍可能执行错误动作；causal feature replacement 可以显著恢复行为；
- COAST / feature steering 等已经开始直接干预成功 / 失败方向。

这可以形成一个非常漂亮的题：

> **Do robot policies act on what they know?**

或者：

> **When a VLA internally represents that an action is likely to fail, does that knowledge actually control action generation?**

### 为什么没有升 C

这个邻域在 2026 已经快速从 probing 进入 causal intervention：

- success / value probes；
- task-progress probes；
- semantic feature swap；
- success steering；
- action-feature manipulation。

所以 broad thesis：

```text
representation may contain useful information that action head underuses
```

已经开始成为一条完整 research program。

如果继续只能问：

> “哪一层的 value feature 没被 readout？”

又会压窄。

**结论：不摘。**

---

# 4. 新母题：Do Standard VLAs Already Contain a World Model?——被一篇非常直接的诊断工作挡住

WAM 与 VLA 的一个最自然机制问题是：

> WAM 显式训练 future prediction；标准 VLA 只训练 observation → action。那标准 VLA 内部到底有没有隐式形成 action-effect / predictive world representation？

如果有：

> WAM 的 qualitative advantage 可能不是“从无到有学会物理”，而是让已有 predictive structure 更可用。

如果没有：

> WAM 与 VLA 的 representation computation 才存在真正 qualitative difference。

这听起来很像一个完整 ICLR/RSS 题。

但搜索撞到：

**Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA**（2026）

已经跨多个 policy 做 behavioral + representational diagnosis，直接讨论 reactive / predictive / memorized structure，并发现不同 WAM/VLA family 在 predictive information 上明显不同。

再叠加 UniVLA、VLA-JEPA、LARA、AHEAD 等 world-aware VLA 工作，这个 broad question 已经太近。

**结论：砍。**

---

# 5. 新母题：Can Robot Foundation Policies Adapt to Hidden Physical Dynamics from Their Own Consequences?

这一条本轮仍然值得保留为未来搜索线。

构造一个视觉上几乎相同的场景，但改变：

- mass；
- friction；
- damping；
- compliance；
- object 是否卡住；
- controller gain / actuator response。

第一次动作之后，机器人获得新的 evidence：

```text
action
   -> realized motion / contact consequence
```

于是问题是：

> **Does the policy update subsequent control from the consequence of its own action?**

这比普通 visual OOD 更接近真正 physical intelligence：

```text
interaction
   -> infer hidden dynamics
   -> adapt action
```

### 为什么本轮不摘

问题本身很自然，但 system identification / adaptive control 是老领域，2026 又已经出现 `In-Context World Modeling` 这种 foundation-policy 版本。

当前还缺一个足够强、足够现代的 anomaly，证明：

> 大型 VLA 明明有 history，却在 hidden object dynamics 上系统性失败 / 或反常地成功。

没有 phenomenon-first evidence 时，现在摘会有拍脑门风险。

**结论：保留搜索线，不摘候选。**

---

# 6. Failure / recoverability / irreversibility：很自然，但 2026 已经太拥挤

曾考虑：

> **Do robot foundation policies know which mistakes are recoverable?**

因为真实 manipulation failure 至少分：

```text
recoverable deviation
vs
catastrophic / irreversible failure
```

一个 policy 如果分不清，可能会在已经不可恢复的 state 上继续重复动作，或者把本可恢复的小偏差错误 reset。

但搜索很快撞到：

- FLARE：显式区分 task policy 无法恢复的 catastrophic failure，并学习 reset / retry；
- TCoT、FPC-VLA、Dream2Fix、FailSafe、ViFailback、Foresight 等已经把 failure anticipation / correction / fallback / recovery 做成活跃赛道。

所以 generic recoverability question 不够新。

**结论：砍。**

---

# 7. Physical feasibility / impossible instruction：也已经有人正面做

另一条非常自然：

> VLA 能理解语言，不代表当前物理世界里目标可实现。

也就是区分：

```text
semantic competence
vs
physical feasibility / reachability
```

例如：

- 目标被固定；
- 容器打不开；
- 物体不存在；
- required clearance 不足；
- 当前 arrangement 已经让目标不可达。

但 2026 已有 PhysReflect-VLA、goal reachability / physical feasibility work，以及对 contradictory / impossible instruction 下 VLA “仍继续做 plausible action”的系统研究。

**结论：不摘。**

---

# 8. 整模型数据里另一个很漂亮的结构：hindsight state-transition language

## Xiaomi-Robotics-1

- arXiv: https://arxiv.org/abs/2607.15330
- project: https://robotics.xiaomi.com/xiaomi-robotics-1.html

Xiaomi-Robotics-1 的预训练数据非常值得注意：超过 100K 小时 UMI real-world trajectories，自动把 trajectory clip 标成**场景状态变化的自然语言描述**，再用这种 transition language 条件 action learning。

这不是普通人工 task instruction：

```text
先看到完整片段
    -> 描述实际上发生了什么变化
    -> 用这个 hindsight transition description 条件动作
```

而部署时更像：

```text
先给 desired instruction
    -> 产生让目标发生的动作
```

所以一度出现一个很诱人的问题：

> **Does hindsight outcome language teach the same control abstraction as prospective goal language?**

### 为什么没摘

继续查发现这一条有很长谱系：

- goal-conditioned RL 的 hindsight relabeling；
- CoRL 2023 goal-change representation；
- Figure Helix / interactive-language data 的 hindsight instruction generation；
- 2026 VLA hindsight RL。

所以“用结果反标目标”不是新问题。

更细地问 causal-direction mismatch 又容易变成一个 dataset-label ablation。

**结论：不摘。**

---

# 9. Active information gathering：方向很大，但已经有人把它直接命名成新 manipulation problem

从 embodied intelligence 的基本问题出发，曾考虑：

> **Do robot foundation policies ever act to learn, rather than act only to make immediate task progress?**

比如：

- 移开遮挡物只是为了看后面；
- 轻推物体只是为了估计 mass / friction；
- 调整 camera / arm pose只是为了获得更好信息；
- probe contact 只是为了确认插孔位置。

这是 classic information-seeking / epistemic action 在 foundation robot policy 上的自然版本。

但 2026 已经出现：

**Towards Exploratory and Focused Manipulation with Bimanual Active Perception**  
https://arxiv.org/abs/2602.01939

它明确把“主动收集完成任务所需信息”上升为 `Exploratory and Focused Manipulation` 新问题，并建立 benchmark / strategy。

ActiveVLA、CURA-PPO 等也在 active perception 上推进。

所以 broad problem 已经有人在占。

**结论：不作为 C。**

---

# 10. 本轮真正长出来的新候选：Why Does Task Decomposition Help Robot Foundation Policies?

这是 Round 8 大范围搜索后真正值得单独摘出的新问题。

# **Why Does Task Decomposition Help Robot Foundation Policies?**

更机制化：

> **When hierarchy makes a foundation policy dramatically better at long-horizon tasks, is the gain really coming from high-level reasoning, or from repeatedly translating the global task back into atomic instructions that the low-level VLA already knows how to execute?**

一句更尖锐的版本：

> **Does the planner reason better, or does it keep the controller on-support?**

---

# 11. 为什么这个问题不是拍脑门：三个结果形成了一个很强的 tension

## 11.1 RoboHiMan：不需要聪明 planner，rule-based decomposition 就能拿回很大收益

**RoboHiMan: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation**  
https://arxiv.org/abs/2510.13149

最重要的真实机器人结果：

同一个 `π0.5` low-level policy：

```text
flat / vanilla: 17.5%
rule-based planner + same π0.5: 47.5%
```

扰动下：

```text
10.0% -> 27.5%
```

而这个 rule-based planner 的核心只是按照人工 atomic sub-instruction 切换，不是一个强 reasoning model。

所以：

> **hierarchy gain 并不自动等于 reasoning gain。**

---

## 11.2 What Matters in Orchestrating Robot Policies：低层 steerability 决定 hierarchy 能不能工作

**What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents**  
https://arxiv.org/abs/2606.10267  
https://jiahenghu.github.io/hi-vla/

这篇系统研究 planner、VLA、handoff、memory、observation representation。

它得到一个对本题非常关键的结果：

> low-level VLA 必须保持对 planner language subgoal 的 steerability。

甚至某些 in-domain fine-tuning 会让 VLA 更像固定 movement pattern matcher，语言响应下降，最终 hierarchical long-horizon performance 反而变差。

这说明 hierarchy 不只是“planner 算对答案”，还要求 planner 输出落在 low-level controller 真正能接收的接口上。

---

## 11.3 Diagnosing Compositional Generalization：很多失败不是 skill 不存在，而是 instruction steering 不出来

**Diagnosing Compositional Generalization in Sequential Robot Tasks**  
https://arxiv.org/abs/2607.29687

该工作发现 sparse compositional training 的很多 OOD failure 来自 instruction steering / coverage，而不是缺少 low-level skill。

一个 setting 中，每 task 只补一条 demonstration，OOD success 就从 `0.4%` 到 `54.7%`。

所以：

```text
model does not have the motor capability
```

和

```text
motor capability exists but current instruction cannot reliably call it
```

必须分开。

---

# 12. 这里真正尚未被干净拆开的机制

现代 hierarchy 一次同时改变：

```text
A. sequencing / reasoning
B. language abstraction / subgoal form
C. execution horizon / termination
D. re-conditioning frequency
E. failure recovery opportunity
```

所以：

```text
hierarchy success > flat VLA success
```

并不能告诉我们 hierarchy 为什么成功。

本题真正想拆的是：

### Mechanism 1 — Planning

高层模型根据当前 state 推断“下一步应该做哪个 skill”。

### Mechanism 2 — Controller-support matching

高层模型把 abstract/global instruction 编译成一个 low-level VLA 已经稳定可执行、可 language-steer 的 atomic instruction。

### Mechanism 3 — Temporal handoff / reset

分段执行本身让 policy 在每个 subtask boundary 重新条件化，避免一个错误 chunk / drift 持续传播。

已有 hierarchy work 知道这些 component 都重要，但还没看到一个 fixed-controller causal experiment 系统回答：

> **hierarchy gain 到底主要来自哪一项？**

---

# 13. 为什么这个问题符合我们要求的顶会尺度

题目不是：

> “RoboHiMan 上换 prompt 能涨多少？”

而是：

> **Why Does Task Decomposition Help Robot Foundation Policies?**

这和最近好的 mechanism papers 的尺度一致：

- `Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?`
- `What Matters in Orchestrating Robot Policies?`

它问的是一个领域现在大规模采用的 paradigm 为什么有效。

### AI 顶会叙事

ICLR / ICML / NeurIPS：

- hierarchy computation；
- pretrained capability utilization；
- instruction-action interface；
- compositional generalization；
- modular foundation policies。

### Robot 顶会叙事

RSS / ICRA / IROS：

- long-horizon manipulation；
- planner-controller interface；
- closed-loop handoff / termination；
- robust deployment。

而且即使实验先从一个公开 VLA + simulation 开始，问题本身不被 benchmark 锁死。

---

# 14. 第一枪可以非常干净

不要先训练新的 planner。

固定一个公开 low-level VLA；选择它已经稳定会做的 atomic skills，再构造 2–4 step composition。

同一模型、同一 initial states，比较：

```text
A. flat global instruction

B. oracle atomic subgoal sequence
   + oracle / success-based switching
   + NO learned planner

C. learned VLM planner

D. same oracle subgoals
   but canonical atomic wording
   vs paraphrase
   vs abstract/global-like wording

E. global language
   + same boundary/reset schedule as B
```

这样一次可以拆：

```text
planning intelligence
vs
language/support narrowing
vs
temporal reset
```

最关键 prerequisite：

> **Oracle decomposition without planner intelligence 是否已经能恢复 hierarchy 的大部分 gain？**

如果不能，题快速降级。

如果能，再继续研究为什么 atomic subgoals 更可执行。

---

# 15. 正反结果都成立

### Positive for support-matching hypothesis

假设：

```text
flat             20%
oracle atomic     50%
learned planner   55%
```

则大结论是：

> **Hierarchical robot foundation policies work in large part because decomposition keeps the low-level controller in an executable / steerable regime, not simply because the high-level model reasons better.**

这会直接改变系统设计：

- planner 应优化 controller compatibility，而不只 semantic correctness；
- fine-tuning 不能牺牲 low-level language steerability；
- subgoal interface 应与 low-level repertoire 一起设计。

### Negative for support-matching hypothesis

如果：

```text
flat ≈ oracle atomic << learned planner
```

则反过来证明：

> hierarchy 真正的收益主要来自 state-aware online reasoning / sequencing。

这同样回答了一个领域级问题。

### Temporal-reset result

如果 oracle decomposition 帮助主要来自 boundary/reset，而 atomic wording 不重要，那么：

> hierarchy 的关键不是 reasoning 或 semantic compilation，而是更频繁地重新闭环。

这又会连接到 action chunking / receding-horizon control。

因此不是一个只有单向显著才有意义的题。

---

# 16. collision audit：有邻近，但当前仍有相对新的叙事与实验

这里不要求“完全没人碰”。

### RoboHiMan

已经分离 planner / executor error，也已经证明 rule-based planner 很强。

但它没有把 rule-planner gain 因果拆成：

```text
atomic-language support
vs
temporal handoff
vs
state-aware planning
```

### What Matters in Orchestrating Robot Policies

这是最近邻。

它系统回答：

> hierarchy 各组件该怎么选？

我们的题则问：

> **固定 low-level foundation policy 后，decomposition 本身为什么带来这么大的 gain？**

### Diagnosing Compositional Generalization

已经证明 instruction steering / coverage 可以是瓶颈。

但它不是 hierarchy mechanism study，也没有比较 flat global instruction、oracle decomposition、learned planner、boundary-only control。

### classical HRL / options

老的 hierarchical RL 已经知道 manager subgoal 必须 reachable / compatible，不能把这个理论事实包装成新贡献。

现代的新角度在于：

> foundation VLA 的 worker interface 是 natural language，且 worker 已经有巨大的 pretrained motor repertoire；高层 VLM 是否事实上扮演一个 **controller-aware semantic compiler**？

所以当前判断是：

> **有 collision，但相对叙事角度和 controlled experiment 都仍然够新。**

这符合本轮重新校准后的 novelty 标准。

---

# 17. 新候选 C

单独摘出：

`topic_search_logs/candidates/why_does_task_decomposition_help_robot_foundation_policies.md`

状态：

> **PROVISIONAL SEARCH CANDIDATE — not a registered Topic.**

标题：

# **Why Does Task Decomposition Help Robot Foundation Policies?**

副问题：

> **Does the planner reason better, or does it keep the controller on-support?**

目前我认为它达到：

- 顶会自然问题尺度；
- 有真实 anomaly，而不是拍脑门；
- 相对现有 hierarchy literature 有新的 causal decomposition；
- 第一枪可以很便宜、很干净；
- 正反结果都有意义。

所以值得进入 provisional shortlist。

---

# 18. A / B 在 Round 8 后怎么处理

## A — What Does History Actually Model in Robot Policies?

**移出 active shortlist。**

原因不再展开反复审计：IntentVLA、history causal audit、BPP、RoboMME / memory-VLA 等已经把 broad history/memory mechanism 围得很紧。

最后剩下的 `world-state memory vs producer/strategy memory` 虽然仍可能没人完全同题，但已经越来越依赖一个窄 decomposition 才维持 novelty，不符合我们对顶会题目尺度的要求。

保留历史 candidate 文件，不删除，作为搜索记录。

## B — How Do Robot Foundation Policies Generalize Actions?

**继续保留。**

不再要求它“没有任何 retrieval / composition / primitive 邻近工作”。它当前的相对新角度仍然成立：

> **统一审计 successful OOD behavior 到底属于 retrieval / interpolation / composition / support escape，以及这个 action-generalization mechanism 是否随 foundation pretraining scale / diversity 发生 qualitative transition。**

B 继续作为 active provisional candidate。

---

# 19. Round 8 之后的 active provisional shortlist

现在是：

## B — How Do Robot Foundation Policies Generalize Actions?

核心：

```text
retrieval
vs interpolation
vs composition
vs support escape / synthesis
```

关注 foundation pretraining 是否改变 action-generation mechanism。

## C — Why Does Task Decomposition Help Robot Foundation Policies?

核心：

```text
reasoning / sequencing
vs controller-support matching
vs temporal reset / handoff
```

关注 hierarchy 为什么有效。

当前不追求候选数量。

---

# 20. 本轮留下、下一轮可以继续搜但未摘出的线

### Hidden physical dynamics adaptation

> foundation VLA 能否从自己动作的后果中在线推断 mass / friction / controller dynamics，并改变后续控制？

问题很大，但目前还缺现代 foundation-model anomaly。

### Pretraining supervision direction

Xiaomi 的 state-transition descriptions、human-video / future prediction、robot action labels 之间到底分别塑造什么 computation，仍值得继续从整模型 scaling report 里找异常；但不要退化成 supervision ablation。

### Planner-controller interface failures

Round 8 已经摘出了 task decomposition mechanism，但还可以继续寻找：planner 语义正确却 controller 无法执行的真实 failure taxonomy，尤其在 open-vocabulary / unseen embodiment setting。

### Physical interaction as information

Active perception 已经有人占 broad question，但“通过 manipulation probe hidden dynamics”与 foundation VLA 的关系仍可能有独立空间。

---

# 21. Round 8 总结

本轮真正的进展不是又扫了多少关键词，而是重新把搜索过程拉回正确轨道：

> **已有候选不要无限做零碰撞审计；主要精力应该继续找新的自然大问题。**

广搜之后，大部分看起来很漂亮的问题仍然被砍：

- closed-loop reactivity：已经成为 2026 主赛道；
- online system ID：ICWM 已经正面做；
- physical time：容易缩成 preprocessing；
- representation→action utilization：已经形成 causal steering research line；
- implicit world model in VLA：已有直接 comparative diagnostics；
- recoverability / feasibility：赛道已经拥挤；
- hindsight state-transition labels：有长历史谱系；
- active information gathering：已经有人明确提出 EFM。

真正留下来的 C 是从一个**已经存在但没有被充分解释的 hierarchy success**中长出来：

```text
rule-based decomposition already helps a lot
+
low-level steerability is essential
+
compositional failure often occurs despite existing low-level skills
```

因此 Round 8 的核心新问题是：

# **Why Does Task Decomposition Help Robot Foundation Policies?**

这一次不是为了凑第三个候选；是因为它已经同时满足：问题自然、现象真实、相对新叙事、实验干净、正反都能解释、方法空间自然。
