# Why Does Task Decomposition Help Robot Foundation Policies?

> Status: **PROVISIONAL SEARCH CANDIDATE — not a registered Topic.**
>
> 来源：Round 8。这个文件只把本轮经过大范围搜索后仍值得继续验证的问题单独摘出来；它不进入根目录正式 Topic，也不因为进入这里就默认成立。

---

## 1. 问题

> **Why does task decomposition help robot foundation policies?**

更机制化地说：

> **When a hierarchical VLA greatly outperforms a flat VLA on long-horizon tasks, how much of the gain comes from genuine high-level planning/reasoning, and how much comes from repeatedly translating the global task back into atomic instructions that the low-level policy already knows how to execute?**

一个更尖锐的表述是：

> **Does the planner reason better, or does it keep the controller on-support?**

这里的 `support` 不要求定义一个玄学 latent manifold。第一阶段只指一个行为上可识别的事实：某个 low-level VLA 对一组 atomic sub-instructions 已经稳定可执行、可语言 steering，而对整个 abstract / compositional instruction 明显失败。

---

## 2. 为什么这个问题现在自然

2025–2026 的 long-horizon robot systems 正迅速收敛到同一种结构：

```text
high-level VLM / agent
        -> language subgoal
        -> low-level VLA
        -> execution / success detection
        -> next subgoal
```

代表性工作包括 RoboHiMan、`What Matters in Orchestrating Robot Policies`、LoHo-Manip、Tool-Aligned VLA、VoLo、τ0-VLA 等。

常见解释是：

> flat VLA 缺少高层 reasoning；VLM planner 会分解、规划、纠错，所以 hierarchy 成功。

这当然可能是真的，但现有结果里有一个很重要的张力：**有时甚至不需要一个“聪明 planner”，只要把任务切回 atomic sub-instructions，性能就会大幅恢复。**

这说明 hierarchy 可能同时做了三件不同的事：

1. **Planning / sequencing**：决定下一步做什么；
2. **Controller-support matching**：把 global goal 编译成 low-level VLA 熟悉且可 steering 的 atomic instruction；
3. **Temporal handoff / reset**：在 subtask 边界重新条件化，减少错误累积和错误动作持续执行。

目前很多 hierarchy paper 把三者一起改变，因此最终看到的 hierarchy gain 并不能直接告诉我们主要机制是哪一个。

---

## 3. 这个题从什么已知现象长出来

### 3.1 RoboHiMan：rule-based planner 已经能拿回巨大的长任务增益

**RoboHiMan: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation**  
https://arxiv.org/abs/2510.13149

它的真实机器人实验非常关键。

同一个 low-level `π0.5`：

```text
flat / vanilla execution
    -> 17.5% compositional-task success

same π0.5 + rule-based planner
    -> 47.5%
```

在 perturbation setting 下：

```text
10.0% -> 27.5%
```

这里的 rule-based planner 并不是一个更强的 reasoning model。它使用人工标注的 atomic sub-instructions，并按一个简单规则切换到下一条。

因此至少在这个 setting 中：

> **大块 hierarchy gain 可以在几乎没有高层智能的情况下出现。**

这不是我们的结论，但它是一个非常干净的 anomaly。

---

### 3.2 Hi-VLA systematic study：低层 policy 的 steerability 是 hierarchy 成败的关键

**What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents**  
https://arxiv.org/abs/2606.10267  
https://jiahenghu.github.io/hi-vla/

该工作在统一框架下比较 planner、low-level VLA、termination、observation representation、memory 等设计。

一个与本题特别相关的结果是：

> hierarchy 要工作，low-level VLA 必须对 high-level planner 给出的 subgoal 保持足够强的 language steerability。

更反常的是，某些 in-domain fine-tuning 虽然让 low-level policy 更专门化，却会削弱 language responsiveness，最终反而伤害 long-horizon hierarchy performance。

这说明 planner 并不是在向一个任意 controller 发命令；它必须通过一个**低层 policy 实际可解释、可执行的接口**来控制。

---

### 3.3 Compositional diagnosis：很多“组合失败”并不是 low-level skill 不会，而是 instruction steering 不进去

**Diagnosing Compositional Generalization in Sequential Robot Tasks**  
https://arxiv.org/abs/2607.29687

该工作把 generalization gap 分成 marginal instruction shift、instruction-compositional shift 和 context-action shift，并发现一个非常重要的结果：

> sparse training 下很多 OOD failure 并不是缺 low-level skill，而是 instruction steering / coverage structure 没有把已有 action capability 正确调出来。

甚至每个 task 只补一条 demonstration 就可以让一个 setting 的 OOD success 从 `0.4%` 提高到 `54.7%`。

这再次支持一个需要被单独区分的解释：

```text
capability absent
```

和

```text
capability present but global instruction cannot reliably steer it
```

不是一回事。

---

## 4. 为什么不是普通“hierarchical VLA 更好”论文

这题不问：

> hierarchy 是否比 flat VLA success 高？

这个已经做烂了。

也不问：

> 哪个 planner / memory / termination rule 更好？

`What Matters in Orchestrating Robot Policies` 已经做了非常系统的 component study。

我们真正问的是：

> **Hierarchy 为什么有效？**

而且要把一个通常被混在一起的效果拆开：

```text
better reasoning
vs
better interface to an existing low-level repertoire
vs
temporal reset / handoff
```

如果最后发现 oracle atomic decomposition 本身就拿回绝大多数 gain，那么 hierarchy 的主叙事会被改写：

> planner 的关键作用之一不是“发明动作”，而是把 broad goal 编译成 low-level policy 已经能执行的控制语言。

如果反过来发现 oracle decomposition 帮助很小，真正的 gain 只在 state-aware reasoning planner 下出现，也同样是一个干净结果：

> hierarchy 的优势确实来自 online task reasoning，而不是 semantic support matching。

所以这个问题不是只有正结果才有价值。

---

## 5. 和已有工作的 collision 到哪里

### RoboHiMan

已经分离 high-level planning error、low-level execution error，并证明 rule-based planner 很有效。

但它的目标是 hierarchy evaluation / compositional generalization，并没有因果拆分：

```text
atomic-language narrowing
vs
subtask boundary reset
vs
state-aware planning
```

谁解释了 rule-based hierarchy 的增益。

### What Matters in Orchestrating Robot Policies

这是最接近的现代工作。它系统研究 Hi-VLA 的组成件，并明确发现 low-level steerability 很重要。

但它解决的是：

> **how should a hierarchy be built?**

而本题解决的是：

> **why does decomposition itself help a fixed foundation policy?**

只要实验真正固定 low-level VLA、固定 task states、控制 planner intelligence 和 switching，就仍然是不同的机制问题。

### Diagnosing Compositional Generalization

已经说明 instruction coverage / steering 可以成为组合泛化瓶颈。

但它不是 hierarchical-policy mechanism paper，也没有问 planner 的主要价值是否就是把 abstract task 映射回 low-level instruction support。

### Classical hierarchical RL / options

老的 HRL 早已知道 manager 给 worker 的 subgoal 必须 reachable / compatible；因此不能声称“低层可达 subgoal 很重要”是新理论。

真正的现代问题在于：

> foundation VLA 的 low-level interface 是自然语言，而且已有大量 pretrained motor repertoire；VLM planner 是否在事实上扮演一个 **controller-aware semantic compiler**？

这是 foundation-policy setting 下的新机制角度。

---

## 6. 最干净的 prerequisite experiment

第一枪不需要训练新 planner，也不需要设计复杂 method。

选一个公开 low-level VLA，优先考虑：

- `π0.5` + RoboHiMan / HiMan-Bench；
- 或一个已有 atomic-skill 能力明确、可在 simulation 闭环运行的公开 VLA。

构造由 2–4 个已验证 atomic skills 组成的 compositional tasks。

同一个 low-level policy、同一批初始状态，比较：

### A. Flat global instruction

整段 episode 一直输入：

```text
完成完整任务 X
```

### B. Oracle atomic decomposition

不给任何 learned planner。

人工提供正确的：

```text
subgoal 1 -> subgoal 2 -> subgoal 3
```

并用 oracle / success-based switching。

这是最关键的 condition。

它直接问：

> **没有 planner intelligence，仅仅把目标重新写成 atomic subgoal，能拿回多少 hierarchy gain？**

### C. Learned VLM planner

标准 hierarchy。

用于估计真正 state-aware reasoning 相对 oracle decomposition 还能额外带来多少。

### D. Atomic wording support intervention

保持**同一个 subgoal semantics 和同一个 switching schedule**，只改低层指令形式：

- canonical / training-like atomic phrase；
- semantic paraphrase；
- 更抽象、global-like 的说法。

如果 hierarchy gain 明显随 low-level steerability / wording support 变化，而 planner correctness 不变，就支持 controller-support interpretation。

### E. Boundary-only control

保持 global task language，但使用与 B 相同的 subtask boundary / reset schedule。

这样可以把：

```text
language support narrowing
```

和

```text
temporal reset
```

拆开。

---

## 7. 第一枪应该测什么

行为层面优先，不要一开始上 SAE。

至少测：

1. **full-task closed-loop success**；
2. **per-stage success**；
3. **instruction steerability**：同一 state 下切换 atomic subgoal 时 action 是否系统改变；
4. **planner correctness / subgoal correctness**；
5. **handoff / termination failures**；
6. flat → oracle decomposition → learned planner 的 gain decomposition。

一个简单但非常有解释力的量是：

```text
Fraction of hierarchy gain recovered by oracle decomposition

= (S_oracle - S_flat) / (S_hierarchy - S_flat)
```

它不是最终论文唯一 metric，但可以作为第一枪判断机制有没有量级。

---

## 8. 强结果长什么样

### 结果 A：大多数 hierarchy gain 不需要 planner intelligence

例如：

```text
flat              20%
oracle atomic      50%
learned planner    55%
```

那就得到一个很大的机制结论：

> **Hierarchical robot foundation policies work in large part because decomposition keeps the low-level policy inside an executable, steerable subgoal regime.**

这会直接改变 hierarchy 的设计重点：

- planner 不只优化 semantic correctness；
- 还应优化 controller compatibility；
- low-level post-training 不应牺牲 language steerability；
- subgoal vocabulary / interface 应与 low-level action data 一起设计。

### 结果 B：oracle decomposition 有帮助，但 temporal reset 才是大头

那结论变成：

> hierarchy 主要通过重新闭环 / 及时终止 action segments 减少 drift，而不是通过更聪明的 reasoning。

这同样很有价值，并自然连接 action chunking / receding-horizon control。

### 结果 C：只有 state-aware planner 明显提升

如果：

```text
flat ≈ oracle atomic << learned planner
```

那么 support-matching hypothesis 被否掉，但得到反向机制结果：

> 当前 foundation VLA 的 long-horizon bottleneck 真的是 online sequencing / reasoning，而不是简单 instruction support mismatch。

只要跨模型成立，这也值得写。

---

## 9. 方法空间

如果 controller-support matching 是主机制，方法空间非常自然：

- **controller-aware subgoal compiler**；
- 用 low-level VLA 的 steerability / executability 对 planner candidate subgoals 打分；
- low-level post-training 时显式保持 subgoal controllability；
- 自动学习对低层 policy 最“可执行”的 language interface；
- planner 生成语义正确 subgoal 后，再做 controller-compatible rewriting；
- 将 decomposition 粒度自适应到 low-level repertoire 的边界。

如果 temporal handoff 是主机制：

- learned success / progress termination；
- interruptible low-level execution；
- uncertainty-aware chunk reset。

如果 reasoning 才是主机制：

- 更强 state-aware planner / world-model planning；
- explicit failure reasoning / replanning。

因此这个题成立以后不会没有 method 口。

---

## 10. 顶会叙事尺度

这个问题本身不依赖某个 benchmark 名字才能成立：

> **Why Does Task Decomposition Help Robot Foundation Policies?**

它和：

- `Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?`
- `What Matters in Orchestrating Robot Policies?`
- mechanism / diagnosis 型 ICLR、ICML、NeurIPS、RSS 论文

处于相同的叙事尺度。

如果成立，AI venue 可以强调：hierarchical foundation policy 的 computation / interface / generalization；机器人 venue 可以强调 long-horizon closed-loop manipulation 与 planner-controller interface。

因此目标可以自然对齐：

- ICLR / ICML / NeurIPS；
- RSS / ICRA / IROS。

---

## 11. Kill standard

以下情况直接移出 shortlist：

1. 固定 low-level VLA 后，oracle atomic decomposition 在多个 compositional tasks 上几乎没有 gain；
2. 控制 switching / termination 后，atomic wording / steerability 与 hierarchy gain 没有稳定关系；
3. 找到已有工作已经在同一固定 VLA 上系统拆分 `planning intelligence vs atomic support matching vs temporal reset`；
4. 结果只能在一个 benchmark 的 annotation artifact 上成立；
5. 最终标题必须缩成某个 planner prompt / 某个 skill list 才显得新。

同时不要求“零 collision”。真正要求的是：

> **相对已有 hierarchy literature，机制叙事和 controlled experiment 都要有清楚的新角度。**

---

## 12. 当前判断

**保留，继续审计，不注册。**

它目前最强的地方不是“hierarchy 很重要”，而是三个已有结果之间形成了一个还没有被干净拆开的 tension：

```text
rule-based decomposition already yields a large gain
+
low-level steerability determines hierarchy success
+
many compositional failures occur despite existing low-level skills
```

这使下面的问题变得非常自然：

> **How much of hierarchical VLA success is actually planning, and how much is keeping the controller in a regime it already knows how to execute?**

第一枪也足够干净：先不用训练 planner，只做 flat vs oracle atomic decomposition vs learned planner，并控制 wording 与 handoff。
