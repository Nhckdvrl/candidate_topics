# 2026-08-23：VLA / WAM 机制选题搜索日志（第十一轮）

> 状态：**本轮只完成了启动与问题框架切换，尚未完成系统 collision audit，也没有新增 provisional candidate。今晚在这里收尾。**
>
> Round 10 之后继续遵守同一标准：不为了 shortlist 数量硬造题；不把“某个模型有个小 failure mode”直接包装成顶会题；只有问题尺度、相对 novelty、第一枪识别和正反结果价值都够，才摘成候选。

---

# 0. 为什么 Round 11 要换搜索区域

Round 8–10 已经大量扫过：

- hierarchy / task decomposition；
- whole-body motor equivalence；
- physical prompting；
- heterogeneous action padding / missingness；
- tool / affordance substitution；
- active perception / physical feasibility / abstention；
- failure recovery / termination / progress；
- camera-role shortcut；
- MoE action-expert specialization；
- hybrid action space 与 grasp / contact event failure；
- human-video pretraining transfer；
- WAM future use / controllability / representation。

其中很多方向不是“不重要”，而是 2026 已经形成非常拥挤的独立赛道；继续压 scope 很容易变成局部 architecture / benchmark paper。

因此 Round 11 开始主动从 **optimal feedback control、human motor control、task-space control** 这类更基础的问题结构里找可迁移到 robot foundation policy 的机制问题。

核心策略是：

> 不再只问 policy 最后输出了什么动作，而问 **它把什么偏差视作需要纠正的 error，以及这种纠正规律是否跨状态、对象和运动实现保持结构。**

---

# 1. 本轮启动的母问题：Robot Foundation Policies Learn Actions, or Feedback Laws?

一个自然的大问题是：

> **Do robot foundation policies merely map states to actions, or do they learn reusable task-space feedback correction laws?**

行为克隆最表面的统计对象是：

```text
observation -> action
```

但真正闭环控制更接近：

```text
current task error
    -> corrective response
```

同一个 corrective principle 理论上可以跨：

- object pose；
- initial trajectory；
- end-effector configuration；
- execution phase；
- 甚至不同 embodiment / motor realization。

例如：

- 抓取过程中目标向左偏，policy 是否在不同绝对位置下都产生结构一致的左向 correction？
- object 已经被 grasp 后发生一个 task-null 的手臂姿态扰动，policy 是否允许这种冗余变化存在，而不是机械拉回 demonstration trajectory？
- 对会破坏任务结果的 task-relevant perturbation，policy 是否强纠正；对不影响 task outcome 的 null-space perturbation，是否明显更宽容？

这与普通 robustness benchmark 不同。

robustness 只问：

> perturbation 后还能不能成功？

这里真正想问的是：

> **policy 的 corrective computation 是否对 task structure 有选择性。**

---

# 2. 为什么这个问题值得继续搜

这条线和 Round 9 的 D 有关联，但不是同一个题。

## D 问的是

> **Do Robot Foundation Policies Learn Motor Equivalence Classes?**

即：同一个 task effect 是否能用与 demonstration 不同的 motor realization 实现。

## Round 11 新线问的是

> **在 closed-loop execution 中，policy 到底纠正哪些 deviation、忽略哪些 deviation；这些 corrective responses 是否组成一个跨状态共享的 feedback law？**

一个是：

```text
solution set / equivalence class
```

一个是：

```text
feedback geometry / correction law
```

二者可以相互支持，但科学问题不同。

如果 foundation policy 真正学到了 task-space structure，那么理论上应出现：

```text
task-relevant error
    -> strong structured correction

motor-null / goal-equivalent variation
    -> weak correction / tolerance
```

反之，如果模型主要在复现 demonstration trajectory，则更可能表现为：

```text
any deviation from demonstrated state-action path
    -> pull back toward canonical trajectory
```

这会给 imitation vs task-level control 一个很干净的 behavioral distinction。

---

# 3. 一个潜在的一击实验结构

这部分目前只是 **search hypothesis / prerequisite design**，还不是冻结实验。

选一个已经能稳定完成简单 manipulation task 的公开 policy，在同一批 trajectory states 上做 paired perturbation。

## A. task-relevant perturbation

改变会直接影响 task outcome 的变量，例如：

- target object lateral displacement；
- grasp point shift；
- insertion-hole pose shift；
- door / faucet state change。

测 policy 的 corrective action：

```text
Delta a_relevant
```

## B. task-null / goal-equivalent perturbation

只改变不影响当前 task effect 的冗余变量，例如：

- redundant joint posture；
- whole-body null-space configuration；
- 某些不改变 EEF / contact geometry 的身体姿态变化。

测：

```text
Delta a_null
```

核心不是简单比较 action magnitude，而是看：

1. correction 是否朝真正 task-error reducing direction；
2. 同一种 task error 在不同绝对状态下是否诱导相似 correction；
3. task-null perturbation 是否被 policy 不必要地拉回 demonstration manifold；
4. 这种结构是否随 foundation pretraining / diversity 增强。

如果能得到：

```text
same task-space error
    -> consistent corrective field across states
```

那比“模型在 OOD perturbation 下还能成功”更接近一个 learned-control-computation 结论。

---

# 4. 可能出现的三类结果

## 结果 A：明显的 task-space feedback law

如果 policy 对 task-relevant error 做强、方向一致的 correction，而对 goal-equivalent / null variation 明显宽容，则说明：

> foundation policy 可能已经从 trajectory imitation 走向了某种 task-structured feedback control。

这会自然连接：

- motor equivalence；
- optimal feedback control；
- redundancy exploitation；
- foundation pretraining 的 qualitative transition。

## 结果 B：所有 deviation 都被拉回 demonstration trajectory

那结论也很重要：

> 高 success / robustness 不等于学到 task-level feedback law；policy 仍可能主要通过 trajectory restoration 工作。

这会重新解释 imitation scaling 的能力边界。

## 结果 C：只有局部 / task-specific correction，没有跨状态共享结构

那说明 broad thesis 目前不够强，应直接降级，而不是把它包装成某个 perturbation benchmark。

---

# 5. 当前 collision 状态：尚未完成，不做结论

今晚没有继续把这条线做完，因此这里必须明确：

- **还没有系统查完 optimal feedback control / task-space imitation / error-correcting policy / goal-equivalent manifold 在 2025–2026 foundation-VLA setting 下的直接 collision；**
- **还没有证明现代 VLA 存在一个足够强的现象 anomaly；**
- **还没有确认最合适的公开模型 / simulation task；**
- **因此不能摘成 E。**

下一次如果继续 Round 11，优先顺序应该是：

1. 查 foundation VLA / diffusion policy 中有没有直接做 task-relevant-vs-null perturbation 的机制论文；
2. 查 imitation learning / optimal feedback control / uncontrolled-manifold / task-space invariance 的经典和现代对应工作；
3. 找一个公开 policy + simulator，验证 task-null perturbation 是否真的可构造且不改变任务可行性；
4. 只有 broad novelty 和第一枪都站住，才考虑 provisional E。

---

# 6. 今晚收尾判断

Round 11 **没有完成到候选级别**，但完成了一个有价值的搜索方向切换：

```text
architecture / data-pipeline anomaly
            ->
feedback computation / task-space structure
```

当前没有新增候选。

active provisional shortlist 保持：

- **B — How Do Robot Foundation Policies Generalize Actions?**
- **C — Why Does Task Decomposition Help Robot Foundation Policies?**
- **D — Do Robot Foundation Policies Learn Motor Equivalence Classes?**

今晚到这里停止，不继续为了凑 Round 11 结果而草率搜索。
