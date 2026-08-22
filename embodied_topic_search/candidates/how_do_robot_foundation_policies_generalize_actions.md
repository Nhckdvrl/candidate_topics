# How Do Robot Foundation Policies Generalize Actions?

> Status: **PROVISIONAL SEARCH CANDIDATE — not a registered Topic.**
>
> 这个文件只把 Round 6 中审出来、值得继续 collision 的问题单独摘出来。它不进入根目录正式 Topic，也不因为进入这里就默认成立。

---

## 1. 问题

> **How do robot foundation policies generalize actions?**

更机制化地说：

> **When a robot foundation policy succeeds outside its training distribution, is it retrieving a known motor pattern, interpolating among familiar trajectories, composing known motor segments, or truly extrapolating into new motor behavior?**

这个题不问 OOD success 本身，而问：

> **OOD success 是通过什么 action-side computation 得到的？**

---

## 2. 这个题从哪里挑出来的

来源：

`topic_search_logs/2026-08-22_vla_wam_mechanism_search_round6.md`

Round 6 没有从一个抽象直觉起题，而是从一个已被 ICLR 2026 系统证明的反直觉现象出发：

### Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative

ICLR 2026。

该工作在同一 task / data 下比较三类 policy：

- Diffusion Policy；
- ACT；
- pretrained GR00T-N1.5。

得到非常不同的 action-generation behavior：

```text
Diffusion Policy
≈ strong action retrieval / memorization

ACT
≈ interpolation

GR00T-N1.5
≈ interpolation + visual OOD robustness,
  but no reliable systematic extrapolation
```

这给出一个非常好的起点：

> 机器人 policy 的高 success 并不自动意味着它学会了新的 motor behavior。

与此同时，2026 foundation VLAs 又在报告：

- zero-shot instruction following；
- cross-embodiment transfer；
- reactive recovery；
- novel skill compositions；
- demo 外 emergent strategies。

因此真正值得问的不是“VLA 会不会泛化”，而是：

> **这些所谓泛化在 action space 中究竟属于哪一种机制？**

---

## 3. 为什么问题尺度合格

这个题可以独立成为一篇顶会论文的问题，而不依赖某个 benchmark / statistic 才能理解。

它和：

- `Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?`
- `Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative`
- `What Matters for Batch Online Reinforcement Learning in Robotics?`

属于相同的问题尺度：

> 对一个已经非常成功、被领域普遍采用的 robot-policy paradigm，问它真正通过什么 computation 得到能力。

第一枪可以窄，但题目本身不能缩成：

> “GR00T 在某个 cup task 上的 trajectory similarity”。

如果最后只能做到这一层，就应该砍。

---

## 4. 四种需要区分的 action generalization

### Retrieval

当前 observation 映射到训练集中相似 observation，然后输出对应训练 action segment。

高 visual robustness 可以与几乎没有 motor generalization 同时存在。

### Interpolation

输出位于已知 trajectories / action modes 之间。

这可以处理连续几何变化，但仍没有真正离开 training action support。

### Composition

模型把 training repertoire 中分别存在的 motor segments / skills 重新组合成新的 sequence。

这已经比普通 interpolation 强，但组成单元仍来自已知 repertoire。

### Extrapolation / Synthesis

关键 motor behavior 超出训练 trajectory / primitive support，不能被简单解释成近邻 retrieval 或局部 composition。

这是很多 `emergent capability` 叙事真正隐含、但常常没有直接证明的强版本。

---

## 5. 为什么现在特别 relevant

### ICLR 2026 已经证明：性能可以来自 memorization

Diffusion Policy 在 sparse data 下可以表现得像 action lookup table，而且这种 memorization 本身还能给出不错的 OOD robustness。

所以：

```text
robustness != motor generalization
```

### GR00T 提示 pretraining 可能改变机制，但证据还不闭合

同一 ICLR 2026 audit 中，GR00T-N1.5 不再像 task-specific diffusion policy 那样强 retrieval，表现出更多 interpolation 和 visual OOD robustness。

但在 OOD extrapolation 中，它会出现 average-like prediction，而不是稳定产生系统性外推动作。

这说明：

> pretraining 似乎改变了 action behavior，但尚不能直接推出 genuine action synthesis。

### 2026 foundation models 又开始声称更强行为泛化

Qwen-RobotManip、π0.7、Wall-OSS 等开始把大规模 pretraining 与 zero-shot / recovery / composition / cross-embodiment capability 联系起来。

因此真正缺的是把这些 capability claims 映射到 action-side mechanism。

---

## 6. 已有 collision

这个题不能声称以下内容是新：

### Action memorization

ICLR 2026 已经直接做掉：Diffusion Policy 在 sparse demonstrations 下高度 memorization / retrieval。

### ACT interpolation

同一工作已经展示 ACT 更接近 trajectory interpolation。

### Generic compositional generalization

AC-VLA、ACT-VLA、SkillNet、Diagnosing Compositional Generalization 等已经正面研究 unseen task / skill composition。

所以我们不能只问：

> VLA 能不能组合技能？

### Generic scaling / memorization

已有 robot-policy scaling / SAE / memorization work 开始分析小数据 memorization 与大数据 representation generality。

所以我们也不能只做：

> 大模型 memorization 少一点。

---

## 7. 当前真正可能的新东西

当前尚未看到一篇工作系统建立：

> **robot foundation policy 的 action-generalization mechanism 随 pretraining scale / diversity / post-training 发生怎样的 qualitative change。**

希望回答的不是一个模型，而是一条机制谱：

```text
retrieval
  -> interpolation
  -> composition
  -> extrapolation / synthesis ?
```

以及不同类型的 OOD success 分别落在哪一层。

一个很重要的可能结果甚至是：

> 大型 foundation VLA 的“emergent generalization”主要来自更大的 behavior repertoire + 更强 perception invariance，而不是 motor extrapolation。

这并不是负结果；如果证据干净，这反而会重写我们对 robot-data scaling 的解释。

---

## 8. 最便宜的 prerequisite

不需要先训练大 VLA。

第一枪应该复用公开 checkpoint，在一个可以明确构造 action-support geometry 的 task 上比较：

- task-specific Diffusion Policy；
- ACT；
- 一个公开 pretrained generalist VLA；
- 如资源允许，再加另一个不同 scale / pretraining 的 VLA。

训练 / evaluation 状态需要区分：

```text
In-support
Interpolation
Compositional holdout
Extrapolation
```

不能只看 success。

至少同时测：

1. nearest-training-trajectory similarity；
2. local interpolation position；
3. output 能否分解成已有 motor segments；
4. true support escape；
5. closed-loop success。

核心 prerequisite 是：

> pretrained VLA 的 successful OOD behavior 是否真的出现和 task-specific policy 不同的 action-side mechanism。

如果没有 separation，题直接降级。

---

## 9. 强结果长什么样

### 结果 A：pretraining 改变了 computation

随着 pretraining 规模 / diversity 增加：

```text
retrieval-dominated
   -> interpolation / composition-dominated
```

这说明 foundation scaling 不只是扩大 lookup table，而改变了 behavior generation mechanism。

### 结果 B：大模型仍主要依赖 repertoire support

即使 OOD success 高，输出关键 motor segment 仍高度贴近训练 repertoire。

则结论是：

> foundation-policy generalization 主要来自 observation generalization + repertoire coverage，而不是 action extrapolation。

这个结果对 data scaling 同样重要。

### 结果 C：不同 OOD axis 对应不同机制

例如：

- visual shift：same-action retrieval；
- geometry shift：interpolation；
- task composition：segment recombination；
- genuinely new dynamics：仍失败。

这会给 VLA generalization 一个比单一 success rate 更有解释力的 decomposition。

---

## 10. 方法空间

如果 retrieval 是主机制：

- retrieval-aware data acquisition；
- behavior repertoire coverage；
- explicit support / OOD detection。

如果 interpolation 是主机制：

- action-manifold densification；
- structured data placement；
- geometry-aware augmentation。

如果 composition 是主要 frontier：

- compositional motor representation；
- primitive factorization；
- skill-support-aware data generation。

如果 extrapolation 是真正缺口：

- RL / interaction 应该重点扩展 motor support，而不是继续重复已知 demonstration modes。

---

## 11. Kill standard

以下任一项成立就移出 shortlist：

1. 找到已有工作已经在多个 large pretrained VLA 上系统区分 retrieval / interpolation / composition / extrapolation；
2. `new behavior` 只能靠任意 latent metric 定义，无法 behavior-level 识别；
3. pretrained VLA 和 task-specific policies 没有稳定 mechanism separation；
4. 所谓 improvement 全部来自 visual robustness，action-side 没变化；
5. 最终必须把标题压成某一 benchmark 的 trajectory similarity。

最后一条继续沿用 Round 5 标准：

> **如果必须越压越窄才能活，就不做。**

---

## 12. 当前判断

**保留，继续审计，不注册。**

它当前最大的优点是问题非常自然：

> **How Do Robot Foundation Policies Generalize Actions?**

并且它直接长在一个已经被顶会证明的异常上，而不是靠猜。

最大风险是：Mac Schwager 组的 memorization / generalization 研究线和 2026 compositional-VLA 赛道都很活跃，collision 会继续上升。

所以后续需要继续核：是否已有工作已经把 pretrained VLA 的行为生成机制从 retrieval 推到 composition / extrapolation 层级做了系统审计。