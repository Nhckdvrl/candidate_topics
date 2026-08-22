# 2026-08-22：VLA / WAM 机制选题搜索日志（第六轮）

> 状态：**继续搜索。正式 Topic 仍不新增。**
>
> Round 5 重新校准了问题尺度：题目必须像 ICLR / ICML / NeurIPS / RSS / CoRL / ICRA / IROS 的自然 research question，不能靠不断缩成 dataset/operator/statistic 来躲 collision。Round 6 因此不围绕已有候选 A 继续细化，而是重新广泛搜索其他“大问题”。
>
> 本轮材料继续混合：2025–2026 顶会论文、2026 最新 technical reports、开源整模型、官方 blog、项目页、practitioner ablation、以及相邻 imitation / RL / generative-policy 文献。

---

# 0. 本轮目标

不是再找一个“看起来新”的小机制，而是寻找满足下面四条的大问题：

1. **问题本身自然。** 不读实验细节也能理解为什么重要；
2. **有真实 anomaly / tension。** 不是靠我们先猜它存在；
3. **第一枪可以便宜。** screening 不需要一开始训练大 VLA；
4. **机制一旦成立，自然留下方法空间。**

当前保留的 A：

> **What Does History Actually Model in Robot Policies?**

本轮不继续压窄 A，而是寻找独立的第二问题。

---

# 1. 砍：Do Robot Policies Actually Need Generative Action Heads?

这一条来自 practitioner controlled comparison：固定 SmolVLA backbone、action expert、data 与训练 recipe，只替换 flow matching / L1 regression / DDPM action objective，在 LIBERO-Spatial 上 L1 regression 可以接近 flow，而 diffusion 明显更差。

这很容易形成一个漂亮标题：

> **Do Robot Policies Actually Need Generative Action Heads?**

但 collision 太近。

现有 diffusion-policy / flow-policy 文献已经大量研究：

- multimodal action distributions；
- diffusion vs regression；
- flow matching 的效率与稳定性；
- action memorization；
- generative policy 的 test-time compute。

更重要的是，如果为了保持 novelty，最后必须把题压成：

> “在近 unimodal task 上 regression 是否足够？”

那已经是一个 architecture ablation，而不是我们要的大问题。

**结论：砍。**

---

# 2. 砍：What Is the Multimodality in Robot Demonstrations?

这个题一开始很有吸引力：diffusion / flow policy 的常见卖点之一是能够表示 multimodal action distribution，但 demo 中的 mode 到底来自：

- 同一 task 的多种有效解；
- latent task state；
- 不同 demonstrator / strategy；
- timing / teleop artifacts；
- controller / embodiment differences？

如果大量 multimodality 其实是 producer heterogeneity，那么“更强地保留所有 modes”未必总是正确。

但 collision audit 后不再保留。

已有工作已经直接连接：

```text
expert diversity
    -> velocity / trajectory multimodality
    -> imitation confounding
```

并且 multi-human imitation literature 早已证明不同 human demonstrators 会提供互不兼容但都合理的策略。

继续往下只能缩成：

> “哪些 modes 是 task-relevant，哪些是 nuisance？”

这又开始靠压窄活。

**结论：砍。**

---

# 3. 砍：Does Reasoning Actually Control Robot Actions?

2026 reasoning-VLA 很多，于是一个很自然的机制问题是：

> 模型生成的 CoT / subtask / latent plan 是否真的因果控制后续 action，还是只是与 action 同时生成的 rationalization？

这个题尺度很好，但 exact collision 非常直接：

**Do Vision-Language-Action Models Mean What They Say? On the Role of Faithfulness in Embodied Reasoning**（2026-07）已经正面区分 functional reasoning 与 faithful reasoning，并讨论 reasoning-action faithfulness。

因此不能再拿。

**结论：正面 collision，砍。**

---

# 4. 砍：What Transfers from Human Videos to Robot Policies?

human / egocentric video 正在成为 VLA pretraining 的重要数据源，因此曾考虑：

> human video 到底 transfer 了 semantics、motion prior、task structure，还是 embodiment-agnostic manipulation representation？

但 2025–2026 已经有非常直接的工作：

- `Emergence of Human to Robot Transfer in VLA Models`；
- EgoScale / human-motion pretraining；
- ACE-Ego-0；
- Qwen-RobotManip H2R synthesis。

这些工作已经把关键故事推到 embodiment-agnostic representation / motor prior / aligned human-to-robot action transfer。

继续做只会统一已有结论。

**结论：砍。**

---

# 5. 砍：What Does RL Actually Change in a VLA?

这个标题非常自然，而且 Round 3 已经留下 `RL emergent behavior` 搜索线。

真实现象包括：

- NeurIPS 2025 `What Can RL Bring to VLA Generalization?`：RL 对 execution OOD 的提升远大于 vision OOD；
- SimpleVLA-RL：出现 demonstration 中没有的 `pushcut`；
- RL 又严重依赖已有 SFT foothold；
- continual-VLA-RL / SARL 等已经开始分析 RL 与 pretrained skill repertoire。

问题是：这个大问题本身已经被 NeurIPS 2025 的标题几乎原样占掉，后续 2026 又有大量 representation / continual / emergent strategy work。

如果再做，只能缩成：

> “RL 改了哪些层 / 哪些 action modes？”

这又开始变成局部机制补充。

**结论：不作为新的 shortlist candidate。**

---

# 6. 砍：Does Better Context Require More Action-Generation Compute?

这个问题来自 Qwen-RobotManip 一个非常反常的技术报告结果。

Qwen-RobotManip 加入 execution history 后，context 本来应该更充分，但 action distribution 反而更复杂：

- 无 history 时，4 denoising steps 已经稳定；
- 加 history 后，同样 4 steps 出现 jitter，收益几乎被抵消；
- 提高到 10 steps 后，history 才真正释放完整收益；
- 20 steps 不再明显增加。

这很自然地引出：

> **Does richer conditioning require more action-generation compute?**

但 collision 也已经很近：

- ELASTIC：state/task/policy-dependent test-time compute；
- ProbeFlow：动态调 ODE / denoising steps；
- 一系列 adaptive generative-control compute work。

Qwen 的 anomaly 是很好的 supporting evidence，但不足以再形成独立题。

**结论：砍。**

---

# 7. 砍：Do Robot Policies Know When They Are Done?

真实部署里，很多 benchmark 是 evaluator 在成功后替 policy 终止 episode；policy 自己可能并不知道什么时候该停止。

这可以形成：

> **Do Robot Policies Know When They Are Done?**

但 2026 已经有 ProgressVLA、SeqVLA、PALM 等直接把 task progress / completion / termination 建模作为核心问题。

**结论：砍。**

---

# 8. 降级：Does Fine-Tuning Collapse the Behavioral Repertoire of Robot Foundation Policies?

这一条一度很强。

问题是：

> foundation VLA 预训练可能拥有很宽的行为 repertoire；下游少量单一风格 demo 的 SFT 是否在提高目标 task success 的同时，把替代策略、恢复行为和 OOD action support 压缩成一条窄路径？

这个版本和普通 `VLM catastrophic forgetting` 不同：它问的是**robot behavior support 的收缩**。

真实旁证很多：

- OpenVLA 官方早期就观察到，在某些窄单任务上，从 scratch 的 Diffusion Policy 可以超过 fine-tuned generalist VLA；
- PLD 发现，远离 base-policy state distribution 的 narrow expert data 更容易损害原模型 generalizability，而围绕 base policy failure states 的 recovery data 更安全；
- Qwen-RobotManip 明确报告 benchmark-specific SFT 会出现 `VLA-to-VA degradation`：task success 提升但 language conditioning 下降，模型更像 visual-action pattern matcher；
- Qwen 进一步使用 VL / pretraining VLA co-training 缓解 domain overfitting。

但 exact collision 后不升格：Mac Schwager 组目前已经把 robot-policy memorization / generalization 当成持续研究线；2026 mechanistic / SAE work 也在讨论小数据 SFT 放大 episode-specific memorization、数据规模增加 generalizable motion primitives。

因此如果我们继续，容易落到他们当前主线旁边。

**当前状态：有真实 tension，但 collision 高，暂不摘成候选。**

---

# 9. 砍：Where Does a VLA Learn a New Skill?

一度形成：

> **Where does a VLA learn a new skill — VLM backbone, action expert, interface layers, or their alignment?**

真实证据也很好：

- 很多 VLA 可以删掉大量 backbone layers 后再 fine-tune，仍保持很强 task performance；
- ActionX / action-expert pretraining 显示 action expert prior 可以决定下游 learnability；
- continual-VLA 工作发现 VLM 与 action head 在新 skill adaptation 中变化模式不同。

但 exact collision 后不留：

- continual-VLA 已经做 component swapping；
- Robotic Steering 已经定位 task-specific attention heads；
- task-vector / skill-edit audit 开始直接分析 skill locality。

继续只能压成“训练到第几个 checkpoint 时 skill 写进去”，不符合 Round 5 的尺度标准。

**结论：砍。**

---

# 10. 砍：How Stable Is VLA Fine-Tuning?

又找到一个漂亮的真实异常：同一 VLA / data / code，只换 seed，13 次里有 1 次 success 从约 91–94% 静默跌到约 65%。

这可以形成：

> **How Stable Is VLA Fine-Tuning?**

但 2026 已经有工作把它明确命名为 `seed lottery` 并分析 output collapse。

**结论：砍。**

---

# 11. 砍：Why Do VLAs Fail Compositional Generalization?

这个大问题当然重要，但 2026 现在已经极度拥挤：

- `Diagnosing Compositional Generalization in Sequential Robot Tasks`；
- AC-VLA；
- ACT-VLA；
- SkillNet（ICML 2026）；
- InternVLA-A1.5；
- FineVLA；
- π0.7 的 steerable / compositional generalization。

其中 `Diagnosing Compositional Generalization` 已经把 gap 拆成 marginal instruction shift、instruction-compositional shift、context-action shift，并指出很多失败不是缺 low-level skill，而是 instruction steering / coverage structure。

**结论：赛道已经形成，砍。**

---

# 12. 新的强备选：How Do Robot Foundation Policies Generalize Actions?

这一条来自 ICLR 2026 一个已经证明的反直觉机制结果：

## Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative

ICLR 2026。

这篇把三类 policy 放在**同一 task / 同一 data**下比较，发现：

### Diffusion Policy

在小数据 regime 下，本质上很像：

```text
current image
  -> nearest training observation
  -> recall associated action chunk
```

即使给高度 OOD 的图像，也会倾向输出训练 action chunk。

所以它的 robustness 很大程度可以来自 **reactive memorization / retrieval**，而不是 action generalization。

### ACT

更接近 action interpolation：

```text
between training trajectories
```

但 OOD robustness 更差。

### GR00T-N1.5

最关键：GR00T 同时表现出 interpolation 和 OOD visual robustness。

但在真正的 OOD extrapolation 下，它并没有显示出可靠系统性 action extrapolation；论文报告它会退化成 average-like predictions。

因此我们已经有一个非常干净的三分结构：

```text
retrieval
interpolation
?
```

而现在 2026 foundation VLAs 又开始报告：

- unseen instruction following；
- cross-embodiment transfer；
- reactive recovery；
- novel skill composition；
- zero-shot behavior；
- demonstration 外的新策略。

这就留下一个比普通 benchmark generalization 更基本的问题：

# **How Do Robot Foundation Policies Generalize Actions?**

或者更机制化地说：

> **When a large robot foundation policy succeeds outside its training distribution, is it retrieving a known motor pattern, interpolating among known behaviors, composing familiar primitives, or truly extrapolating into new motor behavior?**

---

# 13. 为什么这个问题不是普通“泛化 benchmark”

它不问：

> OOD success 是多少？

而问：

> **OOD success 是通过什么 action-side computation 得到的？**

例如同一个成功 episode，可能有完全不同的机制：

### Retrieval

测试 observation 变了，但 model 找到训练中相似 scene / state，然后 replay 训练 action chunk。

### Interpolation

model 输出落在两条已知 action trajectories 之间。

### Composition

model 把训练中分别出现过的 sub-trajectories / motor primitives 重新拼接成新的 sequence。

### Extrapolation / synthesis

输出的关键 motor segment 在 training repertoire 中没有近邻，也不能简单分解成已有 segment 的局部组合。

这些机制对我们如何理解 foundation policy、如何扩数据、如何做 RL、如何判断 zero-shot capability，含义完全不同。

---

# 14. 为什么现在特别值得问

2026 的模型正在出现一个明显张力：

### 一边：大量工作强调“emergent / compositional / zero-shot behavior”

- π0.7：报告 unseen task compositions 与 steerability；
- Qwen-RobotManip：38,100h aligned pretraining 后报告 zero-shot instruction following、reactive recovery、cross-embodiment transfer；
- Wall-OSS-0.5：强调 pretraining checkpoint 本身已经有 executable zero-shot behavior；
- SimpleVLA-RL：RL 中出现 demo 外 `pushcut`。

### 另一边：mechanistic audit 对 action generalization 非常谨慎

ICLR 2026 diffusion audit 显示：

- 强 performance 不等于 action synthesis；
- memorization 可以非常强、甚至对 OOD 有 robustness；
- pretrained GR00T 比小模型更进一步，但仍没有展示可靠 extrapolation。

因此现在的问题不是：

> foundation VLA 会不会泛化？

而是：

> **它的泛化到底发生在 observation side、action side，还是只是 repertoire coverage 变大？**

这是一条 foundation-level mechanism question。

---

# 15. collision audit：已有工作做了什么，还没做什么

## 已经被 ICLR 2026 做掉的

不能声称：

- Diffusion Policy 会 action memorization；
- ACT 更像 interpolation；
- GR00T 比小模型有更强 visual OOD robustness；
- simple lookup table 在小数据 regime 可以很强。

这些都已经是明确结果。

## 已经被 compositional-generalization work 做掉的

不能只证明：

> VLA 对 unseen skill combinations 很差 / 可以通过 subtask decomposition 变好。

AC-VLA、ACT-VLA、SkillNet、Diagnosing Compositional Generalization 已经很深。

## 当前仍可能存在的空白

目前还没有看到一篇工作系统回答：

> **large pretrained VLA 的 action-side generalization mechanism 会随 pretraining scale / data diversity / post-training 改变吗？**

特别是：

```text
small task policy
   retrieval / memorization
        ↓
pretrained generalist VLA
   interpolation + repertoire reuse
        ↓
very large aligned foundation policy
   ? composition
   ? genuine extrapolation
```

是否真的存在这种 qualitative transition，目前并没有被建立。

---

# 16. 第一枪可以很便宜，而且不需要先训练大模型

这个题的 screening 不需要一开始训练 foundation model。

可以先复用公开模型 / checkpoint，在一个**可控 action-support 几何**里测试：

```text
training actions / trajectories
        |
        |--- held-out interpolation states
        |--- held-out compositional states
        |--- held-out extrapolation states
```

同时比较：

- task-specific Diffusion Policy；
- ACT；
- 一个公开 pretrained VLA（例如 GR00T / open π 系 / Wall-OSS 等可运行 checkpoint）；
- 如果成本允许，再加更大 pretrained checkpoint。

核心不是只算 success，而是同时量：

1. **nearest-training-trajectory similarity**；
2. **piecewise compositionality**：输出是否能由已有 motor segments 拼成；
3. **trajectory-space interpolation coefficient**；
4. **true support escape**：关键动作是否明显超出 training action support；
5. closed-loop success。

希望看到的不只是“大模型更成功”，而是：

> **成功机制本身发生了可识别变化。**

---

# 17. 这条题真正值得的结果是什么

### 强结果 A

随着 robot pretraining scale / diversity 增加：

```text
retrieval-dominated
    -> interpolation/composition-dominated
```

而不是简单“lookup table 变得更大”。

这说明 foundation policy 的 scaling 确实改变了 action computation。

### 强结果 B

即使大 VLA OOD success 很高，它的成功动作仍高度落在巨大 training repertoire 的局部支持中。

那结论同样非常重要：

> 所谓 emergent robot generalization 主要来自 repertoire coverage / perception invariance，而不是 motor extrapolation。

这会直接改变 robot data scaling 的解释。

### 强结果 C

不同 generalization axis 对应不同机制：

- visual OOD：same motor retrieval + better perception；
- geometric OOD：interpolation；
- task composition：segment composition；
- genuinely new dynamics：仍失败。

这会给 foundation-policy generalization 一个比 benchmark success 更有解释力的 decomposition。

---

# 18. 方法空间

如果机制成立，方法并不需要现在就定死，但后路很自然。

如果 foundation policy 主要是 retrieval：

- 显式 repertoire coverage；
- retrieval-aware data acquisition；
- OOD support detection；
- memory / retrieval augmentation。

如果主要是 interpolation：

- action-manifold densification；
- data placement / support design；
- structured augmentation。

如果 composition 能发生但 extrapolation 不行：

- compositional action representation；
- primitive factorization；
- targeted RL 去扩展 support，而不是重复已有 modes。

如果 pretraining scale 会产生 qualitative transition：

- 数据 / model scaling law 应从 success-rate scaling 升级到 **behavioral-mechanism scaling**。

所以机制结论无论落在哪一边，都有自然方法口。

---

# 19. Kill standard

以下情况直接把它移出 shortlist：

1. exact collision 找到已有工作已经在多个 pretrained VLA 上系统分解 retrieval / interpolation / composition / extrapolation；
2. 所谓 action synthesis 最后只能用任意 latent-space metric 定义，无法在 trajectory / behavior level 识别；
3. 第一枪发现 GR00T / pretrained VLA 与小 task-specific policy 没有任何 mechanism separation；
4. 只有视觉 OOD 差异，没有 action-side difference；
5. 为了 novelty 必须把标题缩成某个 benchmark 的 trajectory-similarity analysis。

尤其第 5 条：如果大问题站不住，就砍，不继续压窄。

---

# 20. Round 6 后的 provisional shortlist

## A. What Does History Actually Model in Robot Policies?

来源：Round 3 temporal anomaly → Round 4 producer-mixture audit → Round 5 scope recalibration。

核心：现代 history-aware policy 的 memory 到底在表示 task/world state，还是也在吸收 training-data behavior structure。

状态：**保留。**

## B. How Do Robot Foundation Policies Generalize Actions?

来源：Round 6，主要从 ICLR 2026 action-memorization audit 与 2026 foundation-VLA emergent-generalization claims 之间的张力长出来。

核心：foundation-policy OOD success 的 action-side mechanism 到底是 retrieval、interpolation、composition 还是 extrapolation / synthesis。

状态：**新强备选，继续 collision，不注册。**

---

# 21. 本轮最重要的教训

本轮砍掉了很多“标题很好听”的问题：generative action heads、reasoning faithfulness、human-video transfer、termination、skill localization、seed stability、compositional generalization。

这反而说明现在筛选标准在工作：

> **大标题只是必要条件，不是充分条件。**

真正值得留下的题必须同时满足：

```text
natural big question
+
real unexplained tension
+
not already closed by 2026 work
+
cheap prerequisite
+
mechanism -> method path
```

Round 6 当前只新增 B，不为了数量继续塞候选。
