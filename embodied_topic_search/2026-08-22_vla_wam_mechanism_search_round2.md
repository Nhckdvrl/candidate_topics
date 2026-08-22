# 2026-08-22：VLA / WAM 机制选题搜索日志（第二轮）

> 状态：**继续搜索，不注册新 Topic。**
>
> 这一轮的重点不是继续“想点子”，而是把第一轮产生的若干机制母题逐一做 collision audit。凡是 2026 已经从现象做到机制、甚至已经自然长出方法的方向，直接降级或砍掉。当前目标仍然是找到：**现象真实存在、机制没有闭合、结论之后自然留下 method knob、第一枪不依赖稀有 variation** 的题。

---

# 1. 先砍：contact / phase / replanning 这一大块已经过于拥挤

第一轮曾考虑一个很自然的现象：

> VLA 大部分时间在 free-space motion，而真正决定成功的往往是极短的 grasp/contact/insertion/phase-transition 时刻；普通 BC / flow loss 会不会在时间平均上把这些关键事件稀释掉？

这个问题本身非常合理，而且如果成立，后面也有清楚的方法口：event-balanced sampling、phase-aware loss、contact-aware chunking、phase-conditioned expert。

但 collision audit 后不再推进，原因不是问题差，而是**已有方法已经密集围绕同一结构展开**：

- **PAMAE: Phase-Aware-MoE Action Experts Towards Reliable Flow-Matching VLA**：直接按 manipulation phase 做 action-expert specialization。
- **DVAC: Denoising Tells When to Replan**：发现 free-space 阶段 denoising 更稳定、contact/precision 阶段更不稳定，并据此动态决定 execution horizon。
- **PACE: Phase-Aware Chunk Execution**：直接根据 motion phase 调整 chunk execution。
- **BCP: Continue or Replan?**：用 RL 学是否继续当前 chunk。
- 还有一系列 tactile/contact WAM（例如 VT-WAM、Dream-Tac）把 contact dynamics 本身做成预测目标。

因此即使我们证明“关键 contact event 在平均训练目标里被稀释”，审稿人也很容易认为这是给现有 phase-aware / contact-aware 方法补解释。

**结论：砍。**

---

# 2. 先砍：generic action-chunk / adaptive replanning / denoising commitment

曾考虑两个相邻问题：

1. action chunk 内部是不是一个真正的“计划”，远未来 action token 会不会因果影响第一个动作；
2. flow/diffusion VLA 在第几个 denoising step 后“改不了主意”。

它们都很有机制味，但 2026 邻近工作已经太密：

- Coarse-to-Control / ACoT-VLA：直接把 action-space intermediate structure 当 planning medium；
- AutoHorizon：从 action/self-attention 结构预测可执行 horizon；
- DVAC：利用 denoising variance 做 replanning；
- SnapFlow：直接研究多步 flow 到 one-step generation 的蒸馏；
- StructRL：刚在 2026-08-15 报告 flow chain 内部的 **Structured Noise Dilution**，并针对这一计算机制重新设计 RL stochasticity；
- Selected Diffusion Noise / UniSteer / FRS 等工作也已经把 noise space 当可控 action knob。

这意味着“denoising 中不同阶段承担不同角色”已经快速从观察走向方法。

只有出现比现有工作更强、更加基本的新命题时才值得回来，例如：

> 外部 observation 对 action mode 的因果控制力只存在于某个 flow-time 窗口，过了窗口以后即使重新注入新 perception 也无法改变高层 mode。

但目前没有独立文献证据表明这个具体现象存在，所以不能再次犯 Topic 09 的错：先觉得“应该有”，然后搭昂贵 harness 去赌。

**结论：暂砍，不注册。**

---

# 3. 降级：为什么 RL post-training 比 SFT 更不容易忘

这一条最初很吸引人，因为存在一个漂亮的真实矛盾：

- VLA supervised fine-tuning 会明显损坏 pretrained/generalization ability；
- 2026 continual-VLA-RL 工作却报告 sequential on-policy RL + LoRA 可以学新任务而几乎不 catastrophic forget。

如果机制没做过，那么很自然可以问：

> 为什么 reward-driven policy update 能比 imitation update 更局部地改机器人行为？

而且后面显然有 method 空间：设计 selective post-training update，既提高新任务能力，又保护 foundation capability。

但仔细读 **Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning** 后，这条不能作为新主问题。

该工作已经不只是报告现象，它明确分析了三个因素：

1. **on-policy RL 本身形成 implicit regularization**：更新集中在当前 policy support 上，不像 SFT 那样强迫模型拟合任意 demonstration distribution；
2. **大 pretrained model 的高维参数空间让新任务梯度与旧知识敏感方向更接近正交**；
3. **LoRA 限制 update subspace**，进一步降低旧能力破坏。

论文还做了 SFT / model size / LoRA ablation，并用 pretraining Fisher energy 衡量新任务梯度是否落在旧知识敏感方向。

与此同时，TEMPO 等 2026 工作已经继续沿“semantic/action decoupled RL post-training”发展方法。

所以 generic 问题“RL 为什么比 SFT 不忘”已经有人从现象做到机制和方法入口。

**结论：降级。** 只有找到 VLA 特有、现有 Fisher/implicit-KL 解释覆盖不到的内部机制才可能回来。

---

# 4. 砍：WAM 的 wrist-camera ego-motion / scene persistence

曾想到一个很 embodied-specific 的可能机制：

> wrist camera 的未来像素变化大量由 robot 自己的运动造成，WAM 会不会主要学会预测 camera ego-motion，而不是 object/world dynamics？

如果成立，后面可以做 ego-motion factorization / camera-stabilized residual world modeling，方法口很好。

但是 2026 collision 已经非常直接。

## Mem-World

**Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation** 明确指出：

- rapid wrist-camera motion；
- end-effector occlusion；
- current observation 不足；
- world model 会忘记/幻觉 scene detail。

它直接用 wrist-view-centered 4D surfel memory + future-action-conditioned geometry retrieval 处理这个问题。

## DECOWAM

更关键的是 2026-08-20（昨天）刚出的：

**DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation**

它已经明确把 **camera ego-motion、base action、arm action** 的混合视为现有 WAM 的问题，并通过 dedicated conditional interfaces / factorized latents 进行解耦。

这几乎正面撞掉了我们想问的 mechanism。

**结论：砍。**

---

# 5. 重要 collision：VLA mechanistic interpretability 已经开始真正“做机制”

这一轮发现一篇必须加入以后 collision checklist 的工作：

## Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models

ICLR 2026 Multimodal Intelligence Oral。

它不是普通 linear probe，而是跨 6 个 VLA、42 万+ rollout、520+ SAE checkpoint 做 activation injection。

最关键结果之一：

> 把 task A 的 action-expert activation 注入 task B scene，π0.5 在 99.6% episode 中执行 task A 的 motor trajectory；X-VLA 达到 99.8%。

更惊人的是，注入的 motor program **绑定到 absolute workspace coordinates，而不是当前可见 scene**。这直接给 concurrent object-pose perturbation brittleness 一个 mechanistic explanation。

它还发现：

- language 可以被编码，但在 vision 已唯一决定 goal 时 behavior 上被忽略；
- VLM / action-expert pathway specialization；
- 不同 architecture 的 SAE pooling preference 不一样。

这篇非常符合我们想做的“现象 → causal mechanism”范式，也说明以后任何 VLA representation 题，如果只做 probing/SAE/attribution，很难有竞争力。

### 对我们筛题的直接影响

以后 mechanism 题至少要问：

> **如果我改变这个内部对象，action 是否发生方向明确、可解释的 causal change？**

单纯“这个 feature 能 decode 某东西”不够。

同时，absolute-coordinate motor-program binding 这条本身也不再作为候选：geometry-aware / camera-space / embodiment-canonicalized VLA 已经有很多邻近方法，直接追它太拥挤。

---

# 6. 继续观察但高风险：cross-embodiment control frequency / physical action semantics

第一轮提出：

> 不同 robot dataset 的 action 即使都被写成相同数值空间，它们的物理意义可能不同，尤其 control frequency / controller / delta-vs-velocity 语义不同。

这一轮找到更多事实支持，但同时也发现 collision 比最初想象的高。

## 6.1 真实事实

### OpenVLA / OXE 类 pipeline

action 常按 dataset-specific 统计量归一化到统一数值区间，再在部署时用对应 metadata unnormalize。

因此：

```text
normalized action = 0.5
```

并不是一个独立于 dataset/robot 的物理动作。

### Same Weights, Different Robot (2026)

这篇甚至已经把 action normalization metadata 正式定义成 executable policy specification 的一部分。

它证明：同一个 checkpoint、同一个 normalized output，只要换错 sibling dataset 的 unnormalizer，就可以让 LIBERO replay success 从接近满分直接跌到接近 0。

这很好地说明：**action 的物理语义不在 weights 里闭合，而部分藏在外部 metadata/controller 里。**

### RACE / TempoVLA / Trajectory Standardization

这些工作又从不同侧面说明 temporal semantics 也不是小问题：

- RACE：简单提高执行速度会改变 underlying transition dynamics；
- TempoVLA：VLA 往往继承 demonstration 的单一速度，需要显式 speed conditioning；
- Trajectory Standardization：operator speed / pause / action density 的变化会影响 imitation learning，重采样 trajectory 可以提升性能。

### VLAFlow / DyPES-VLA / Qwen-VLA

最新 cross-embodiment 方法已经明确意识到 heterogeneous action spaces、sampling rates、control conventions 是大问题：

- VLAFlow 把 raw action fragmentation 当成 heterogeneous pretraining 障碍；
- DyPES-VLA 选择共享 dynamics prior + embodiment-specific native action heads；
- Qwen-VLA 甚至保留各 dataset 原始 action format，并用 embodiment-aware prompt 明确告诉模型 control convention。

## 6.2 为什么仍然没有完全砍

现有方法大多从“怎么统一 action”出发，尚未看到一个非常干净的机制结论回答：

> **heterogeneous action pretraining 的 negative transfer 中，到底有多少来自 observation/task 的 domain gap，又有多少来自同一个 normalized action coordinate 被赋予不同物理 semantics？**

如果能把这一点干净分出来，后面 method 空间当然存在：physical-unit/time-aware action representation、explicit execution semantics conditioning、native-head routing 等。

但这里有 Topic 09 风险：

- 我们目前只有“物理上很合理”；
- 没有强证据证明它是 heterogeneous pretraining performance 的主要瓶颈；
- 若为了 isolate 它需要人工构造多套 normalization/controller，再加大量控制，问题就开始变得不自然。

**当前评级：有事实基础，但机制主张尚未立住；继续搜，不注册。**

---

# 7. 新搜索线：training data 的 observation–action 时间对齐

这一条比 control-frequency 更底层，也更少被 VLA method paper 正面做，但当前最大的风险是：最后可能只是 data engineering。

## 7.1 真实现象证据

2026 的 VLA sensor/data survey 明确把 **temporal synchronization** 视为 observation–action causal integrity 的基础问题：

```text
训练假设： (o_t, a_t)
实际有 offset： (o_t, a_{t+Δt})
```

如果 camera、proprio、teleop command、robot execution 有不同 latency，dataset 会出现视觉上合理、因果上错误的监督对。

更具体的实践证据：一个 2026 π0 humanoid fine-tuning 项目在 open-loop audit 中发现了 **20 timestamp 的 visual–proprio misalignment**，修正它对降低初始预测误差是关键步骤。

UMI 也非常重视 camera / robot execution latency measurement，并做 latency matching。

DROID 的 raw data 则保留 camera recordings 和 low-dimensional trajectory，理论上可以做真正 timestamp-level audit。

## 7.2 为什么它可能不仅是工程问题

真正值得研究的版本不能是：

> “同步错了会掉性能。”

这太 obvious。

机制化问题应该更像：

> **当 BC 长期看到固定 lag 的 observation–action pair，它会把 lag 学成怎样的内部控制策略？**

例如：

- 学成一种 implicit anticipation：从旧 observation 预测未来 state 对应 action；
- 学成 trajectory-phase replay：使用 proprio/task phase 补偿 stale vision；
- 不同 dataset 有不同 lag 时，共同训练产生互相冲突的 inverse-dynamics mapping。

如果能证明其中一种，后面才有一般方法空间：learnable delay model、timestamp-conditioned action decoder、causal re-alignment、execution-time target 等。

## 7.3 当前风险

- Academic VLA literature 对 training-data lag 的系统机制分析看起来还不密，但 industrial/teleop 社区早就知道 latency/synchronization；
- 如果结果只是“offset 越大 performance 越差”，论文价值不足；
- 需要找到**一个反直觉、可泛化的 computation consequence**，否则不升格。

**当前评级：继续搜，优先找现成真实 dataset 中的 lag variation 和其 downstream signature。**

---

# 8. 这一轮新增的筛题原则

经过这一轮，机制题再加三条硬约束。

## 8.1 “有人已经做出对应 method”本身就是 collision 证据

即使没人写过我们完全相同的 analysis，如果已有多个方法都围绕同一瓶颈设计并有效，那么我们的机制题很可能只是事后解释一个已经被工程解决的问题。

除非机制能推翻现有方法设计或导出明显不同的 knob，否则降级。

## 8.2 机制题必须区分“configuration bug”和“learned computation”

例如：

- wrong action unnormalizer；
- camera timestamp 配错；
- control frequency 配错。

这些都可能造成巨大性能下降，但单独证明它们不够。

真正值得研究的是：

> **长期在这种系统性不一致监督下训练后，模型内部形成了什么可重复的计算策略或 shortcut？**

这一步才可能产生新的 learning method。

## 8.3 2026 VLA mechanism 的 bar 已经提高

`Not All Features Are Created Equal` 这类 activation injection 工作说明：

> probe AUC 已经不是强机制证据。

未来候选最好从一开始就存在一个 architecture-native 或 behavior-native causal intervention，不要最后才想怎么从 correlation 补到 causality。

---

# 9. 第二轮后的临时状态

| 搜索线 | 真实现象 | collision | 机制后 method 空间 | 当前决定 |
|---|---:|---:|---:|---|
| training-time world modeling / predictive state | 强 | 很高 | 有 | 降级，Topic15 不再是主搜索线 |
| contact/phase critical event dilution | 强 | 很高 | 强 | 砍 |
| adaptive replanning / action chunk | 强 | 极高 | 强 | 砍 |
| denoising commitment | 中 | 高 | 强 | 暂砍，缺独立现象证据 |
| RL vs SFT forgetting | 强 | 高，机制已有人分析 | 强 | 降级 |
| wrist-camera ego-motion in WAM | 强 | 极高 | 强 | 砍 |
| cross-embodiment physical action semantics | 强 | 中高 | 强 | 继续观察 |
| training-data temporal misalignment | 强工程证据 | 中低 | 若能找到 learned consequence 则强 | **继续深挖** |

下一轮不急着从这两条里硬挑一个。继续向更底层的 **data → learned computation** 和 **architecture → physical control** 断层搜索，特别关注那些已经在真实系统反复出现、但现有论文主要当成 preprocessing / system issue，而没有解释模型会因此学成什么的现象。

---

# 10. 本轮关键文献索引

- Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models  
  https://cwru-aism.github.io/vla-interp-page/
- StructRL: Structured Action-Space Exploration for Flow-Based VLAs  
  https://arxiv.org/abs/2608.15139
- Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation  
  https://arxiv.org/abs/2606.18960
- DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation  
  https://arxiv.org/abs/2608.20114
- Same Weights, Different Robot: A Deployment Safety View of VLA Policies  
  https://arxiv.org/abs/2606.03724
- APT: Action Expert Pretraining Improves Instruction Generalization of VLA Policies  
  https://arxiv.org/abs/2606.12366
- VLAFlow: Meta-Action Alignment for Generalizable Vision-Language-Action Models  
  https://arxiv.org/abs/2607.01586
- Qwen-VLA  
  https://arxiv.org/abs/2605.30280
- Sensing the Action: Rethinking Sensor Modalities and Multi-Modal Fusion in VLA Models  
  https://www.mdpi.com/1424-8220/26/11/3541
- DROID dataset  
  https://droid-dataset.github.io/
