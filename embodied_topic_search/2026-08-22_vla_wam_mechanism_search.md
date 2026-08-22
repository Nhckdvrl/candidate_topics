# 2026-08-22：VLA / WAM 机制选题搜索日志

> 状态：**持续搜索中，尚未决定新的正式 Topic。**
>
> 这份文档记录今天这一轮搜索的真实思考顺序，包括已经砍掉的方向。目的不是把每个想法包装成候选题，而是防止重复踩坑，并把后续搜索收敛到真正有机制价值、又能自然留下方法空间的问题上。

---

# 0. 这轮搜索为什么要重新开始

Topic 09 的失败给出了一个非常明确的教训：

> **一个概念上漂亮的 identification，并不意味着现实系统会自然提供足够的 variation。**

Topic 09 需要 same-state、same-family checkpoints 之间大量 bidirectional competence crossover。这个 prerequisite 没有被文献证明，我们只是觉得“不同 checkpoint 总会各有所长”。最后 3,600 个 rollout 证明这基本不成立。

因此这次选 VLA / WAM 机制题，硬性要求变成：

1. **现象先存在。** 最好已有两篇以上独立工作从不同角度观察到，不再赌一个新 anomaly。
2. **机制问题必须比现象本身更具体。** 不能只问“为什么会这样”，而要有清楚的 computation / representation / training-dynamics 对象。
3. **不能依赖稀有 natural crossover / disagreement 才能识别。** 如果依赖，必须先有 instance-level evidence。
4. **证明完以后要有路。** 不要求现在就把 method 设计死，但机制结论必须自然留下可改的 knob，例如 loss、routing、schedule、action representation、conditioning、training mixture、replanning 等。
5. **collision-first。** 在投入代码前先查清楚：问题是不是已经被 2026 的邻近工作做成了方法论文。
6. **工程成本单独评估。** “第一枪科学上很简单”不等于“只跑一下就很便宜”。

一个候选如果只能形成：

```text
发现现象 -> 做 attribution -> 结束
```

不再优先。

更希望形成：

```text
已有现象
  -> 找到控制性能的具体机制瓶颈
  -> 机制成立后自然知道该改哪个地方
  -> 后续可以发展成方法
```

---

# 1. 第一条搜索线：training-time world modeling 到底留下了什么

## 1.1 为什么一开始觉得它很强

近期 WAM 有一个非常明显的共同趋势：

```text
训练时预测未来 -> action 变好
部署时未必需要真正生成未来
```

这使一个问题很自然：

> future prediction 的收益到底通过什么内部机制进入 policy？

这条线的吸引力在于 prerequisite 已经比较扎实，不需要我们先证明 world modeling 有用。

## 1.2 相关工作

### Fast-WAM

核心现象：training-time video co-training 明显有用，但 test-time 可以删掉 future generation branch，部署性能仍然基本保留。

这说明 world-modeling 的价值至少有很大一部分发生在**训练 representation shaping**，而不是测试时显式 rollout。

### Light-WAM

走轻量 WAM 路线，冻结大部分 video backbone，用 adapter / action 模块吸收 future supervision。公开 checkpoint / 训练路径，相对适合作为机制分析平台。

### World Tokens

同样强调训练时未来监督，部署时留下被 world-model objective 塑造过的中间 state / token，再直接服务 action。

### Beyond Task Success

开始分析不同 VLA/WAM 内部是否存在 memorized / reactive / predictive structure，说明“future-oriented representation 能被读出来”这件事已经有人正面做。

### AGRA / Making Foresight Actionable

已经开始对 WAM representation 与 action grounding 的 mismatch 做 causal / interventional analysis，说明“future representation 是否被 action 正确使用”也正在快速变成一个拥挤方向。

## 1.3 一开始形成的机制题

曾考虑：

> **Does training-time world modeling leave a causal predictive state inside the deployed policy?**

更直白地说：

> 训练时预测未来帮助了 action，那部署时 policy 真的是靠“关于未来的信息”在行动，还是 future prediction 只是一个辅助 regularizer？

这个问题本身是好的，而且已经注册过 Topic 15：

`15_predictive_policy_state/`

## 1.4 为什么现在降级

用户指出这一块**太拥挤**，这个判断是对的。

2026 已经同时出现：

- predictive representation analysis；
- future latent alignment；
- action-grounded world representation；
- world-token / latent-world supervision；
- causal intervention；
- test-time / training-time world model 分离。

所以即使我们找到一个漂亮机制，审稿时也很容易落到：

> “又一篇解释 future prediction 为什么有用的 WAM 文章。”

问题不是它不成立，而是**研究密度太高，方法空间也会被快速挤压**。

### 当前处理

- Topic 15 目前仍在仓库正式候选里，暂不自动归档；
- 但本轮搜索不再围绕它继续扩展；
- 除非后面出现一个非常不同、非常窄而又重要的机制切口，否则不把它作为首选。

---

# 2. 第二条搜索线：VLA 到底是不是真的 closed-loop

## 2.1 出发点

已有工作反复报告：

- 轻微 object pose shift 会显著打击 VLA；
- action chunk 让系统在执行一段时间内看不到新反馈；
- trajectory overfitting 会让模型在环境变化后继续 replay familiar motion；
- action chunk execution horizon 对性能影响很大。

这引出一个很自然的问题：

> **VLA 虽然 API 上每隔几步重新看图，但 computation 上到底对世界变化有多敏感？**

一度考虑过把 vision change 和 proprio change 正交干预，问 policy 的下一次 replan 主要跟谁走。

## 2.2 为什么没有继续推成正式候选

这个问题非常自然，但 collision 也很近。

2026 已经有：

- trajectory overfitting / object-pose robustness 分析；
- action chunk blind spot；
- visual attribution / interventional interpretability；
- adaptive replanning / execution horizon 方法。

所以如果最后只是得到：

> “VLA 对 proprio / execution phase 更敏感，对视觉变化反应不足”

那很可能只是把已有 robustness 结果换成 mechanism language。

### 结论

**保留为母题，但不够稀缺。**

如果以后找到一个更具体的内部承诺机制，例如“某个计算阶段以后视觉反馈已经无法改变高层动作模式”，才可能重新变强。

---

# 3. 第三条搜索线：action chunk 到底是不是一个内部计划

## 3.1 为什么想到它

现代 flow / diffusion VLA 通常一次生成一个 future action chunk。

表面上看：

```text
[a_t, a_{t+1}, ..., a_{t+H}]
```

只是一起输出。

但 action tokens 往往互相 self-attend，因此一个更机制化的问题是：

> **当前真正要执行的第一个动作，是否因果依赖更远未来的 action tokens？**

如果依赖，chunk 就不仅是 batching，而是模型内部真正存在 future-action planning / coordination。

如果不依赖，很多“长 horizon chunk 等于规划”的叙事可能过强。

## 3.2 collision audit 后为什么降级

这一块 2026 已经非常快地拥挤起来。

### Coarse-to-Control

论文：`Coarse-to-Control: Action-Token Planning for Vision-Language-Action Models`

https://arxiv.org/abs/2606.07107

直接把 coarse future action tokens 当成 planning medium，再生成 executable actions。

### ACoT-VLA

CVPR 2026：`Action Chain-of-Thought for Vision-Language-Action Models`

已经明确主张 reasoning 应该发生在 action space，并使用 coarse action intent 引导控制。

### Continuous Reasoning for VLA

https://arxiv.org/abs/2606.00229

虽然不是同一种 action-token 因果分析，但同样在研究“连续控制里的 internal reasoning medium”。

### AutoHorizon / VLA Knows Its Limits

https://hatchetproject.github.io/autohorizon/

已经对 action self-attention 和 chunk structure 做分析，并基于它决定 execution horizon。

因此即使“远未来 action token 是否反过来约束当前 action”仍可能没有被完全做掉，邻近工作已经太密。

### 结论

**问题不错，但容易被归入 action-token planning / adaptive chunking 大赛道，暂时不推。**

---

# 4. 第四条搜索线：flow / diffusion VLA 在什么时候“改不了主意”

## 4.1 机制直觉

近期几类工作共同暗示：

- denoising 早期形成较粗的动作结构；
- 后期更多做局部 refinement；
- 不同 task phase 的 denoising stability 不同；
- execution horizon 应该随状态变化。

因此曾形成一个很清楚的问题：

> **在一次 flow/diffusion action generation 里，主运动方向/子任务意图是在第几个 denoising step 被“承诺”的？承诺以后，新信息还能不能改变它？**

这比普通 hidden-state probing 更接近真正的 computation dynamics。

## 4.2 为什么目前不够空

### DVAC

`Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies`

https://arxiv.org/abs/2606.03847

已经发现：

- free-space / predictable phase 的 clean-action estimates 比较稳定；
- contact-rich / precision phase 的 denoising trajectory 波动更大；
- 可以直接利用 denoising variance 决定 replanning horizon。

### AutoHorizon

同样从 action attention / chunk structure 里提取 predictive limit，直接做动态 horizon。

这意味着：

> “denoising 内部有一个能预测哪里该重规划的结构”

已经不是空白。

如果我们只做 commitment point，很容易变成解释 DVAC / AutoHorizon。

### 什么时候可能重新有价值

只有在能找到一个更强的机制命题时，例如：

> **感知信息只在 denoising 的某个早期窗口有能力改变动作 mode，后期 conditioning 实际上已经失去控制权。**

如果这个成立，后面自然可以发展新的 conditioning schedule / feedback injection / partial restart 方法。

但目前还没找到足够文献证据说明这个具体现象真的存在，因此**不注册**。

---

# 5. 第五条搜索线：cross-embodiment 数据里的时间尺度 / action semantics 不一致

这是目前值得继续深挖、但**还远没有到注册题目**的一条线。

## 5.1 事实基础

Open X-Embodiment 本身就明确存在：

- 不同 robot；
- 不同 action representation；
- 不同 absolute / delta / velocity semantics；
- 不同 control frequency；
- 不同传感器和 controller。

Open X repo 对 action space 的描述就指出，不同数据里的 action dimension 可能表示 absolute value、delta change 或 velocity。

https://github.com/google-deepmind/open_x_embodiment

CrossFormer 也直接把 **varying control frequencies** 列为 cross-embodiment policy 的核心困难之一。

论文：`Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation`

https://proceedings.mlr.press/v270/doshi25a.html

当前很多数据混合方案主要解决：

- coordinate frame；
- dimension alignment；
- normalization；
- padding / action mask；
- embodiment conditioning。

但一个物理问题值得继续核：

> **“数值相同的 action”在不同 control frequency / controller 下，是否对应完全不同的物理变化速度和时间尺度？统一 action space 后，模型是不是被迫把一个 token / vector 同时解释成多个不同的动力学语义？**

例如直观上：

```text
Δx = 2 cm @ 5 Hz
```

与

```text
Δx = 2 cm @ 20 Hz
```

如果每个 control step 都解释成相同 delta，它们隐含的是完全不同的 end-effector velocity / action duration。

这类问题不是“维度没有对齐”，而是：

> **离散 control step 本身缺少统一的物理时间语义。**

## 5.2 为什么这条线有潜在 method 空间

如果以后能证明一部分 cross-embodiment interference 来自这种 **time-scale aliasing**，后面的方向非常自然：

- time-conditioned action representation；
- rate-normalized action target；
- physical-duration token；
- continuous-time / velocity-normalized action interface；
- 根据 embodiment controller 的实际时间尺度做 data mixing / action decoding。

不需要现在决定具体方法，但至少机制结论不会停在“哦，模型有 bias”。

## 5.3 最大风险

目前还不能把它叫好题，因为还缺两件关键东西：

### 风险 A：可能已经有人系统做过

cross-embodiment / action-unification 方向很大，需要继续查：

- 有没有明确研究 control-rate normalization；
- 有没有用 physical time / dt 统一动作；
- 有没有把 negative transfer 直接归因于 control frequency mismatch；
- 最新 foundation policy 在 preprocessing 时是否早已隐式处理这个问题。

如果已有一条成熟线，这个方向立刻降级。

### 风险 B：现象可能没有足够 effect size

不同 dataset 虽然 frequency 不一致，但 model 可能通过 image history、embodiment identity、数据归一化等自动解开。

因此不能像 Topic 09 一样直接默认它一定造成严重 aliasing。

必须先找**文献里的真实 negative transfer / controlled evidence**，或者用一个很小的 dataset-level audit 先确认 action-time semantics 的冲突密度。

### 当前状态

**继续搜索，不注册。**

这是目前相对更值得查的一条“数据物理语义 -> 内部机制 -> 方法口”路线。

---

# 6. 已明确砍掉 / 降级的方向

| 方向 | 当前判断 | 主要原因 |
|---|---|---|
| training-time future prediction 如何进入 policy | 降级 | WAM future representation / grounding / causal use 已迅速拥挤 |
| future representation 是否存在 | 砍 | 已有直接 representation analysis |
| WAM 是否真正 action-conditioned | 砍/拥挤 | controllability / counterfactual consistency 已有人直接做 |
| VLA vision vs proprio shortcut | 砍/拥挤 | 已有 shortcut / gradient rebalance / interpretability 工作 |
| VLA 是否 closed-loop | 保留母题但不注册 | trajectory overfitting、chunk blind spot、adaptive replanning 已很近 |
| action chunk 是否构成 planning | 降级 | ACoT、Coarse-to-Control、Continuous Reasoning、AutoHorizon 邻近过密 |
| denoising 内部是否有 task-phase signal | 砍 | DVAC 已直接证明并做成方法 |
| denoising commitment point | 暂存 | 有机制味，但尚缺已知现象支撑，且邻近 adaptive chunking 太密 |
| cross-embodiment control-time semantics | **继续深挖** | 可能有物理上真实而未被充分机制化的问题，且后续方法口自然 |

---

# 7. 目前读到的关键论文 / 项目索引

这不是完整综述，只记录对本轮筛选真正改变判断的工作。

## Action chunk / planning / replanning

1. **Coarse-to-Control: Action-Token Planning for Vision-Language-Action Models**  
   https://arxiv.org/abs/2606.07107  
   影响：使“action chunk 内部是不是 plan”这一方向显著拥挤。

2. **Continuous Reasoning for Vision-Language-Action**  
   https://arxiv.org/abs/2606.00229  
   影响：continuous latent reasoning 已经开始成为独立方向。

3. **VLA Knows Its Limits / AutoHorizon**  
   https://hatchetproject.github.io/autohorizon/  
   影响：action self-attention / chunk predictive limit 已经被用于动态 execution horizon。

4. **Denoising Tells When to Replan / DVAC**  
   https://arxiv.org/abs/2606.03847  
   影响：denoising variance 与 task phase / replanning need 的联系已经被正面利用。

5. **ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models**  
   CVPR 2026  
   https://openaccess.thecvf.com/content/CVPR2026/html/Zhong_ACoT-VLA_Action_Chain-of-Thought_for_Vision-Language-Action_Models_CVPR_2026_paper.html  
   影响：action-space reasoning 本身已经很拥挤。

6. **FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking**  
   https://arxiv.org/abs/2607.24008  
   影响：异步执行带来的 observation/action 时间错位已经有人专门处理，说明 time alignment 是现实工程问题，但也需要注意 collision。

## Cross-embodiment / action unification

7. **Open X-Embodiment**  
   https://github.com/google-deepmind/open_x_embodiment  
   关键事实：不同数据的 action 可能是 absolute / delta / velocity；control loop 也不是统一物理时间接口。

8. **Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation (CrossFormer)**  
   https://proceedings.mlr.press/v270/doshi25a.html  
   关键事实：论文明确把不同 sensors、actuators、control frequencies 列为 multi-robot policy 的挑战。

9. **The Embodiment Gap in Robot Foundation Models**  
   https://arxiv.org/abs/2608.18433  
   影响：cross-embodiment 仍然是活跃大方向，因此任何“统一动作语义”题都必须非常具体，不能泛泛谈 embodiment mismatch。

---

# 8. 下一步搜索顺序

接下来不应该继续随机看“有趣 VLA paper”，而是按下面顺序验证第五条线及其他可能的新线。

## A. 先把 control-frequency / time-scale 这条查死

重点搜索：

- control frequency normalization；
- action duration conditioning；
- continuous-time robot policy；
- cross-embodiment action rate mismatch；
- delta action vs velocity action unified training；
- resampling / temporal abstraction 对 VLA transfer 的影响；
- OXE / DROID / Bridge / RoboNet 等实际 Hz 和 preprocessing。

要回答：

1. 有没有人已经明确提出并解决这个问题？
2. 最新大 VLA 的 preprocessing 是否已经把 dt 消掉？
3. 有无 controlled ablation：same data 不同 sampling rate 会不会导致 performance / representation 改变？
4. negative transfer 是否真的与 temporal semantics 冲突相关？

任何一个答案如果说明“这已经是标准 practice”，就砍。

## B. 继续找类似的“物理语义 mismatch”

优先找以下类型，而不是再找一个 feature probe：

- 同一个 action representation 在不同 controller 下物理效果不同；
- gripper action 的离散/连续/滞后语义不一致；
- camera timestamp 与 action timestamp 不一致；
- demonstration latency / teleoperation delay 被模型当成 dynamics；
- action chunk target 在 contact transition 前后是否存在系统性标签不一致；
- proprio state 与图像并非严格同步，却被当作同一时刻 observation。

这些问题如果真实存在，往往既有机制含义，也能自然导出 data / architecture / objective 的方法。

## C. 对任何新候选都先问“之后呢？”

候选进入下一轮之前必须能回答：

> 如果机制成立，模型里有什么 knob 是因此变得明确可改的？

只接受类似：

```text
发现 temporal aliasing
-> 说明统一 action space 缺少 physical time
-> 后续可研究 time-aware action interface
```

不接受：

```text
发现某层有某 feature
-> 很有意思
-> 然后不知道怎么办
```

---

# 9. 当前判断

截至这一版日志：

- **还没有找到我愿意强推的新正式 Topic。**
- WAM future-prediction 主线虽然问题好，但过于拥挤，暂时退出主搜索方向。
- action-chunk planning / denoising / adaptive horizon 同样已经快速形成拥挤带。
- 当前相对值得继续深挖的是：**跨 embodiment / 跨数据集 action 的物理时间语义是否被统一表示抹掉，从而产生 time-scale aliasing / negative transfer。**
- 但它现在仍只是“搜索方向”，不是候选题；下一步必须先做 collision audit 和现象证据核查。

后续每出现一个新的母题、关键论文或 kill 判断，都继续更新本文件，而不是等最后才补历史。
