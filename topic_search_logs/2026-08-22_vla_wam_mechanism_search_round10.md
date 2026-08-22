# 2026-08-22：VLA / WAM 机制选题搜索日志（第十轮）

> 状态：**继续搜索。本轮没有为了数量新增 E。**
>
> Round 10 按照用户再次强调的标准执行：不反复审 B/C/D，不追求“零 collision”，而是继续沿完全新的区域寻找 **顶会尺度的大问题 + 相对新的叙事角度 + 相对新的第一枪实验**。本轮最重要的结果不是新增候选，而是把几条一开始很像题的路线查到了足以明确降级/砍掉的程度，同时留下一个来自多个开源整模型 issue 的真实 action-generation anomaly，供后续继续观察。

目标 venue 仍按 ICLR / ICML / NeurIPS / RSS / ICRA / IROS 校准。

---

# 0. 本轮结论先行

本轮系统扫了：

1. tool / functional substitution；
2. self-caused vs externally-caused state change / agency；
3. `reach works, grasp never fires` 的跨模型 action asymmetry；
4. absent-target / impossible instruction / no-op；
5. multi-camera geometry / camera-role shortcut；
6. forward vs recovery / rollback asymmetry；
7. human-video pretraining 到底迁移什么；
8. termination / progress / hold-still；
9. MoE action expert 到底专门化什么；
10. capability-wise scaling。

结果：**没有一条比 B/C/D 更干净的新 E。**

这不是搜索不足，而是多条 broad question 在 2026 已经形成非常密集的方法赛道。

本轮最值得保留的未升候选现象是：

> ## **Motion learned, event missed**
>
> 多个公开 VLA 在 closed-loop 中都出现：
>
> ```text
> object localization / approach / alignment 正常
>     -> 到 pre-grasp/contact boundary
>     -> gripper close / decisive event 不发生
> ```
>
> 而 open-loop loss / trajectory fit 可以看起来很好。

这可能只是 normalization / padding / action weighting 的工程问题，也可能反映：现代 continuous generative action heads 对**长持续运动**与**稀疏 mode-switch event** 的学习结构不同。

但是 hybrid-action、arm/gripper decoupling、phase/event-aware VLA 已经有很近的工作，因此当前只记为 **strong anomaly line**，不升候选。

---

# 1. Tool / functional substitution：GEN-1.5 demo 很漂亮，但 broad question 已经被 affordance 线占据

Round 9 的 GEN-1.5 company-reported demo 里有：

- banana 当 brush；
- alternative dustpan / tool strategy；
- prompt composition 后补 intermediate repositioning / regrasp。

这很容易让人起题：

> **Do robot foundation policies understand tool function rather than object identity?**

但 collision 很快变得非常直接。

### AFUN

Affordance foundation model 把 object functionality / post-contact 3D motion 作为核心接口。

### VLAff

IROS 2026，把 human egocentric video 中的 grasp / trajectory affordance 提取成 embodiment-agnostic control cue。

### AffordanceVLA

把可交互区域 / 动作 affordance 正式接进 VLA。

### Tool-as-Interface

CoRL 2025 也已经把工具作为跨 embodiment action interface 来研究。

所以：

> “tool function 比 tool identity 更重要”

已经不是空白。

**结论：砍。**

---

# 2. Agency / self-caused vs externally-caused change：概念有意思，但 identification 不自然

曾考虑：

> **如果当前世界状态一样，policy 会不会区分这个变化是自己动作造成的，还是外部环境造成的？**

这似乎能审计 model 是否有：

- action-effect model；
- agency；
- causal attribution。

但继续推演后发现一个根本问题：

如果：

```text
current physical state fully observed
```

而下一步最优动作只依赖当前 Markov state，理性 controller **本来就不需要**记住“是谁造成的”。

为了让 causal origin 变得必要，我们必须再加入：

- hidden dynamics；
- unobserved damage；
- latent responsibility；
- delayed consequence。

这会让问题越来越靠人为 construction 才成立。

这正符合前几轮总结过的坏信号：

> **如果为了证明一个概念，必须不断追加隐变量和排除替代解释，问题开始离自然系统越来越远。**

**结论：砍。**

---

# 3. 一个真实跨模型异常：reach / align 会，grasp event 不发生

本轮最值得保留的材料来自公开 repo issue，而不是论文标题。

## 3.1 LingBot-VLA issue #48

https://github.com/Robbyant/lingbot-vla/issues/48

用户 full fine-tune LingBot-VLA：

- vision tower / QwenVL / action expert 全部训练；
- single-task pick；
- flow matching；
- open-loop MAE / loss 看起来正常；
- closed-loop 能稳定接近 target；
- 到 pre-grasp pose 后 hover；
- **gripper never closes**。

原模型是 75-D action，任务是 14-D，剩余 action 维度 zero-pad。

更重要的是，issue 下另一个用户报告在另一套 Inspire hand 上遇到同样问题。

这不是一个单独 user screenshot。

---

## 3.2 OpenPI π0.5 issue #912

https://github.com/Physical-Intelligence/openpi/issues/912

现象几乎一样：

```text
smooth approach
correct target alignment
-> no grasp
-> tiny actions keep pushing object
```

open-loop 同样看起来不错。

多个 commenter 报告类似现象。

有社区回复把它归因于 gripper normalization；这个解释必须认真考虑，不能直接拿 issue 当 mechanism evidence。

---

## 3.3 更关键：OpenPI π0.5 issue #1012

https://github.com/Physical-Intelligence/openpi/issues/1012

BEHAVIOR-1K / R1Pro setting：

- 官方 π0.5 checkpoint 与自己 fine-tune checkpoint 都出现同一 failure；
- target localization / navigation / arm reaching 正常；
- gripper dimensions 整个 rollout 接近 open 状态；
- never close。

而且 reporter 专门检查了一个重要替代解释：

### Raw demonstrations

close action 并不稀有：

```text
left close ~32%
right close ~30%
```

### 完整 preprocessing 后

close targets 仍然保留：

```text
left close ~35%
right close ~29%
```

所以至少在这个 reproduction 里，简单的：

> “close label 根本没进训练数据”

解释不够。

当然仍可能是：

- representation mismatch；
- inference postprocessing；
- action convention；
- checkpoint-specific pipeline bug。

因此不能直接宣布 generative head 有结构缺陷。

---

# 4. 一个诱人的机制叙事：Do VLAs Learn Motions Better Than Events?

这些 issue 共同暗示一个非常直观的 decomposition：

```text
continuous movement:
    reach / align / move / track

sparse control event:
    gripper close
    release
    contact switch
    mode change
    terminate
```

很多 action head 把它们都塞进：

\[
a_t \in \mathbb{R}^D
\]

再统一做 flow / diffusion / regression。

于是可以问：

> **Do continuous generative action heads systematically underrepresent sparse mode-switch events?**

这个题如果成立，会解释一个很有实践意义的现象：

> overall action loss / open-loop trajectory fit 很好，却因为极短的 event timing / mode switch 没学到而 closed-loop 彻底失败。

这比“gripper channel 权重太小”稍大一层。

---

# 5. 但不能现在升候选：hybrid / event / phase action 已经很拥挤

## DAM-VLA — ICRA 2026

已经明确认为：

- arm movement；
- gripper manipulation

具有不同学习难度和精度需求，并设计不同 diffusion experts + routing / weighting。

## FPC-VLA

明确把 pose 与 gripper state 当不同 behavioral semantics，做 decoupled fusion，并围绕 gripper failure 生成 supervision。

## Libra-VLA — ACL 2026 Long

直接批评 monolithic action generation 忽略 manipulation 的 **Hybrid Action Space**，将 discrete macro intent 与 continuous fine control 拆开。

## ProgressVLA / PALM / SwitchVLA / event-based work

termination、phase change、contact switching、progress 都已经形成专门 line。

因此 broad title：

> `robot action is not one homogeneous continuous vector`

已经不能拿。

当前剩余的新意只能非常具体地证明：

> **统一 continuous generative geometry 对 sparse action events 本身存在系统性学习偏差。**

这个 prerequisite 现在还没有被证明。

所以：

**保留 anomaly，不摘 E。**

---

# 6. 如果以后继续这条，第一枪应该怎么做

不是再做一个 hybrid head。

第一枪应做 controlled diagnosis。

固定：

- same task；
- same demonstrations；
- same vision-language backbone；
- same training budget。

比较 action-generation families：

1. flow / diffusion continuous head；
2. ACT-like regression / chunk model；
3. autoregressive / discretized action head；
4. optional hybrid event+continuous diagnostic head。

把 time points 分成：

```text
motion interior
vs event boundary
```

事件包括：

- gripper flip；
- release；
- contact onset；
- mode switch；
- terminate。

测：

- conditional per-event error；
- event timing error；
- event miss rate；
- action calibration；
- closed-loop success conditional on reaching pre-event state。

最关键的 matched test：

> **把 robot reset 到 expert 的 pre-grasp state。**

如果不同 architecture 都能 reach，但 flow policy 在同一 clean pre-event state 仍显著漏 close，才开始像 action-generation mechanism。

如果只是 data / normalization，题直接死。

---

# 7. Do Robot Foundation Policies Know When Not to Act? —— 标题很好，但已被直接占

另一条一度很自然的方向：

> demonstrations 几乎都是“任务可做、目标存在、操作者开始执行”的 positive trajectories；VLA 会不会形成一个 `instruction -> always act` 的 actionness prior？

这会导致：

- target 不存在还乱抓；
- impossible request 还执行 plausible motion；
- 应等待却强行推进。

但 exact collision 很直接。

## Do What? Teaching Vision-Language-Action Models to Reject the Impossible

https://arxiv.org/abs/2508.16292

已经专门研究 false-premise / absent-object instruction，并训练 VLA detection + correction。

## OBEYED-VLA

https://uark-aicv.github.io/OBEYED_VLA/

专门评估 absent-target rejection，并发现普通 end-to-end baseline 在目标不存在时仍有非常高的 off-diagonal grasp rate。

所以：

> **Do VLAs know when not to act?**

本身已经不能拿。

`positive-demo -> actionness prior` 可以作为 data insight 保留，但不是新候选。

---

# 8. Camera-role shortcut：真实，但 view robustness / camera geometry 已成赛道

曾考虑：

> multi-view VLA 到底理解 camera geometry，还是记住 `slot 0 = base camera, slot 1 = wrist camera` 这种 dataset convention？

这很适合做 camera permutation / role swap。

但 collision 很快：

## CamVLA

https://alibaba-damo-academy.github.io/CamVLA/

作者直接指出 standard VLA 会把 fixed camera-to-action mapping memorize in weights；仅 15° 相机变化就可把 success 从 65.3% 降到 6.3%。

## Cross-View Action Consistency — 2026-08

https://arxiv.org/abs/2608.06965

构造 same MuJoCo state / different camera views 的 action-equivalent pairs，并直接约束 flow field cross-view consistency。

## OC-VLA / Multi-Camera View Scaling

也已经系统处理 observation/action frame alignment 与 view diversity。

所以 camera shortcut 虽然是真的，但 broad story 已经进入主线。

**结论：不摘。**

---

# 9. Forward vs recovery / undo：2026 已经非常拥挤

一个很自然的 capability asymmetry：

> demonstration 都单调向 success 前进，policy 会不会只会“继续往前”，不会 rollback / undo / recover？

但：

## FLARE — CVPR 2026

直接把根因写成：

> VLAs are trained on **trajectory-monotonic, failure-free demonstrations**。

并做 Retry / Reset recovery。

## RePO-VLA

把 success / recovery / failure trajectories 分开处理。

## See, Plan, Rewind / B2FF / CoRe

分别从 progress、familiar milestone、counterfactual realignment 做 recovery。

## SwitchVLA

甚至直接统一 forward / rollback / advance behavior。

因此：

> `VLA can progress but cannot undo`

已经不能作为新大题。

---

# 10. Human-video pretraining 到底迁移什么：问题很大，但答案候选已经各自成赛道

现在 foundation robot policy 大量使用 human egocentric video。

一个自然问题：

> **What does human video actually teach a robot policy?**

至少可拆：

```text
semantic / perceptual diversity
vs 3D geometry
vs interaction-effect / dynamics
vs human motor pattern
vs high-level intent
```

但是每一个答案已经有很强的工作线：

- EgoVLA：强调 scale、scene/task diversity 与 human hand action；
- Being-H0：human hand as foundation manipulator + physical instruction tuning；
- VIPA-VLA：visual-physical / 3D alignment；
- ConLA：action dynamics vs visual nuisance；
- motion-focused latent action：motion dynamics / action intent；
- GazeVLA：intention as embodiment-agnostic bridge；
- HARP：human-robot aligned representations；
- Mimic / VAM：video backbone 的 semantics + dynamics + behavior prior。

所以“human video 为什么有用”如果只做 component ablation，很容易变成 survey-style decomposition，而不是新的 mechanism thesis。

**当前不摘。**

如果未来出现一组公开 nested checkpoints 能做 clean causal decomposition，再回来。

---

# 11. Termination / keep-moving / no-op：ProgressVLA / PALM 已经正面占据

另一个实际部署问题：

- task 已完成还继续动；
- 接触成功后继续推；
- repeated action；
- premature / late termination。

但 ProgressVLA 已直接指出现有 VLA 缺乏 progress awareness、依赖 hand-crafted termination；PALM 又联合预测 action + subtask progress 来决定 continue / transition / terminate。

因此 termination 不再作为新题。

---

# 12. MoE action expert 到底专门化什么：也已经进入机制论文

LingBot-VLA 2.0 等新模型使用 MoE action expert，很自然可以问：

> experts 到底按 embodiment、skill、phase、task 还是 action semantics 分工？

但是：

## Emergent Compositional Skills in Mixture-of-Experts VLAs — 2026-07

https://arxiv.org/abs/2607.20771

已经直接问：无预定义 decomposition 时，MoE experts 是否会**涌现成 reusable interpretable primitives**，并观察到跨任务 expert reuse / low-level behavior specialization。

同时：

- AtomicVLA：skill-guided MoE；
- SkillNet（ICML 2026）：hierarchical skill-context MoE；
- PAMAE：phase-aware action experts；
- DiTEA：instruction-gated task experts；
- DriveMoE：skill-specialized action MoE。

因此继续缩成 `embodiment vs skill routing` 不够大。

**结论：砍。**

---

# 13. Capability-wise scaling：重要，但和 B 太近

Xiaomi-Robotics-1、Dyna-2、Rethinking VLA Scaling 已开始提供真实 foundation scaling curves。

曾考虑：

> scale 到底先消除 perception/OOD 错误，还是也会消除 dexterity/contact/recovery/event errors？

这可以做 capability-wise scaling decomposition。

但它和 B 的核心：

> **foundation scale 是否改变 action-generation mechanism**

高度相邻，而且如果只做 failure taxonomy vs scale，很容易变 benchmark / scaling analysis。

因此不另起 E。

---

# 14. 一个重要的风格校准：2026 mechanism paper 已经在问“how should a signal enter VLA?”

2026-08-05 新工作：

**How Should Vision-Language-Action Models Use Proprioceptive State?**  
arXiv:2608.03052

它不是“加 proprioception 提几分”，而是 controlled 地问：

1. current state 到底在哪些 task 有用？
2. state history 的 gain 是真实 temporal information 还是只是更多 conditioning capacity？
3. state 应该进入 VLM backbone 还是 action-generation module？

并固定 backbone / data / action representation / protocol，对多种 state interface 和 history length 做 matched experiment。

这正是以后我们筛 mechanism 题的 bar：

> **一个单一、自然、可以被 clean factorial experiment 回答的 computation question。**

Round 10 没有找到比 B/C/D 更强的新母题，就不硬升 E。

---

# 15. Round 10 当前状态

Active provisional shortlist **不变**：

- B — How Do Robot Foundation Policies Generalize Actions?
- C — Why Does Task Decomposition Help Robot Foundation Policies?
- D — Do Robot Foundation Policies Learn Motor Equivalence Classes?

A inactive / archived。

本轮新增两个值得以后观察、但尚未摘出的 search lines：

### S1 — Do Continuous Action Heads Miss Sparse Control Events?

有跨 OpenPI / LingBot practitioner anomaly，但 hybrid action / gripper-specialized / phase-aware literature 很近。必须先证明 architecture-level event bias，不能从 issue 直接跳到 paper claim。

### S2 — What Does Human Video Transfer to Robot Control?

问题重大，但 semantic / geometry / dynamics / intent / motion 等每个解释都已经形成方法线。需要未来更干净的 causal decomposition 才值得回来。

---

# 16. 下一轮

Round 11 不再继续枚举：

- camera shortcut；
- MoE routing；
- no-op；
- termination；
- gripper head。

转向更基础的 **control / motor-learning / physical learning** 问题：

1. task-relevant vs task-irrelevant variability；
2. feedback correction laws 是否从 demonstrations 中被抽象出来；
3. policy 是否学习 invariant task-space error correction，而不是 state-action lookup；
4. same physical error 在不同 task / object / embodiment 下是否触发共享 correction computation；
5. 从 human motor control / optimal feedback control / adaptive control 借问题结构，而不是借一个新 module。

继续坚持：

> **先找自然大问题，再找方法；没有新题也可以有高质量搜索日志，但不为了候选数量降低 bar。**
