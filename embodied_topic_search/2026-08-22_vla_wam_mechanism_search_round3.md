# 2026-08-22：VLA / WAM 机制选题搜索日志（第三轮）

> 状态：**继续搜索，不注册新 Topic。**
>
> 第二轮已经说明：2026 年 VLA/WAM 里第一层 architecture limitation 基本都有人占。第三轮开始改变策略：不再从某篇 paper 的 limitation 往前续一格，而是寻找 **两个已经成立的结果之间尚未解释的二阶矛盾**。同时继续要求：机制成立以后必须自然留下方法口，但现在不把方法设计死。

---

# 1. Cross-embodiment “bridge / anchor data”——直接砍

第二轮一度形成一个比较自然的解释：

- `Rethinking VLA Scaling` 报告真实 heterogeneous robot datasets 混得越多可能越差，出现 negative transfer；
- OXE-AugE 把同一批 task / scene / trajectory cross-paint 成多个 embodiment，反而看到 robot diversity 的正向 scaling。

于是曾提出：

> 机器人多样性本身也许不是关键；关键是不同 embodiment 之间有没有共享 task/state anchors，让模型知道两套 action manifold 是同一 skill 的不同实现。

这个问题听起来很好，而且后续方法也自然是 bridge data / anchor tasks。

但 collision audit 直接撞到：

## Data Analogies Enable Efficient Cross-Embodiment Transfer

arXiv:2603.06450。

这篇工作的核心就是比较 unstructured cross-embodiment diversity 与**场景、任务、轨迹相互对应的 paired demonstrations**，并发现 morphology shift 下 paired analogies 明显更有效；真实机器人平均提升约 22.5%。

也就是说“共享任务锚点 / paired analogies 是跨 embodiment transfer 的关键”不仅有人想到，而且已经从问题做到数据方法。

**结论：砍，不换名字包装。**

---

# 2. WAM 中 `command != realized motion`——直接砍

曾考虑一个很物理、也很自然的 WAM 机制问题：

> robot dataset 记录的 action 经常是 controller command / target；真正发生的 robot motion 还经过 IK、servo、compliance、latency、contact。让 world model 直接从 command 预测 future，会不会把 controller realization 和 scene dynamics 混成一个问题？

如果成立，后面显然可以做“先预测/提供 robot realization，再预测 scene response”。

但 2026-07-24 的：

## Robot-Factored World Models via Robot Rendering

已经几乎原样提出这个问题。

作者明确把 action → future 拆成：

```text
command
  -> controller / robot realization
  -> realized robot motion
  -> contact / scene response
```

并指出直接从 command 学 future 等于让 WAM 同时学习 controller realization 与 world dynamics；他们使用 controller / kinematics rollout 得到 nominal robot trajectory，作为 world model 的中间物理条件。

**结论：正面 collision，砍。**

---

# 3. Action chunking 的真正机制：一个值得深挖、但很危险的二阶矛盾

这一轮最重要的新论文是：

## Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?

arXiv:2608.02547，2026-08-03，Lazzati et al.

这不是普通 action-chunk method paper，而是一篇真正的 mechanism paper。它系统检验了过去常见的几种解释：

- temporal consistency；
- horizon reduction；
- representation learning。

结果表明这些解释都不足以解释 action chunking 的成功。

论文最后把收益分到两个更基本的机制：

1. **non-Markovian expressivity**；
2. **implicit ensembling across temporal relationships**。

最反常、也最重要的实证是：在 LIBERO-90 的 human demonstrations 上，当前 action `a_t` 经常从**过去的 observation**预测得比从当前 `o_t` 更好。

作者比较：

```text
a_t | o_t
a_t | o_{t-1}
...
a_t | o_{t-d}
```

在 LIBERO-90 上，约 `d≈10` 的过去 observation 甚至可以比当前 observation 给出更低 action-prediction error；这种 delayed policy 在多个设置中可以 match / exceed 普通 action chunking，π0.5 也观察到类似现象。

论文因此认为 human demonstration 具有明显 non-Markovianity。

这条结果非常值得认真，因为它改变了我们对 action chunking 的解释。

---

# 4. 第一种解释：chunk tail 是“当前 observation 不可能知道”的监督——有意思，但邻近太密

最初想到一个机制：

> action chunk 用 `o_t` 一次性监督 `a_t ... a_{t+H}`。但人类在未来真正做出的 `a_{t+k}` 可能依赖未来视觉、contact、滑动、grasp success、subtask transition。因此 chunk tail 中存在从当前 observation 不可实现的监督。

如果 action tokens / flow trajectory 内部强耦合，不可预测的 tail 可能反过来伤害真正马上执行的 head action。

这条后面自然有 predictability-aware horizon / tail masking / causal horizon 的方法空间。

但 collision 很近：

- `Learning to Assist` 已经研究 chunk 跨 latent transition 导致 premature actions 的 demonstration action leakage；
- TRACT 直接处理 chunk 跨 procedural phase 的 temporal mismatch；
- adaptive horizon / chunk execution 论文又已经很多。

因此 generic “long chunk tail is bad”不够独立。

**当前决定：不作为主候选。**

---

# 5. 第二种解释：action chunking 的 delayed-predictor advantage 到底来自“任务 memory”，还是“数据采集 delay”？

这是第三轮最值得保留的一条。

Aug-3 action-chunk mechanism paper 把 `a_t` 更容易由 `o_{t-10}` 预测解释为 human demonstrator non-Markovianity。

但 human teleoperation 本身天然存在另一套时间结构：

```text
human sees scene
   -> perception / reaction delay
   -> input device command
   -> network / software delay
   -> controller realization
   -> robot motion
```

因此 dataset 中记录成同一个 index 的 `(o_t, a_t)`，未必对应真正的 human causal decision pair。

一个非常具体的问题由此出现：

> **Action chunking 所谓的 non-Markovian advantage，有多少来自任务本身真的需要 history，有多少只是 human reaction / teleoperation / sensing-control latency 让当前 action 实际上对应过去 observation？**

这个问题和普通“latency 会影响机器人”不同。

如果答案是后一部分很大，那么一个 canonical robot-learning trick 的解释会被改写：

> action chunking 的一部分强大，也许是在替 demonstration collection pipeline 吸收 causal temporal misalignment。

## 已有支持

- robomimic 早期工作已经明确指出 human actions 会受当前 observation 之外的因素影响，包括 control device 和 action history；
- UMI 明确做 latency matching；
- Mobile ALOHA 观察到 base 与 arm 硬件 delay 不同，并利用 action chunk 灵活处理；
- 2026 VLA sensor/data review 把 temporal synchronization 称为 observation–action causal integrity 问题；
- IsaacLab teleoperation 文档明确指出 controller gain 与 tracking latency 的 trade-off；
- 多种 real teleoperation 系统公开测量几十到上百毫秒的 human/device/vision/control latency。

## 但必须提高证据 bar

仅仅画出 `loss(delay)` 曲线不够，因为 Aug-3 paper 已经做了。

真正值得的结果必须类似：

> **在 causal temporal re-alignment 后，过去 observation 的预测优势显著消失，并且 Markov policy 与 action-chunk policy 的闭环差距同步缩小。**

这样才能说明我们解释的是 chunking mechanism，而不是又发现 teleoperation 有 latency。

## collision 风险

这条离 Aug-3 Berkeley 新论文很近；作者自己已经在结论中讨论 human visually guided behavior 的有效更新频率可能只有 2–10 Hz，并指出 demonstrator behavior × control frequency × chunking 值得继续研究。

因此它虽然当前没被正面做掉，但**非常可能快速拥挤**。

**当前评级：强备选，高 bar，不注册。**

---

# 6. 更一般的时间参数化解释——被 ISR 明显占住

把上面问题再一般化，会得到：

> human demonstration 中 pause、slow motion、hesitation、operator style 产生的时间参数化，本身会不会制造 non-Markovian contradiction？

这条已经被 IROS 2026：

## Improving Robotic Imitation Learning via Trajectory Standardization (ISR)

做得很深。

ISR 直接把 variable operator speed、intermittent pauses、inconsistent action density 作为 imitation 数据问题，并用 information-geodesic trajectory resampling 去掉低价值冗余、保留高加速度/关键操控阶段。

结果：

- π0.5 平均 success 从约 47.8% → 71.8%；
- VO-DP 也大幅提升；
- mixed-operator 场景特别明显；
- 数据量和训练成本还下降。

这说明“human time parameterization hurts BC”已经不是空白。

Action-chunking 与 ISR 的交叉机制也许仍然有空间，但因为两个邻边都很新、很强，不能作为首选。

**结论：降级。**

---

# 7. 多模态 native-rate / stale vision——直接撞 DAM-VLA

曾考虑：

> camera 低频、有 latency，而 proprio / force / action 高频。如果训练 pipeline 强制把它们放在同一个 timestep，所谓 proprio shortcut 是否部分来自“proprio 更及时”，而不只是它语义更强？

这个问题非常自然，也确实 embodied-specific。

但 2026-06 的：

## DAM-VLA: Decoupled Asynchronous Multimodal VLA

已经把这个 mismatch 作为核心问题：

- vision 低频；
- proprio / force 高频；
- synchronous VLA 要么 oversample slow modality，要么 undersample fast modality；
- 采用 per-modality latent buffer + native-rate update。

七个真实 contact-rich tasks 上，异步模型平均约 95.2%，最强同步基线约 40.95%。

**结论：正面做掉，砍。**

---

# 8. q1/q99 action normalization clipping——真实坑，但目前太像工程问题

很多 VLA pipeline 对 action / proprio 使用 `q01/q99` 归一化，并 clip 到 `[-1,1]`。

曾考虑：

> rare high-magnitude corrections / recovery / contact actions 会不会恰好落在 distribution tail，被 quantile clipping 系统性抹掉？

这有潜在方法口：tail-aware normalization / event-aware scaling。

但 OpenVLA-OFT 的 ALOHA 文档已经明确警告：absolute joint-angle action **不能**使用 q1–q99 clipping，因为会让某些解决任务必须达到的 joint angles 根本无法输出。

因此“clipping can delete crucial actions”已经是已知工程事实。

如果未来不能证明：

> 在主流 delta-action VLA 中，被裁掉的 tail 系统性对应 recovery / failure correction，而这构成一个主要闭环瓶颈，

就没有研究价值。

**当前评级：低优先级，不注册。**

---

# 9. VLM semantic invariance vs robot geometry——问题重要，但极度拥挤

曾考虑一个更 foundation-level 的 tension：

> VLM pretraining 希望对很多视觉变化保持 semantic invariance；机器人 action 却要求精确的 3D geometry / spatial equivariance。VLA fine-tuning 到底在哪里、以什么代价重新长出控制所需的几何？

但 2026 已经有一整片工作：

- GEAR-VLA；
- VGA（Vision-to-Geometry）；
- VIPA-VLA；
- FALCON；
- SG-VLA；
- VisualThink-VLA；
- CVPR 2026 `VLA Models Are More Generalizable Than You Think`。

其中后者甚至已经把 viewpoint brittleness 主要定位成 Spatial Modeling misalignment，并用极小的 visual-token adaptation 恢复 generalization。

**结论：超级拥挤，砍。**

---

# 10. action loss 与闭环重要性不一致——也已有直接工作

曾考虑：

> 训练时各 action groups / dimensions 在归一化空间里基本等权，但闭环中不同 group 的物理重要性差很多；总 action MSE 最优 checkpoint 是否可能不是最好 policy？

这条也已经有非常直接的 2026 工作：

## Per-Group Error, Not Total MSE

ICRA 2026 workshop。

它报告：

- total MSE 最低的模型并不是 robot rollout 最好；
- arm-group MSE 反而正确预测实机排名；
- 作者明确把 group-weighted / scheduled action loss 作为后续研究。

因此 generic loss-weighting 题也不再拿。

**结论：砍。**

---

# 11. WAM success-only optimism——刚被 MiraBench + FACT 从现象做到方法

这是一个很典型的“如果我们慢一周就会撞车”的例子。

曾考虑：

> WAM 多数使用成功 expert demonstrations。action 与成功 future 高度共现。坏 action 时，world model 会不会不是模拟失败，而是自动 pattern-complete 成成功结果？

这个问题非常强：如果成立，直接有 failure/counterfactual data 的方法口。

但：

## MiraBench

已经系统发现 current robotic world models 存在 pervasive optimism bias：坏 action 下仍预测成功 future。

## FACT: Failure-Aware Causal Training for World-Action Models

2026-08 最新工作更进一步，直接把根因指向 success-heavy training：

- success-only WAM 对 bad action hallucinate successful grasp；
- 加入 failure rollouts 后，failure future prediction PSNR 提升约 +6.4 dB；
- success future 不受损；
- 真实任务 success 同时提升。

这已经完整走完：

```text
现象 -> causal explanation -> failure-aware method
```

**结论：直接砍。**

---

# 12. 搜索重心转向 VLA / flow policy 的学习过程

到第三轮为止，一个模式已经非常明确：

> **最终模型的显性缺陷，2026 基本都有方法论文。**

所以开始把重点转向：

- SFT / RL 过程中行为模式如何变化；
- 哪些能力是重新加权，哪些是真的新出现；
- 哪些已有 representation 被 unlock / recombine；
- flow policy 的 exploration 与 action-mode structure 如何互动。

这里更接近“learning dynamics / mechanism”，也更符合后续方法空间。

但 flow-RL **算法本身**已经非常拥挤：

- FPO / Flow Policy Gradients；
- ReinFlow；
- πRL；
- FlowDPG；
- Q-VGM；
- ForesightFlow；
- StructRL。

所以不做“再发明一个 flow RL optimizer”。

---

# 13. 新的强备选：RL 到底是在发现新行为，还是释放 / 重组已有行为？

SimpleVLA-RL 给出了一个很漂亮、已经存在的现象：

## Pushcut

在 RoboTwin2.0 的 `move can pot`、`place A2B` 等任务里，所有 demonstration 都采用：

```text
grasp -> lift/move -> place
```

但 RL 后 policy 会自行发现：

```text
push / drag -> target
```

也就是监督数据中没有的 `pushcut` shortcut。

这不是我们假设出来的 anomaly，而是已发表论文明确报告的 emergent behavior。

但同一篇论文还有另一个同样重要的结果：

- 0-trajectory SFT：初始 success=0，RL 后仍然 0；
- 100 demonstrations：SFT avg 7.3%，RL → 25.4%；
- 1000 demonstrations：SFT avg 28.2%，RL → 50.4%。

换句话说：

> RL 可以产生 demonstration 中没有的新策略，但又极度依赖一个已经有基本 task ability 的 prior。

这让“RL simply creates new capabilities”变得不够准确。

一个更深的问题是：

> **RL 发现的所谓新策略，到底是通过 interaction 真正学出了新的 motor capability，还是把 pretraining / SFT 中已经存在、但没有在当前 task behavior 中表达的 primitive 重新组合 / 解锁出来？**

这件事如果能回答，会直接影响 embodied RL 的方法方向：

- 如果主要是 **reweight/recombine existing repertoire**，重点应该放在保留和搜索 pretrained behavior modes；
- 如果真的可以 **create new motor modes from reward interaction**，重点才应该放在 action-space exploration / online skill acquisition。

## 为什么这条目前还不能注册

最大的危险是 identification。

“新 behavior 是否已经 latent 存在”非常容易变成不可证伪的故事。为了证明“已有但未表达”，不能依赖：

- 随便做一个 probe；
- 找最近 neighbor action；
- 在巨大 latent space 做 subspace matching。

否则又会走 Topic 05/09 那种复杂 gate 路线。

需要找到一个**行为上就能识别的自然实验**，例如不同来源 prior 在相同 SFT foothold / reward 下对同一种 emergent strategy 的出现概率产生可预言差异，而不是先发明 latent metric。

目前还没有找到足够干净的 identification，因此：

**当前评级：高 interestingness，机制很值得，方法空间强，但 identification 尚未过关。继续深挖，不注册。**

---

# 14. 第三轮后的排序

| 搜索线 | 已有真实现象 | collision | 机制后 method 空间 | identification 风险 | 当前状态 |
|---|---:|---:|---:|---:|---|
| action chunk non-Markovianity：task memory vs teleop/data delay | 强 | 中高 | 强 | 中 | **强备选** |
| RL emergent behavior：new capability vs latent repertoire recombination | 强 | 中 | 很强 | **高** | **强备选，需找干净实验轴** |
| action chunk tail future-feedback supervision | 中强 | 高 | 强 | 中 | 降级 |
| demonstration time-parameterization / pause | 强 | 高（ISR） | 强 | 低 | 降级 |
| q99 tail clipping critical recovery actions | 工程证据强 | 中 | 中 | 中 | 低优先级 |
| generic flow-RL credit / optimizer | 强 | 极高 | 强 | 低 | 砍 |
| WAM success optimism | 极强 | 已被 FACT 做完 | 强 | 低 | 砍 |
| multimodal rate mismatch | 极强 | DAM-VLA 正面做完 | 强 | 低 | 砍 |
| VLM spatial grounding | 极强 | 极高 | 强 | 低 | 砍 |

---

# 15. 下一轮搜索原则

第四轮重点不再扩大关键词，而是围绕两个强备选寻找**能让问题变干净或直接把它砍掉的证据**：

### A. action chunk × demonstrator timing

要找：

- 同任务的人类 teleop vs scripted/autonomous expert；
- 已知不同 latency / control frequency 的 demonstration；
- 是否已有工作在 re-alignment 后测 delayed-policy advantage；
- 是否能不训练大 VLA 就先用 offline prediction 看到核心效应。

如果只能靠人工构造很多 delay controls才能成立，就砍。

### B. RL emergent behavior source

要找：

- 是否有多篇 VLA/robot RL 独立报告 SFT 数据外的新策略；
- 是否已有工作证明这些策略来自 pretrained repertoire；
- 有没有自然的 behavior-level intervention，可以区分 `recombination` vs `de novo acquisition`；
- 能否从 released checkpoints / rollouts 先做小规模 existence test。

如果“latent ability”只能靠 arbitrary representation metric定义，就砍。

继续搜索。