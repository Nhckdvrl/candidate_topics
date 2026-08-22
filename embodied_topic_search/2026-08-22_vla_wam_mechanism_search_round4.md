# 2026-08-22：VLA / WAM 机制选题搜索日志（第四轮）

> 状态：**继续搜索，仍不注册新 Topic。**
>
> 第三轮留下的最强备选是 `action chunking × demonstrator timing`。第四轮没有沿着它继续包装，而是重新做了一次 collision-first 审计，并把材料源从论文扩展到：最新开源整模型、官方技术博客、GitHub / dataset documentation、机器人数据生成系统、offline RL / causal imitation learning、以及 self-consuming synthetic data 等相邻领域。
>
> 本轮最重要的结果不是“把原题证明得更漂亮”，而是：**原来的窄题明显降级；同时从它下面长出了一个更一般、也更贴近当前 foundation robot data reality 的机制问题。**

---

# 0. 本轮重新设定的问题

第三轮的问题是：

> Action chunking 到底是在建模 task 的 non-Markovianity，还是在吸收 human demonstrator 的 reaction / teleoperation delay？

这条的第一枪看上去非常漂亮：robomimic 同一 simulator / task / robot / observation / action space 下，比较 Proficient-Human（PH）与 Machine-Generated（MG）的 delayed-prediction curve。

但第四轮开始后，先不跑实验，而是问三个更严格的问题：

1. **Aug-3 action-chunking mechanism paper 自己有没有已经削弱这条因果链？**
2. **“不同 data producer 有不同时间结构”是否已经被 2026 heterogeneous VLA data work 正面处理？**
3. **PH vs MG 真的是 clean natural contrast 吗，还是我们又把一个复杂 mixture 当成 Markov control？**

答案分别是：

- 是，Aug-3 paper 自己已经削弱“non-Markovianity 决定 chunking gain”的强版本；
- 是，physical-time / control-frequency temporal mismatch 已经被 ACE-Ego-0 等工作正面处理；
- 不是，robomimic MG 本身就是多个 SAC checkpoints 的 rollout mixture，因此不能简单当作单一 Markov demonstrator。

所以第四轮不能再写成“继续验证原题”，而必须重构问题。

---

# 1. 先把原来的 action-chunking 叙事降级

## 1.1 Aug-3 paper 已经把 action chunking 机制做得很深

核心论文：

**Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?**  
Filippo Lazzati, Kyle Stachowicz, William Chen, Alberto Maria Metelli, Andrew Wagenmaker, Sergey Levine  
arXiv:2608.02547, 2026-08-03  
https://arxiv.org/abs/2608.02547

这篇已经系统否掉了几个过去常见解释：

- temporal consistency；
- horizon reduction；
- generic representation-learning explanation。

并把 action chunking 的收益拆成：

1. non-Markovian expressivity；
2. 通过过去 observation 降低 compounding error；
3. 多个 temporal relationships 带来的 implicit ensembling。

最反直觉的现象仍然非常重要：

```text
预测当前 a_t
有时 o_{t-10} 比 o_t 更好用
```

在 LIBERO-90 上 delayed policy 可以 match / exceed ordinary action chunking；在 robomimic 和真机上，Randomized Delay Ensemble 又能补回单 delayed policy 缺掉的 implicit ensemble benefit。

这说明“过去 observation 为什么这么有用”仍然值得研究。

但一个关键结果会直接限制我们的原叙事：

> 论文跨 LIBERO task 比较 demonstrator non-Markovianity 与 action-chunking success gain，二者相关性并不强。

也就是说，不能把下面这条链当作默认机制：

```text
human temporal delay / non-Markovianity
        -> delayed-prediction advantage
        -> action chunking gain
```

前两项可以存在，但第三项还受 compounding error / implicit ensemble 等因素控制。

### 结论

如果我们最后只做：

> human delay 更大，所以 action chunking 更有用

很可能直接被这篇 paper 自己的 task-level correlation analysis 反杀。

因此本轮不再把“解释 action chunking 为什么有效”当主贡献。

---

## 1.2 Tri-Manual 已经直接证明：teleop interface 会把不属于 task 的 delay 写进 demonstration

**Tri-Manual Visuomotor Imitation Learning of Robot Policies**  
arXiv:2607.25731, 2026-07-28  
https://arxiv.org/abs/2607.25731

这篇提供了非常强的真实证据：

- 三臂机器人理论上可以同时动作；
- 人只有两只手；
- teleoperation interface 必须 pairwise mode switching；
- 本来可以同步的三臂动作因此被示教成串行；
- BC 会把这种由 interface 强加的 delay 复制进部署行为。

作者因此对 demonstration offline retiming，再训练同步 policy。

这个结果非常重要，因为它证明：

> **demonstration temporal structure 确实可以来自 data-collection interface，而不是 task dynamics。**

但它也意味着“demonstration delay may be artifact”这个一级现象已经不能作为我们的 novelty。

---

## 1.3 ACE-Ego-0 已经处理了 heterogeneous source 的 physical-time mismatch

**ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining**  
arXiv:2606.17200  
https://arxiv.org/abs/2606.17200

官方 repo：  
https://github.com/ACERobotics-VLA/ACE-Ego-0

它明确把 human / robot / sim joint training 的 mismatch 分成：

- spatial mismatch；
- embodiment mismatch；
- temporal mismatch；
- supervision-quality mismatch。

对 temporal mismatch，它不是固定预测同样 step 数，而是做 **time-aligned action chunking**：

\[
H_d = \mathrm{round}(f_d T^*)
\]

也就是不同 control frequency 的 dataset 使用不同 step horizon，但对应相同的 physical-time horizon。

所以：

> “不同数据源不应该共享固定 H=16/32/50，因为 control frequency 不一样”

已经不能拿。

### 这里留下的空白是什么？

ACE-Ego-0 对齐的是：

```text
step horizon -> physical duration
```

它没有回答：

```text
当前 action 真正依赖过去哪一段 observation history？
```

也就是说，**rate/horizon normalization** 和 **decision-time / causal-support structure** 不是一件事。

这一点成为本轮后续重构的关键。

---

# 2. PH vs MG 这个“完美 natural contrast”其实没有想象中干净

原先的想法是：

```text
PH = human teleoperator
MG = autonomous SAC policy
```

于是把 MG 当作接近 Markov control。

但 robomimic MG 的真实构造不是“一个固定 SAC expert”。

robomimic / 后续 offline-RL 文献对 MG 的描述是：

- 先训练 SAC；
- 保存多个 training checkpoints；
- 从多个 checkpoint rollout；
- 混成一个包含不同质量、不同 policy stage 的 dataset。

所以 MG 实际上更像：

\[
D_{MG} = \bigcup_j D_{\pi_{SAC}^{(j)}}.
\]

即使每个 `π_SAC^(j)(a|s)` 单独是 Markov，**把多个 policy identity marginalize 掉以后，aggregate dataset 也未必能由一个 stationary Markov behavior policy 描述。**

这直接暴露了一个比 human latency 更一般的问题。

---

# 3. 本轮真正出现的新母题

# **When a robot policy uses history, is it remembering the world — or identifying the data producer?**

中文：

> **机器人 policy 使用历史信息时，它到底是在记住世界 / task progress，还是在判断“这段轨迹是哪一种 demonstrator / behavior policy / strategy 产生的”？**

这不是语言游戏，而有一个很简单的统计结构。

假设环境 state 已经充分 Markov，并且每一个 producer 本身也是 Markov：

\[
\pi_s(a_t\mid o_t).
\]

其中 `s` 可以是：

- human operator；
- SAC checkpoint；
- autonomous specialist；
- synthetic trajectory generator configuration；
- intervention / correction policy；
- 不同 control strategy。

如果 dataset 把多个 producer 混起来，但是训练时不给 `s`，那么：

\[
p(a_t\mid o_t,H_t)
=
\sum_s p(s\mid o_t,H_t)\pi_s(a_t\mid o_t).
\]

历史 `H_t` 可以提高 action prediction，不是因为世界本身必须记忆，而是因为：

\[
H_t \rightarrow p(s\mid H_t)
\]

帮助推断当前轨迹属于哪一种 behavior mode。

这意味着一个非常重要、但容易被忽略的事实：

> **把多个 Markov policies 混在一起，本身就可能制造 aggregate non-Markovianity。**

所以现代 robot policy 的 memory capacity 可能同时承担两种功能：

```text
A. world / task memory
   - 哪个抽屉找过了
   - 物体刚才在哪里
   - 当前 subtask progress

B. producer / strategy memory
   - 这个 operator 通常怎么接近物体
   - 当前轨迹来自哪个 SAC checkpoint
   - 这个 autonomous specialist 的动作风格是什么
   - 当前 episode 属于 fast / slow / cautious / recovery strategy
```

目前我认为 B 是第四轮最值得继续追的机制。

---

# 4. 为什么这个问题现在特别 relevant：最新 VLA 正在同时加“memory”和“heterogeneous data”

## 4.1 MemoryVLA / MemoryVLA++

**MemoryVLA**：  
https://arxiv.org/abs/2508.19236

**MemoryVLA++**：  
https://arxiv.org/abs/2606.09827

它们都把 temporal context / memory 解释成机器人任务本身的 temporal dependency：历史 interaction、episodic detail、semantic gist、未来 imagination。

这是合理的，但没有分解 memory 到底在利用哪些统计信号。

---

## 4.2 HAMLET

**HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy**，ICLR 2026。

公开实现已经覆盖 NVIDIA GR00T N1.5 / N1.6：  
https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T

HAMLET 用 moment tokens + block-causal memory transformer，把 pretrained Markov VLA 改成 history-aware policy。

同样，主叙事是“记住过去以解决 memory task”。

---

## 4.3 BPP：history 本身会带来 spurious correlation

**BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames**  
arXiv:2602.15010  
https://arxiv.org/abs/2602.15010

project：  
https://bigpicturepolicies.github.io/

这篇对我们非常重要，因为它已经证明：

> naive history conditioning 会因为训练 history coverage 不足而 latch onto spurious correlations，甚至比 current-observation policy 更差。

它提出只保留 task-relevant key history frames。

这意味着“history 里混入不该依赖的东西”不是拍脑门风险，而是已存在的真实问题。

但 BPP 主要关注的是：

- exponentially large history space；
- incidental features；
- OOD history coverage。

它没有把 **producer identity / behavior strategy** 作为一个具体的、可因果检验的 history shortcut 单独拆出来。

---

## 4.4 LaMem-VLA / G0.5：长历史已经进入最新整模型

**LaMem-VLA**：  
https://arxiv.org/abs/2607.07608

它把 short-term + long-term historical experience 重建成 latent memory tokens，并直接编入 VLA reasoning/action stream。

**G0.5: One Autoregressive Stream for Robot Reasoning and Action**：  
https://arxiv.org/abs/2608.11739

官方开源实现 / checkpoint：  
https://github.com/OpenGalaxea/GalaxeaVLA

G0.5 已经把 multi-second visual history 作为 foundation VLA 的原生组成部分，并且 release 了 pretrained checkpoint、DROID / LIBERO / RoboTwin / real-robot 入口。

这说明“robot policy 是否应该有 history”已经不是小架构讨论，而是在进入 foundation-model default design。

---

# 5. π0.7 提供了一个非常关键的工业级旁证，同时也是 collision

Physical Intelligence 官方博客：

**π0.7: A Steerable Model with Emergent Capabilities**  
https://www.pi.website/blog/pi07

paper：  
https://www.pi.website/download/pi07.pdf

它直接说了一件非常关键的事：

> broad robot data、human data、autonomous episodes 不能 naive merge；需要更丰富的 context 去 disambiguate 不同 strategy / behavior / proficiency。

π0.7 的 prompt 除了 task instruction，还加入：

- subtask language；
- subgoal image；
- speed metadata；
- quality metadata；
- control modality；
- observation memory。

官方博客甚至明确解释：这些 context 的作用之一就是 **disambiguate the behavior**，让不同 strategy、不同 proficiency、不同 autonomous data 可以共同训练。

并且 π0.7 已经把 RECAP specialist 产生的 autonomous experience 蒸馏回一个 generalist model，通过 strategy / quality metadata 保留行为差异。

## 对我们意味着什么

这是双刃剑。

### 支持

producer / strategy heterogeneity 确实已经是 frontier foundation robot model 的真实训练问题，不是 robomimic toy artifact。

### collision

所以我们不能把方法贡献写成：

> “给不同 demonstrator 一个 source ID / metadata。”

π0.7 已经在更一般的规模上这么做了。

如果这个题要成立，贡献必须是：

> **解释 memory/history 在 heterogeneous robot data 中到底学到了什么，并定量分解 task-state memory 与 producer/strategy inference。**

然后再由这个机制导出新的 temporal architecture / objective，而不是简单 metadata conditioning。

---

# 6. 相邻领域 collision：数学思想不是新的，robot mechanism audit 才可能是新的

这里必须把 novelty bar 写清楚。

## 6.1 Offline RL 已经明确说过：history 可以帮助推断 behavior policy identity

**Semi-Supervised Offline Reinforcement Learning with Action-Free Trajectories**  
ICML 2023  
https://proceedings.mlr.press/v202/zheng23b.html

这篇在讨论多个 behavior policies 时明确指出：

> 即使多个 behavior policy 本身是 Markovian，一段过去 state history 也比单个 state 更容易推断实际是哪一个 behavior policy 产生了当前 trajectory。

所以我们不能声称：

> “发现 mixture of Markov policies can look non-Markovian。”

这个统计事实本身不是 novelty。

---

## 6.2 Multi-human imitation 早就知道 demonstrator heterogeneity / incompatibility

**Eliciting Compatible Demonstrations for Multi-Human Imitation Learning**  
CoRL 2022  
https://proceedings.mlr.press/v205/gandhi23a.html

它明确指出不同 human demonstrator 可以采用互相冲突但都合理的 action modes，并设计 compatibility metric 与 active elicitation，避免把不兼容 demonstration 混在一起。

所以：

> “multi-human data 有不同 style，会让 BC 难学”

也不是 novelty。

---

## 6.3 Causal imitation learning 已经研究 hidden expert-observable factors

**Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding**  
ICLR 2026。

相关 preprint：  
https://arxiv.org/abs/2502.07656

它利用 trajectory histories 作为 instruments，并允许 history-dependent policy 去推断 expert-observable hidden variables。

所以从 causal-IL 理论上说，“expert 有 learner 看不到的 latent factor，history 可以提供信息”也已经有体系。

---

## 6.4 Re-Mix 已经说明 robot dataset mixture 本身是重要 optimization object

**Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning**  
https://arxiv.org/abs/2408.14037

它直接优化 robotics dataset domain mixture weighting，在 OXE 上证明数据权重会极大影响 downstream policy。

因此我们不能把结论停在“mixture matters”。

---

# 7. 所以真正可能的新贡献必须达到这个强度

不能是：

> history 能预测 operator。

不能是：

> multi-human 比 single-human 更难。

不能是：

> 给 source ID 会变好。

而应该是：

> **当前被解释成 task/environment memory 的 robot-policy history gain，其中有多少其实来自 latent data-producer / strategy inference？这种 producer-induced memory 是否会因果改变 closed-loop action？**

换句话说，真正的 decomposition 是：

\[
\text{history gain}
=
\text{task-state memory}
+
\text{producer/strategy disambiguation}
+
\text{other temporal regularization}.
\]

我们要问的是第二项到底是不是一个真实、足够大的 component。

---

# 8. robomimic 给了一个比 PH-vs-MG 更干净的第一枪

robomimic Multi-Human（MH）数据：

- 6 个 operator；
- 2 个 worse；
- 2 个 okay；
- 2 个 better；
- 每个 operator 50 条 demonstration；
- dataset 公开保留逐 operator masks：
  - `worse_operator_1`
  - `worse_operator_2`
  - `okay_operator_1`
  - `okay_operator_2`
  - `better_operator_1`
  - `better_operator_2`

TensorFlow Datasets 也公开了这些 mask：  
https://www.tensorflow.org/datasets/catalog/robomimic_mh

robomimic 官方 study：  
https://robomimic.github.io/study/

所以不需要自己从 trajectory clustering 猜 demonstrator identity。

---

# 9. 第一枪应该怎么做：不是“RNN vs MLP”，而是 producer-mixture decomposition

## 9.1 先在同一 proficiency 内混 producer

最干净的 contrast 不是直接：

```text
1 operator vs all 6 operators
```

因为那会同时引入：

- producer identity；
- proficiency / quality；
- strategy diversity。

先做：

```text
better_operator_1
better_operator_2
better_1 + better_2
```

然后同样做：

```text
okay_1 / okay_2 / okay_1+2
worse_1 / worse_2 / worse_1+2
```

这样最先问的是：

> **在 quality stratum 基本固定时，仅增加 producer mixture，会不会增加 history 的价值？**

如果这个都没有，就没有必要跑更大模型。

---

## 9.2 样本量必须匹配

例如固定每个训练集 50 trajectories：

```text
single producer: 50 from one operator
2-producer mix: 25 + 25
6-producer mix: approximately 8/9 each
```

不能用：

```text
50 single vs 100 pair vs 300 all
```

否则 history gain 与 data quantity 又混起来。

---

## 9.3 四个最关键模型条件

在同一 backbone / parameter-budget 下至少比较：

### M0 — Markov

\[
a_t \sim \pi(o_t)
\]

### MH — History

\[
a_t \sim \pi(o_{t-L:t})
\]

### M0+ID — Markov + true producer ID

\[
a_t \sim \pi(o_t,s)
\]

### MH+ID — History + true producer ID

\[
a_t \sim \pi(o_{t-L:t},s)
\]

另外必须有：

### M0+RandomID

给随机打乱的 producer label，确认提升不是多了 embedding / capacity。

第一轮完全可以 low-dimensional state 开始，不需要 image encoder，不需要 VLA。

---

# 10. 核心量不是 raw RNN gain，而是 mixture-induced extra history gain

对 producer `s` 定义：

\[
G_s = L(M0,s)-L(MH,s)
\]

对 mixture `S` 定义：

\[
G_{mix}=L(M0,S)-L(MH,S).
\]

真正关心：

\[
\Delta_{mix}
=
G_{mix}
-
\mathbb{E}_{s\in S}[G_s].
\]

如果：

\[
\Delta_{mix} \gg 0
\]

说明增加 producer mixture 本身额外增加了 history 的价值。

然后看 producer ID 能不能解释它：

\[
R_{ID}
=
L(M0,S)-L(M0+ID,S).
\]

如果：

```text
history gain grows after mixing producers
AND
true producer-ID recovers a large fraction of that extra gain
AND
random ID does not
```

这就是非常直接的第一层证据：

> history 至少部分在做 producer / strategy disambiguation。

---

# 11. 但是 offline action loss 仍然不够，必须闭环

这是从仓库前面失败题学到的教训。

如果结果只是：

> MSE lower with history / ID

没有意义。

下一步必须 rollout，比较：

- task success；
- trajectory efficiency；
- action smoothness / pause structure；
- strategy consistency；
- error-recovery behavior。

最强结果是：

```text
mixed producers
    -> Markov policy behavior degrades
    -> history restores behavior
    -> true producer/strategy conditioning removes much of the need for history
```

并且这个变化在 closed-loop success / behavior mode 上成立。

这才不是普通 supervised-learning observation。

---

# 12. 一个非常重要的 causal intervention：history swap

如果第一层成立，可以做一个比 probe 更强的 intervention。

找到当前 observation 相近、但来自不同 producer / strategy 的两个 trajectory prefix：

```text
current o_t approximately matched
history H_A from producer A
history H_B from producer B
```

给同一个当前 observation，替换 history：

\[
\pi(a_t\mid o_t,H_A)
\quad\text{vs}\quad
\pi(a_t\mid o_t,H_B).
\]

如果 action 系统性切向 A / B 各自的 characteristic strategy，而 environment current state 基本保持一致，那么这比“history feature 能 decode operator”强得多。

我们真正要的是：

> **producer information in history causally steers action.**

而不是 probe accuracy。

这也符合 Round 2 已经确定的 mechanism bar：representation evidence 必须尽量走到 intervention。

---

# 13. 这条题的 kill line 应该非常简单

不要设计十个 gate。

## Kill A：producer mixture 不增加 history gain

在 matched-data、within-proficiency setting：

\[
\Delta_{mix}\approx 0.
\]

直接砍 producer-induced non-Markovianity 作为主机制。

---

## Kill B：history 有用，但 true producer ID 完全解释不了

如果：

```text
MH >> M0
but
M0+ID ≈ M0
```

说明 history 的价值更可能来自 task progress、human reaction、partial observability、phase 等，不是 producer identity。

主题降级。

---

## Kill C：效应只来自 proficiency difference

如果 all-6 mixture 有明显效应，但：

```text
better1 + better2
okay1 + okay2
worse1 + worse2
```

都没有，那么只是 quality mixture / suboptimal action modeling。

这已经被 π0.7 quality metadata、offline RL、Re-Mix 等大量工作覆盖，砍。

---

## Kill D：只有 prediction loss，没有 rollout consequence

如果 offline loss 分解很漂亮，但闭环 success / strategy / recovery 基本不变，不能把它升格成机器人机制题。

砍。

---

# 14. 如果第一枪真的很强，下一层验证轴是天然存在的

## 14.1 MG：producer 从 human operator 换成 SAC checkpoint

robomimic MG 很适合做第二个 axis：

```text
producer = SAC checkpoint identity
```

它可以检验这个机制是不是 human-specific。

如果：

- human operators mixture 有；
- SAC checkpoint mixture 也有；

那结论就明显更一般：

> **non-Markovianity can be a property of dataset aggregation, not a property of human cognition.**

这比“human reaction delay”强很多。

---

## 14.2 MimicGen：可控 synthetic producer

MimicGen 官方公开了：

- core datasets；
- robot variants；
- `large_interpolation` datasets。

官方 documentation 特别指出：`large_interpolation` 会给 imitation learning 带来显著困难。

https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html

生成代码里 `num_interpolation_steps` 又是显式可控参数。

https://github.com/NVlabs/mimicgen

2026 的：

**MinInter: Minimizing Trajectory Interpolation During Data Augmentation for Imitation Learning**  
https://arxiv.org/abs/2606.24078

进一步证明：减少 synthetic trajectory 中的 non-expert interpolation segment，可以同时提高 data-generation success 与 policy success。

### 这里不能拿什么

不能拿：

> synthetic interpolation artifacts hurt imitation.

已经被做了。

### 可以拿它干什么

MimicGen 是一个**可控 producer mechanism**：

我们可以改变 trajectory generator 的 interpolation process，然后测：

- temporal-support kernel 是否变化；
- history gain 是否变化；
- 与 human/operator mixture 的 signature 是否相似。

也就是把 MimicGen 当 intervention platform，不把 interpolation artifact 本身当论文贡献。

---

# 15. Producer-dependent temporal support：保留，但从主标题降成机制轴

原来的 scalar delay：

\[
d^*=\arg\min_d L(a_t\mid o_{t-d})
\]

太容易把很多东西压成一个数。

更一般地，可以定义 producer-specific temporal support：

\[
K_s(\tau)
\]

表示 producer `s` 的当前 action 对过去 physical-time offset `τ` 的信息依赖。

这和 ACE-Ego-0 的 time-aligned chunking 不同：

- ACE-Ego-0：不同 dataset 的 `H steps` 对齐到同一 physical duration；
- 我们这里：即使 physical duration 已对齐，不同 producer 的 **decision support** 可能仍然不同。

例如：

```text
human teleop:
  delayed / history-heavy support

autonomous reactive controller:
  support concentrates near tau=0

MimicGen interpolation:
  artificial deterministic temporal structure

RL specialist:
  another strategy-specific support
```

如果 A 题成立，`K_s(τ)` 可以成为解释 producer mixture 如何制造 history dependence 的具体机制对象。

但当前不把它单独注册成题，因为：

- scalar delay 已离 Aug-3 paper 很近；
- physical-time mismatch 已被 ACE-Ego-0 占；
- human timing artifacts 已被 Tri-Manual / ISR / latency-aware teleop literature 大量碰过。

**当前定位：A 题的机制测量轴，而不是独立主贡献。**

---

# 16. 第二条新搜索线：autonomous data 会不会形成 self-consuming behavioral loop？

这条是从 π0.7 官方博客、RECAP、ROVE、自改进 embodied foundation model 以及相邻生成模型文献共同长出来的。

当前 frontier robot learning 越来越常见：

\[
\pi_k
\rightarrow
\text{autonomous rollouts}
\rightarrow
D_k
\rightarrow
\pi_{k+1}.
\]

### 现实证据

π0.7 官方明确使用：

- demonstration data；
- autonomous data；
- mixed-quality data；
- RECAP specialist experience；

并通过 metadata 把这些 experience distill 回 generalist policy。

ROVE（2026）：  
https://arxiv.org/abs/2606.17011

做 repeated rollout → human intervention → RL improvement。

**Self-Improving Embodied Foundation Models** 也直接让 robot autonomous practice，再用 self-predicted reward / success detector 更新 policy。

**RISE**：  
https://arxiv.org/abs/2602.11075

则在 world-model imagination 中做 self-improving rollout loop。

所以“机器人数据越来越由旧 policy 自己生产”是已经发生的趋势。

---

# 17. 从生成模型迁移来的问题：behavioral-support ratchet

生成模型领域已经系统研究：

> 后一代模型不断训练在前一代模型生成的数据上，会不会丢掉原始 distribution 的 tail？

代表工作：

**AI models collapse when trained on recursively generated data**  
Nature 2024  
https://www.nature.com/articles/s41586-024-07566-y

**Self-Correcting Self-Consuming Loops for Generative Model Training**  
ICML 2024  
https://proceedings.mlr.press/v235/gillman24a.html

**Self-Consuming Generative Models with Adversarially Curated Data**  
ICML 2025  
https://proceedings.mlr.press/v267/wei25o.html

这里不能直接把“model collapse”复制到 robotics。

机器人 policy 数据有关键不同：

- environment transition 会过滤 infeasible action；
- success / reward / intervention 会 curate rollout；
- RL 不是纯 density matching；
- autonomous policy 可以探索到 human demo 没有的新行为。

但一个更 embodied-specific 的问题很自然：

> **当每一代 policy 主要在学习上一代 policy 能访问到的 state-action support 时，rare recovery mode、低概率成功路径、失败边界附近的行为，会被逐代扩展还是逐代消失？**

可以叫：

## **Does autonomous robot data broaden behavioral support, or ratchet the policy into its own repertoire?**

这个问题的意义比普通 “self-training may collapse” 更具体：

foundation VLA 现在正把 specialist/autonomous experience 大量蒸馏回 generalist；如果 data producer 本身不断变化，training distribution 已经不再是外生的。

---

# 18. 为什么这条暂时还不升格

它的 interestingness 很高，但 identification 比 A 难。

需要先有真实 evidence：

- 多轮 autonomous collection 后 behavior diversity / recovery support 是否系统收缩；
- 或者相反，RL exploration 是否稳定扩大 support；
- rare modes 的变化是否影响 final policy，而不是 sampling noise。

目前公开工作更多在展示：

> self-improvement 能提高 success。

还没有找到足够直接的论文证明：

> successive autonomous-data generations 存在可重复的 support contraction / ratchet signature。

如果我们现在就设计五轮 training loop 去赌，又会回到 Topic 09 的错误：

> 概念漂亮，但 prerequisite 没有被证明。

因此本轮只记录为**高上限新搜索线**，不注册。

---

# 19. 第四轮当前排名

| 排名 | 方向 | 已有真实现象 | collision | 第一枪 | identification 风险 | 当前决定 |
|---|---|---:|---:|---:|---:|---|
| **A** | **Memory is modeling the world or the producer? / mixture-induced non-Markovianity** | 强：multi-human heterogeneity、history gains、frontier mixed-source training | 中高，但“memory causal decomposition”尚未看到正面完成 | **很干净：robomimic operator masks** | 中 | **当前最强备选，值得先做 cheap prerequisite test** |
| **B** | producer-dependent causal temporal support `K_s(τ)` | 强：Aug-3 delay anomaly、teleop timing、heterogeneous sources | 高 | 干净 | 中 | **作为 A 的机制轴保留，不单独注册** |
| **C** | autonomous self-consuming data / behavioral-support ratchet | 趋势强，具体 anomaly 证据不足 | 中 | 第一枪尚未足够便宜 | 高 | **高上限搜索线，继续 collision / existence search** |
| **D** | action chunking × human reaction / teleop delay | 强 | **很高** | 很便宜 | 低中 | **明显降级，不再作为主标题** |

---

# 20. A 题为什么暂时比第三轮原题更强

第三轮原题：

> action chunking 到底建模 task 还是 demonstrator？

现在更一般的问题：

> **robot temporal architecture 到底在建模 environment，还是在建模 training-data generator？**

这里 action chunking、RNN、VLA memory、long-context attention 都只是不同 temporal architectures。

它覆盖：

```text
human teleoperation
machine-generated SAC
MimicGen / synthetic data
autonomous RL specialist
intervention / recovery data
```

而不是只绑定一个 action-chunk paper。

更重要的是，它和现在 foundation robot learning 的真实数据趋势直接对齐：

```text
human data != autonomous data != synthetic data != intervention data
```

但现在越来越倾向于：

```text
all data -> one foundation policy with one temporal architecture
```

如果我们证明：

> history / temporal structure 在 mixed-source policy 中有一部分主要服务于 producer inference，

那么后面自然产生的 design question 才是：

> **task memory 与 producer/strategy memory 是否应该共享同一个 temporal pathway？**

这才有真正的方法空间。

可能的 knob 包括，但本轮不锁死：

- producer/strategy-aware temporal routing；
- producer-invariant task memory；
- source-dependent history support；
- explicit latent strategy variable；
- mixture-aware temporal objective；
- 将 task-event memory 与 behavior-style memory factorize。

注意：简单的 source-ID conditioning 因 π0.7 等工作已经不够新。

---

# 21. 本轮新增的一条筛题原则

## **Temporal memory 的解释必须先排除“dataset identity memory”**

以后看到一个 VLA memory paper：

```text
history -> success increases
```

不能直接接受：

> model learned task memory.

至少要问：

1. history 是否能识别 demonstrator / dataset / strategy？
2. 给 explicit strategy metadata 后，history gain 是否消失？
3. matched current state 下 swap history，会不会把 action 拉向另一种 producer behavior？
4. single-source data 上 memory gain 与 mixed-source data 上是否一样？

这和语言模型里的 document/source shortcut 类似，但在机器人里会直接改变 physical action，所以更需要 causal audit。

---

# 22. 下一步：不注册，先跑一个真正能杀题的 prerequisite

当前最合理的下一步不是写 VLA harness，而是给 A 题做一个极便宜的 existence test。

优先 task：

```text
robomimic Square MH
```

原因：

- multi-human heterogeneity 强；
- 单 operator mask 完整；
- low-dimensional state 即可先做；
- rollout environment 成熟；
- history-aware BC 在这个 setting 本来就有明显价值；
- 不需要下载 / 训练大 VLA。

建议 discovery 顺序：

```text
1. better_operator_1 vs better_operator_2 vs 25+25 mixture
2. okay pair
3. worse pair
4. all-6 matched-N mixture
5. true-ID / random-ID controls
6. only if offline + rollout both strong -> MG checkpoint axis
7. only then -> MimicGen / modern VLA
```

第一枪如果没有清楚的 `Δ_mix`，直接砍。

不允许因为题目听起来漂亮再加十个控制救它。

---

# 23. 本轮关键材料索引

## Action chunk / temporal structure

- Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?  
  https://arxiv.org/abs/2608.02547
- Tri-Manual Visuomotor Imitation Learning of Robot Policies  
  https://arxiv.org/abs/2607.25731
- ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining  
  https://arxiv.org/abs/2606.17200
- ACE-Ego-0 GitHub  
  https://github.com/ACERobotics-VLA/ACE-Ego-0

## robomimic / multi-producer data

- robomimic study  
  https://robomimic.github.io/study/
- robomimic dataset docs  
  https://robomimic.github.io/docs/datasets/overview.html
- TensorFlow robomimic MH dataset / operator masks  
  https://www.tensorflow.org/datasets/catalog/robomimic_mh
- Eliciting Compatible Demonstrations for Multi-Human Imitation Learning  
  https://proceedings.mlr.press/v205/gandhi23a.html
- Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning  
  https://arxiv.org/abs/2408.14037

## Memory / history VLA

- MemoryVLA  
  https://arxiv.org/abs/2508.19236
- MemoryVLA++  
  https://arxiv.org/abs/2606.09827
- HAMLET GR00T implementation  
  https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T
- BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames  
  https://arxiv.org/abs/2602.15010
- BPP project  
  https://bigpicturepolicies.github.io/
- LaMem-VLA  
  https://arxiv.org/abs/2607.07608
- G0.5  
  https://arxiv.org/abs/2608.11739
- GalaxeaVLA / G0.5 open model  
  https://github.com/OpenGalaxea/GalaxeaVLA

## Frontier system / practitioner source

- Physical Intelligence π0.7 official blog  
  https://www.pi.website/blog/pi07
- π0.7 paper  
  https://www.pi.website/download/pi07.pdf

## Synthetic data producer

- MimicGen  
  https://mimicgen.github.io/
- MimicGen dataset docs / large_interpolation  
  https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html
- MinInter  
  https://arxiv.org/abs/2606.24078

## Adjacent theory / causal IL

- Semi-Supervised Offline Reinforcement Learning with Action-Free Trajectories  
  https://proceedings.mlr.press/v202/zheng23b.html
- Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding  
  https://arxiv.org/abs/2502.07656

## Autonomous / self-generated robot data

- ROVE  
  https://arxiv.org/abs/2606.17011
- RISE  
  https://arxiv.org/abs/2602.11075
- π0.7 official blog（autonomous data + RECAP experience distillation）  
  https://www.pi.website/blog/pi07

## Self-consuming data analogy

- AI models collapse when trained on recursively generated data  
  https://www.nature.com/articles/s41586-024-07566-y
- Self-Correcting Self-Consuming Loops for Generative Model Training  
  https://proceedings.mlr.press/v235/gillman24a.html
- Self-Consuming Generative Models with Adversarially Curated Data  
  https://proceedings.mlr.press/v267/wei25o.html

---

# 24. 第四轮最终判断

这轮没有得到“终于可以马上注册”的结论，这是刻意的。

原来的：

> **Is Action Chunking Modeling the Task, or the Demonstrator?**

作为 action-chunk mechanism paper 已经**降级**：离 Aug-3 paper 太近，且它自己已经证明 non-Markovianity 不能单独解释 chunk gain；Tri-Manual、ACE-Ego-0 又分别占掉了 interface-induced temporal artifact 与 physical-time alignment。

但它暴露出的更底层问题反而更有潜力：

> ## **When a robot policy uses history, is it remembering the world, or identifying who generated the trajectory?**

这条目前没有被证明成立，所以还不能注册。

但它满足目前最重要的健康条件：

- 不是从零猜 anomaly；
- 有 robomimic multi-human / multi-policy mixture 的真实结构支撑；
- 有 frontier π0.7 对 heterogeneous strategy disambiguation 的现实压力；
- 有 BPP 对 history shortcut 的独立现象支撑；
- 有 offline-RL / causal-IL 理论说明机制是可能的；
- 第一枪不需要大模型；
- kill line 非常简单；
- 如果成立，能自然连接到现代 mixed human/autonomous/synthetic foundation robot training。

因此第四轮后的动作是：

> **先对 A 做一个小而硬的 prerequisite experiment；不为了保题去增加复杂 gate。**

如果 A 被砍，再继续搜索，而不是把标题改成另一个近义词。
