# 2026-08-22：VLA / WAM 机制选题搜索日志（第七轮）

> 状态：**继续搜索。没有为了数量新增候选。**
>
> Round 7 重点不是继续包装 A/B，而是沿 Round 6 留下的四条线重新广搜：foundation pretraining 的 qualitative transition、WAM 的 controllability / intervention / object dynamics、heterogeneous robot data 中被当作 preprocessing 的结构，以及最新整模型 technical report / blog 里“作者为了把模型跑起来修掉、但没有上升成研究问题”的异常。
>
> 本轮材料刻意混合 peer-reviewed paper、2026 arXiv、开源 repo、公司 technical report / blog、独立 practitioner notes，并额外从 LLM scaling / world-model / causal representation 的问题意识中寻找可迁移结构。

---

# 0. 先给结论

Round 7 没有找到一个比 A/B 更干净、且 collision 足够低的新 C。

但出现了三个重要变化：

1. **A 的 collision 风险显著上升。** IntentVLA 已经正面研究“相似当前观测对应不同 short-horizon intent，history 用来维持 episode 内行为模式”的机制。A 仍可保留，但不能再把 `history -> latent behavior mode / intent inference` 本身当 novelty。
2. **B 被 2026-08 的新 scaling 证据明显加强。** Dyna-2 把 human-video pretraining 从 1k h 扩到 1M h，并报告跨 embodiment robot performance 随规模提升；但 scaling 到底扩大的是 perception invariance、behavior repertoire、composition，还是 action synthesis，仍未被直接区分。
3. **WAM 这条线比 Round 6 更拥挤。** `Do WAMs need test-time imagination?`、`Does the action branch actually consume future states?`、`Are action-conditioned futures controllable?`、`What should the world branch predict?` 这些看起来非常自然的问题，在 2026-03 到 2026-08 已经被 Fast-WAM / Faster-WAM / RIFT / Ctrl-World / CoCo / AGRA / DreamWAM / SG-WAM / OA-WAM 等连续占掉。

因此当前最健康的动作不是硬凑 C，而是：

> **继续把 B 往 foundation scaling 的机制问题推，同时把 A 的 novelty bar 提高。**

---

# 1. A 的新 collision：IntentVLA 非常接近

A 当前问题：

> **What Does History Actually Model in Robot Policies?**

Round 7 新查到：

## IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation

- arXiv 2026-05: https://arxiv.org/abs/2605.14712
- repo: https://github.com/ZGC-EmbodyAI/IntentVLA

它的出发点几乎就是：

```text
same / similar current visual-language observation
    -> different valid action chunks across episodes
```

原因包括：

- different short-horizon intents；
- task phase；
- recent context；
- human-demonstration multimodality。

frame-conditioned VLA 每个 chunk 独立重采样 intent，会发生 inter-chunk conflict；IntentVLA 因而从 recent history 提取 short-horizon intent 表示来维持行为一致性。

这对 A 是一个非常实质的 collision，因为 A 原先的一条核心机制假设就是：

> history gain 的一部分可能来自 history 对当前 episode behavior mode / strategy / producer 的识别，而不只是物理世界状态记忆。

IntentVLA 已经把 `history -> latent intent -> consistent action mode` 做到现代 VLA 上了。

## A 还能不能活？

**还能，但 novelty 必须上移一层。**

不能再声称：

> history helps because it infers latent intent / behavior mode.

A 真正剩下的空间只能是更强的问题：

> **在现代 history-aware policies 中，history representation 到底有多少来自 task/world state，多少来自 training-data behavior structure；当两者冲突时，policy 依赖哪一个？**

也就是说，A 必须从“history 能消歧义”升级成：

```text
world-state memory
vs
behavior-source / strategy memory
```

并要求 counterfactual behavioral consequence，而不是只做 intent probe。

### 当前判断

**A 不砍，但风险从 medium 提到 high。**

如果后续再找到直接做 task-state-vs-behavior-source decomposition 的工作，A 应该立即移出 shortlist。

---

# 2. B 被 Dyna-2 强化：scaling 已经出现，但 action mechanism 仍然没闭合

B：

> **How Do Robot Foundation Policies Generalize Actions?**

Round 6 主要从 ICLR 2026 `Demystifying Robot Diffusion Policies` 出发：task-specific diffusion policy 更像 retrieval，ACT 更像 interpolation，pretrained GR00T 已明显不同但没有可靠 systematic extrapolation。

Round 7 最重要的新证据是：

## Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models

- Dyna Robotics, 2026-08
- announcement: https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-2-world-action-model-demonstrating-first-true-scaling-law-in-robotics-powered-entirely-by-human-data-302847114.html
- practitioner technical summary: https://github.com/geekyutao/dyna-2-summary

Dyna-2 做了一个非常干净的 nested data ladder：

```text
1k h -> 10k h -> 100k h -> 1M h
```

主要是 egocentric human manipulation video，固定 source proportions，并观察 held-out human prediction、zero-shot robot action prediction 和 post-trained real-robot performance。

最值得注意的不是“1M 小时”这个数字，而是两个现象：

### 2.1 cross-embodiment transfer 随 human-video scale 增长

Dyna 报告：human-video pretraining 规模增加时，对**从未在 pretraining 里出现过的 robot embodiment** 的 action prediction 与 real-robot post-training 结果整体提高。

### 2.2 future/video modeling 是 transfer scaling 的关键轴

在 action-labelled data 固定时，只增加 video/world-modeling data，zero-shot robot action prediction 继续改善。

但有一个很有意思的伴随现象：video scaling 对 held-out human action metric 并没有同样收益，甚至可能略有伤害；收益主要出现在 cross-embodiment robot transfer。

这说明 pretraining 确实产生了某种“不是单纯 in-domain behavior cloning”的 qualitative effect。

## 但它仍然没有回答 B

Dyna-2 证明的是：

```text
more human world-modeling data
    -> better transfer to robot control
```

它没有证明：

```text
more scale
    -> qualitatively new motor generation mechanism
```

例如同一个 improvement 仍可能来自：

1. **better perception / geometry invariance**：更准确地把新 robot scene 映射到已有 motion family；
2. **larger behavior repertoire**：百万小时数据本身覆盖了足够多 reusable physical patterns；
3. **better interpolation**：latent/action manifold 更稠密；
4. **composition**：把熟悉的 contact / reach / rotate / stabilize segments 重组；
5. **true extrapolation / synthesis**：产生训练 support 外的关键 motor behavior。

Dyna-2 当前并没有把这五种解释分开。

因此 Round 7 后，B 的重要性反而更高：

> **scaling law 已经开始被证明；下一层自然问题就是 scaling 到底改变了什么 computation。**

---

# 3. π0.7 也在加强 B，而不是替代 B

## π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

- blog: https://www.pi.website/blog/pi07
- arXiv: https://arxiv.org/abs/2604.15483

π0.7 明确 claim：

- unseen instruction following；
- new appliance tasks；
- zero-shot cross-embodiment；
- early signs of compositional generalization；
- new robot folding laundry despite no laundry-folding data for that robot。

关键训练思想是**diverse context conditioning**：不仅给 task language，还给 strategy / quality / speed / visual subgoals 等，使 heterogeneous demonstrations、autonomous rollouts、失败数据可以在条件化后共存。

这说明 foundation model 的 behavior distribution 已经从：

```text
one task -> one canonical trajectory
```

走向：

```text
same task -> multiple strategies / qualities / speeds / contexts
```

但 π0.7 对 compositional generalization 的证据仍主要是 task-level / capability-level。

它没有系统回答：

> successful zero-shot episode 中，关键 motor segments 到底来自 training repertoire 的 retrieval / interpolation / recombination，还是出现 support escape？

所以 π0.7 是 B 的强 evidence source，而不是 exact collision。

---

# 4. Qwen-RobotManip / BeingBeyond / VLAFlow：异构数据的核心已经从“更多”变成“如何让它们不互相冲突”

## Qwen-RobotManip

- report: https://arxiv.org/abs/2606.17846
- blog: https://qwen.ai/blog?id=qwen-robotmanip

核心不是简单堆到约 38,100 h，而是 representation / motion / behavior 三维 alignment。

作者非常明确：

> without alignment, heterogeneous data conflict; without diversity, alignment cannot generalize.

## Rethinking VLA Scaling: Alignment, Mixture, and Regularization

- https://arxiv.org/abs/2602.09722

controlled study 的结论很值得记：

- unified EEF-relative action representation 对 cross-embodiment transfer 很重要；
- naive pooling heterogeneous robot datasets 可以直接产生 **negative transfer**；
- sensory dropout / multi-stage fine-tuning 这类直觉做法并不稳定有效。

## VLAFlow

- https://arxiv.org/abs/2607.01586
- repo: https://github.com/MindVLA-Team/VLAFlow

在固定 π0-style architecture、相同 action expert、统一 action space 和约 5,000 h heterogeneous robot corpus 下比较 pretraining supervision：

- action-only pretraining 对 heterogeneous data 很脆弱；
- language supervision 和 future-latent alignment 提供互补的 intermediate constraints；
- 二者结合 transfer 更稳定。

## JoyAI-RA 0.5

- https://arxiv.org/abs/2608.05674

再次强调 naive human/sim/robot pooling 会 negative transfer，需要 implicit + explicit action alignment。

---

# 5. 一个看起来很大的题：What Makes Heterogeneous Robot Experience Compatible?

这轮曾经认真考虑把下面这个摘成 C：

> **What makes heterogeneous robot experience add rather than interfere?**

或者：

> **When does more robot data become less useful because the supervision is physically incompatible?**

它非常自然，而且材料很强：

```text
Qwen: alignment unlocks scale
BeingBeyond: naive mixture -> negative transfer
π0.7: richer context disambiguates strategies
VLAFlow: action-only heterogeneous pretraining is fragile
JoyAI: dual alignment prevents negative transfer
```

但最后**不摘候选**。

原因不是问题不重要，而是 broad question 已经被 2026 的 scaling/alignment literature 正面占据。

如果我们继续，为了 novelty 很容易被迫缩成：

- 哪种 normalization；
- 哪个 coordinate frame；
- gradient conflict statistic；
- 某两个 embodiments 的 mixture ratio。

这违反 Round 5 标准。

**结论：不作为 C。**

---

# 6. 另一条诱人的题：What Information Must Be Removed for Cross-Embodiment Transfer?

从 Qwen / UniT / ContactFlow / UMA / human-video transfer 里出现一个很漂亮的共同趋势：

> successful cross-embodiment interfaces 往往主动去掉 body-specific information，只保留 object motion / contact intent / EEF-relative motion / visual consequence。

这很像 representation learning 里的 invariance question：

> transfer 是因为模型学得更多，还是因为 interface **忘掉了 embodiment**？

但继续搜索后发现 collision 已经非常危险：

- UMA 用 3D object motion 作为 shared interface；
- ContactFlow 用 3D contact trajectories 做 embodiment-agnostic action conditioning；
- UniT 把 heterogeneous kinematics 映射到 unified physical tokens；
- 甚至已经出现标题几乎就是 `Deleting Body Information from the Action Interface Causes Zero-Shot Transfer Across Robot Arms` 的 2026 preprint。

因此这个问题已经从 insight 变成正在形成中的赛道。

**结论：砍。**

---

# 7. WAM 大问题 1：Do World Action Models Actually Use Their Predicted Futures?

这个标题非常自然，一度是本轮最强的新题：

> WAM 号称通过预测未来来行动，但 action 是否真的因果依赖于 predicted future？

因为很多所谓 WAM 实际部署图很不一样：

```text
A. future prediction only as training auxiliary loss
B. joint future/action sampler
C. predict intended future -> inverse action
D. candidate action -> predicted consequence
E. propose multiple actions -> simulate -> score -> select
```

这些 computation 不应该都被同一个“world model helps control”故事覆盖。

但 exact collision 后，这题已经不能拿。

### Fast-WAM — 2026-03

`Do World Action Models Need Test-time Future Imagination?`

https://arxiv.org/abs/2603.16666

直接把 video co-training 和 inference-time future generation 拆开，发现 training-time world modeling 可以保留大量收益，而 test-time imagination 不一定必要。

### Faster-WAM — 2026-08

https://arxiv.org/abs/2608.04404

又在 OOD setting 下得到相反方向的重要结果：inference-time future conditioning 对 robustness 关键，并设计 sparse future interaction。

### RIFT — 2026-08

`Keep the Future, Drop the Rollout`

https://arxiv.org/abs/2608.11521

更直接：对多个 WAM 做 future-cache masking / reassignment 等 paired closed-loop intervention，证明 action 对 future values 有 sensitivity；同时发现完整 iterative rollout 未必必要，固定/一次性构造的 future representation 可以接近原行为。

这几篇已经把：

```text
training-time prediction
vs inference-time future use
vs evolving rollout necessity
```

连续做掉。

**结论：漂亮，但被快速占领，砍。**

---

# 8. WAM 大问题 2：Do Action-Conditioned World Models Actually Respond to Actions?

另一个非常自然的问题：

> 一个模型能生成 plausible future，不代表 future 真正由 action 控制。

可能存在 shortcut：

```text
current frame + visual inertia
    -> plausible next frame
```

而 action channel 只被弱使用。

这和 video world model / causal representation 中一个经典问题完全同构：observational prediction 不等于 interventional dynamics。

但是 2026 也已经迅速被占：

- Ctrl-World（ICLR 2026）：强调 fine-grained action controllability 和 candidate-policy rollout；
- GeniWorld（2026-08）：直接把 action controllability + OOD generalization 当核心；
- `Overcoming Statistical Bias in Action-Controllable World Models`（2026-08）：明确指出模型可利用 visual inertia / recurring motion shortcut，不真正响应 action，并提出 counterfactual consistency、zero-action / inverse-action checks。

所以：

> **prediction != controllable transition**

这个 thesis 本身现在已经不是空白。

**结论：砍。**

---

# 9. WAM 大问题 3：What Does a WAM Need to Predict for Control?

这条也非常自然：RGB future 里大量是 texture / lighting / background，而 control 真正需要的是：

- geometry；
- motion；
- contact；
- object identity / binding；
- semantics；
- force / tactile transition。

但 2026 已经进入高速分化期：

- AGRA：plausible visual future 不保证 action extraction，做 action-grounded representation alignment；
- DreamWAM：beyond RGB，显式预测 appearance / motion / geometry / semantics；
- DC-WAM：强调 interaction-induced dynamics 而不是 appearance reconstruction；
- SG-WAM：在 policy-derived geometry-aware representation 中做 future prediction；
- OA-WAM：object-addressable slots + causal slot intervention；
- VT-WAM / Tactile-WAM / FAWAM / HiTac-WAM：接触、触觉、force、slip 的未来预测。

因此 broad question：

> what future representation matters for control?

已经被整个 WAM frontier 共同在回答。

**结论：不占候选。**

---

# 10. 一个非常重要但不能拿的题：Does Future Prediction Improve Control Because It Learns Dynamics, or Just Because It Regularizes Representation?

这个问题从别的 AI 领域看尤其自然：auxiliary predictive objective 有时只是 representation regularizer，并不意味着模型真正建立了可调用 simulator。

但 robotics 这边也已经出现直接证据链：

- Fast-WAM：training-time video co-training 很重要，即使 test-time 不生成 future；
- Beyond Task Success（2026-05）：比较多类 WAM/VLA 的 behavioral / representation differences；
- VLAFlow：future latent alignment 作为 intermediate constraint 改善 heterogeneous transfer；
- Dyna-2：video/world-modeling objective 主要增强 cross-embodiment transfer；
- Faster-WAM：又说明在 OOD 下保留 inference-time future representations 仍有额外收益。

所以这个问题现在不是空白，而更像一条已经开始被系统拆解的研究线。

**结论：不摘。**

---

# 11. 最新整模型里真正值得记录的“工程异常”

这些暂时都不足以单独成题，但非常值得继续积累。

## 11.1 Dyna-2：video scaling 对 cross-embodiment 有益，对 in-domain human metric 未必有益

这是一个非常反直觉的 separation：

```text
more video/world-modeling
  -> robot transfer improves
  -> same-domain human action metric may not
```

可能说明 world modeling 学到的是一种对 embodiment shift 特别有用的 physical representation，而不是单纯更精确地拟合 source behavior。

这条目前最适合拿来**加强 B / physical-control transition 的叙事**，而不是独立起题。

## 11.2 π0.7：异构数据不是靠“更强模型”自动吸收，而是靠 richer context 把 behavior disambiguate

这意味着 context 不只是 instruction interface，也可能是**data compatibility mechanism**。

但 broad question 已经被 π0.7 自己强占，继续做容易变 prompt ablation。

## 11.3 Qwen-RobotManip：benchmark SFT 会出现 VLA-to-VA degradation

高 task score、低 language sensitivity，模型退化为 visual-action pattern matcher。

这仍然是非常好的 evidence：post-training 能提高 task performance 的同时改变 learned computation。

但 generic forgetting / behavior-repertoire-collapse 邻域 collision 太高，Round 6 的降级决定不变。

## 11.4 VLAFlow：action-only pretraining 在 heterogeneous corpus 上反而 fragile

这是很重要的经验：低维 action supervision 并不天然随着数据规模变强；language / future-state constraints 可能是让 heterogeneous experience 形成 shared structure 的关键。

## 11.5 CHORUS：centralized policy 看更多信息反而可能更差

CHORUS 的一个 practitioner-level 结果是：centralized VLA 条件在整个 multi-robot observation 上，理论上信息更多，但表现可能比 decentralized shared policy 更差；解释之一是 centralized input 打破了 pretrained backbone 的 semantic correspondence / input distribution。

这再次提示：

> 在 pretrained embodied model 里，more context != more usable information。

但单独拿出来仍容易落成 architecture anecdote。

---

# 12. 从其他 AI 领域迁移过来的三个审题原则

本轮不是直接迁 method，而是迁**问题结构**。

## 12.1 从 LLM emergence 迁来：不要把 threshold metric 当 qualitative transition

LLM scaling 文献已经反复提醒：离散 exact-match / success threshold 可能把平滑 improvement 看成“突然涌现”。

因此机器人 foundation scaling 如果要声称 qualitative transition，不能只看：

```text
0% success -> 90% success
```

而必须找到 continuous mechanism-level quantity。

B 很适合这里：

- distance to training motor support；
- reconstructability from known motor segments；
- interpolation coefficient；
- composition boundary crossing；
- support escape。

这样可以避免把普通 scaling curve 包装成 emergence。

## 12.2 从 causal world modeling 迁来：observational fit 不等于 interventional model

这一原则对 WAM 很重要，但当前已经被 controllability / counterfactual WAM work 抢占，因此更适合作为审计标准，而不是新题。

## 12.3 从 representation learning 迁来：transfer 往往来自正确 invariance，而不是“表示越丰富越好”

cross-embodiment 的大量 2026 工作都在主动删除 embodiment-specific nuisance：object motion、contact flow、EEF-relative actions、shared physical tokens。

但这条也已经形成赛道，所以不再起题。

---

# 13. Round 7 后对 B 的重新定位

Round 6 的 B 可以进一步升格成：

> **What qualitative transition, if any, does robot foundation pretraining induce in action generation?**

但标题仍建议保留更自然、更直接的：

# **How Do Robot Foundation Policies Generalize Actions?**

因为现在可以把 2026 的大证据串起来：

```text
small task-specific policies
    -> retrieval / interpolation evidence

large aligned heterogeneous VLAs
    -> strong OOD / cross-embodiment / recovery

π0.7
    -> compositional task-level claims

Dyna-2
    -> 1M-hour cross-embodiment scaling law
```

真正缺的一层是：

```text
what changed in motor generation?
```

这比“再做一个 scaling law”更机制，也比“某模型 trajectory similarity”更大。

---

# 14. B 的第一枪也因此应该调整

Round 6 里写的 retrieval / interpolation / composition / extrapolation 仍然成立，但 Round 7 后，**最重要的实验轴不应该只是 model family，而应该加入 pretraining scale / diversity。**

理想的 screening 结构：

```text
task-specific policy
vs
small / weak-pretrained generalist
vs
larger / broader-pretrained generalist
```

在相同 downstream task construction 下，分别构造：

1. in-support；
2. geometric interpolation；
3. compositional holdout；
4. true action-support extrapolation。

然后 simultaneous measure：

- closed-loop success；
- nearest training trajectory / segment support；
- piecewise reconstruction by known motor segments；
- action support escape；
- sensitivity to visual / language OOD。

最重要的不是得到“large model 更好”，而是看：

> **success curve 上升时，action mechanism 是否发生 qualitative change。**

如果 mechanism quantity 仍然平滑地只是 nearest-support coverage 变密，那么结论也很重要：

> scaling primarily expands reusable motor coverage rather than inventing new motor behavior.

---

# 15. Round 7 后的 shortlist

仍然只有两个：

## A — What Does History Actually Model in Robot Policies?

**状态：保留，但 collision risk 提高。**

IntentVLA 已经占掉 `history resolves short-horizon intent ambiguity`。A 若继续，必须证明更强的 task/world-state memory vs behavior-source/strategy memory decomposition，并有 causal rollout consequence。

## B — How Do Robot Foundation Policies Generalize Actions?

**状态：明显加强，当前优先级高于 A。**

Dyna-2 / π0.7 / Qwen-RobotManip 把 foundation scaling 与 generalization claim 推到了一个新高度，但 action-side mechanism 仍未闭合。

---

# 16. 下一轮最值得继续扫的区域

Round 8 不建议继续在已经高度拥挤的 WAM representation / future-conditioning 里硬挖。

优先继续：

1. **foundation scaling 的 qualitative transition**：找 controlled scale ladders、checkpoint ladders、data-diversity ladders，看有没有现成可复用 checkpoint 能直接审 B；
2. **pretraining → post-training interface**：尤其寻找“pretraining 明明给了能力，但 post-training 以某种可测方式重写/屏蔽了它”的大异常，不过必须避开 generic catastrophic forgetting；
3. **behavior support / controllability 在真实部署故障中的表现**：找 industrial / practitioner report 里 model 在 new geometry / dynamics / tool-use 下到底是 reuse 还是 invent；
4. **world-model scaling 与 action scaling 的 separation**：Dyna-2 已经给出一个很强 anomaly，继续找是否别的模型也出现“source-domain metric 不升，但 cross-embodiment control 升”的类似现象；
5. **从 LLM / multimodal scaling 迁机制问题**，尤其是：scale 改变的是 representation geometry、retrieval regime、composition regime，还是仅仅覆盖率。

继续沿用硬规则：

> **没有自然、独立、可高兴的大问题，就不新增候选。**
>
> **如果 broad question 已经被做，只能靠越压越窄活下来，就直接砍。**
