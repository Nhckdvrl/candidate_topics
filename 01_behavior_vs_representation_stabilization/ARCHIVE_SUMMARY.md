# Topic 01 归档总结 — Behavior Stabilization vs. Representation Stabilization

## 最终状态

**ARCHIVED / KILLED**

```yaml
G0-A: PASS
G0-B: FAIL
final_decision: KILL TOPIC
G1_crosscoder: NOT RUN
```

这个题目的核心假设没有通过预先设定的 G0 验证，因此在这里停止，不再进入 crosscoder / sparse-feature 阶段。

我们真正想验证的是：

> 当语言模型在全局 output-distribution space 中的变化已经进入很低的 late-training regime 后，内部 representation 是否仍然持续发生明显、非平凡的重组？

如果成立，最有意思的情形应该是：behavior 已经明显稳定，而 representation movement 保持得更多、更久，随后再进入 feature-level 分析去检查 late feature emergence / disappearance / causal relevance change。

实际结果恰好没有出现这种 decoupling。

---

## 1. 题目从哪里来

这个题目来自两条已经成立的研究线的交叉：

1. **Behavior / function-space stabilization**
   - Kishino et al., *Establishing a Scale for KL Divergence in Language Models Across Various Settings* (Findings ACL 2026)
   - 主要现象：Pythia 在 output-distribution / likelihood geometry 中的 movement 会随着训练明显下降，即使参数仍然继续移动。

2. **Representation / feature dynamics**
   - *Evolution of Concepts in Language Model Pre-Training*
   - *Crosscoding Through Time*
   - 主要现象：内部 representation / sparse features 会随 pretraining checkpoint 出现、维持、重组或消失。

自然产生的问题是：

> behavior 已经进入低 movement regime 之后，representation 是否还在继续学？

为了避免把单纯的 weight drift 或坐标漂移误认为“继续学习”，G0 没有直接上 crosscoder，而是先做一个便宜的 residual-geometry screen。

---

## 2. 我们实际做了什么

### 模型

- `EleutherAI/pythia-410m`
- middle GPT-NeoX block：layer 12
- representation hook：`resid_pre`

### 数据

- 1,000 个固定 Pile UTF-8 byte chunks
- 每个 chunk 约 1,024 bytes
- corpus SHA-256 固定并在所有 checkpoint 间复用
- token length 实际范围：52–556
- 无 tokenizer truncation

### checkpoint 设计

所有比较严格使用相同训练跨度：

```text
2k   -> 3k
5k   -> 6k
10k  -> 11k
20k  -> 21k
50k  -> 51k
100k -> 101k
142k -> 143k
```

统一：

```text
Delta = 1,000 training steps
```

这样不同训练阶段的 movement 可以直接比较，不把 checkpoint 间隔长度混进结果。

### G0-A：Behavior premise

对每个 checkpoint 计算 1,000 个固定文本的 sequence log-likelihood，构建 double-centered checkpoint × text matrix，然后计算 local KL proxy（bits/byte）。

同时保留两套结果：

- raw KL；
- robust KL：lower-tail clipping + top 3% movement outlier trimming。

统计不确定性使用 example-level cluster bootstrap。

### G0-B：Representation screen

每个文本固定取 4 个 interior token positions，在相同 fixed-1k checkpoint pairs 上比较 layer-12 `resid_pre`。

使用三种互补指标：

1. matched cosine drift；
2. pooled-standardized matched drift；
3. projected linear CKA（作为 rotation/scale-tolerant control）。

判断重点不是“late drift 是否显著大于 0”，而是比较：

```text
behavior movement 的 early -> late 衰减
vs.
representation movement 的 early -> late 衰减
```

如果 representation 真正在 behavior 稳定后仍持续 reorganize，那么 representation 的 late/early ratio 应该明显高于 behavior 的 late/early ratio。

---

## 3. G0-A 结果：Behavior premise 成立

G0-A 很清楚地通过。

| Pair | Raw KL | Robust KL |
| --- | ---: | ---: |
| 2k -> 3k | 0.549120 | 0.367483 |
| 5k -> 6k | 0.099157 | 0.090732 |
| 10k -> 11k | 0.073646 | 0.069836 |
| 20k -> 21k | 0.070549 | 0.067124 |
| 50k -> 51k | 0.055134 | 0.053024 |
| 100k -> 101k | 0.024534 | 0.025927 |
| 142k -> 143k | 0.020785 | 0.019723 |

Early -> late：

```text
Raw behavior ratio    = 0.03785  (~26.4x decay)
Robust behavior ratio = 0.05367  (~18.6x decay)
```

raw 与 robust 的趋势一致；robust 版本在去除异常文本后仍然保留非常强的 late-training movement decay。

因此这个实验至少确认了一件重要的事：

> seed paper 的 behavior-side premise 在当前 Pythia-410M + 固定 corpus + fixed-horizon 设置下是成立的。

所以后面的 negative result 不能简单归因于“behavior measurement 没复现出来”。

---

## 4. G0-B 结果：核心假设失败

Representation movement 也随着训练强烈下降，而且下降得**不比 behavior 慢，反而更快**。

| Metric | Early -> late ratio | Equivalent decay |
| --- | ---: | ---: |
| Raw behavior KL | 0.03785 | 26.4x |
| Robust behavior KL | 0.05367 | 18.6x |
| Cosine drift | 0.03019 | 33.1x |
| Standardized drift | 0.03428 | 29.2x |
| CKA movement | 0.000370 | ~2705x |

最关键的是：

```text
R_behavior_robust = 0.05367
R_cosine          = 0.03019
R_standardized    = 0.03428
```

我们原本需要看到：

```text
R_repr >> R_behavior
```

也就是 behavior 已经衰减很多，而 representation 仍保留相当多的 movement。

实际看到的是：

```text
R_repr < R_behavior
```

representation movement 反而衰减得更彻底。

到最后一个 142k -> 143k pair：

```text
cosine drift       = 0.005740
standardized drift = 0.016658
1 - CKA            = 0.000212
CKA                = 0.999788
```

这不是我们需要的“behaviorally silent late reorganization”。整体 residual geometry 已经非常稳定。

---

## 5. Robustness check：不是 corpus 抽样偶然

为了避免 1,000 个文本里某一批样本恰好造成 negative result，又做了 30 次 deterministic random half-sample（每次 500 texts）。

得到的 median late/early ratios：

| Metric | Median | Empirical 2.5%–97.5% range |
| --- | ---: | ---: |
| Robust behavior | 0.05244 | [0.04816, 0.06066] |
| Cosine | 0.03024 | [0.02966, 0.03067] |
| Standardized | 0.03437 | [0.03348, 0.03510] |

**30 / 30 个 half-samples 中，没有一次 cosine 或 standardized representation drift 比 robust behavior 保留更多 early-to-late movement。**

因此方向非常稳定：

> representation 没有表现出比 behavior 更晚的 stabilization。

---

## 6. 为什么这足以 KILL，而不是继续上 crosscoder

理论上，CKA / residual geometry 没看到 decoupling，并不能数学上证明“绝对不存在某些 sparse features 在 late training 继续变化”。

但这个候补题的立项标准从一开始就不是：

> 只要还能想出更复杂的 measurement，就继续找 positive result。

而是：

> 先用一个便宜、能够真正否定 premise 的 G0，只有看到清晰 temporal decoupling 才投入 feature-level 工程。

当前结果不只是“CKA 没变化”：

- matched cosine drift 同样快速下降；
- standardized matched drift 同样快速下降；
- CKA 到 final pair 几乎为 1；
- representation early-to-late decay 比 robust behavior 更强；
- 30 次 half-sample 稳定复现同一方向。

在这种情况下继续训练 crosscoder，已经很容易变成：

> G0 不支持假设之后，再不断换更复杂 measurement 寻找一个能把题救活的 positive signal。

这违背了候补题阶段“快准狠判生死”的目标。

而且附近已有 PolyPythias、Evolution of Concepts、Crosscoding Through Time 等工作。如果 aggregate residual dynamics 都没有给出清楚的新 temporal decoupling，仅仅发现某些 sparse feature 仍有少量 turnover，也很难形成足够独立的新贡献。

因此停止在 G0-B 是正确选择。

---

## 7. 这个题具体失败在哪里

### 失败点 1：自然的 cross-paper 空格，不代表自然现象真的存在

这个题在文献结构上非常漂亮：

```text
Paper A: behavior stabilization
Paper B: representation dynamics
=> compare their stabilization time
```

这是一种合理的选题生成方式，但它只保证“问题自然”，不保证“两个时间尺度真的会分离”。

本题的数据告诉我们：至少在 Pythia-410M middle residual geometry 上，两者不是我们预期的 decoupling；representation 甚至稳定得更快。

**教训：cross-filling 一个空格只是生成问题的方法，不是支持假设的证据。**

### 失败点 2：从 weight drift 推到 meaningful representation drift，本身没有依据

最初最吸引人的直觉是：

```text
behavior 已经稳定
weights 还在持续变化
=> 模型内部可能还在继续 reorganize
```

但 weight space 有大量冗余、对称性和 functionally irrelevant directions。

因此：

```text
parameter drift != meaningful representation drift
```

当前结果非常直接地说明，虽然 behavior-side seed 强调 parameter training 仍继续，但 middle residual representation 本身完全可以快速收敛。

**教训：不要把一个空间中的“仍在变化”自动外推到另一个空间。每个中间箭头都应该被单独验证。**

### 失败点 3：如果 G0 的 positive story 需要非常复杂的 feature method 才能出现，题目已经不够“天然”

一个强题目最好能在便宜 measurement 上先看到明显信号，再用复杂方法解释。

如果 cheap geometry 完全沿相反方向变化，只能寄希望于：

> 也许 sparse crosscoder 能找到少数 late feature。

那么研究叙事就从“解释一个明显存在的自然现象”变成“用越来越复杂的方法寻找某种可能存在的信号”。

这类题很容易陷入 method-driven salvage。

**教训：复杂方法应该解释 G0 已经看到的现象，而不是负责创造现象。**

### 失败点 4：候补题要提前定义什么结果算“不值得继续”

这次做得对的一点是，在跑实验之前已经规定：

```text
behavior stabilizes + representation stabilizes similarly => kill
```

所以 negative result 出来后不需要重新发明 story。

**教训：每个新题在正式投入前都应该有明确的 kill criterion，而且必须允许最核心假设真的被判死。**

---

## 8. 这次实验留下的正资产

虽然题目被 kill，但实验本身并不是无效工作。

### 1. 一个可靠的 Pythia fixed-horizon behavior-dynamics pipeline

已经实现并实测：

- 固定 byte-chunk corpus；
- corpus hash consistency；
- dense Pythia revisions；
- sequence LL extraction；
- double-centering；
- local KL proxy；
- raw + robust sensitivity analysis；
- cluster bootstrap。

以后遇到其他 training-dynamics 题，可以直接复用。

### 2. 一个低成本 matched-representation trajectory pipeline

已经实现：

- explicit `resid_pre` hook；
- matched token positions；
- cosine drift；
- standardized drift；
- projected CKA；
- example-level bootstrap。

这比重新搭一套 checkpoint representation comparison 快得多。

### 3. 一个更重要的选题流程验证

这次真正有价值的是：

```text
seed papers
-> exact one-step question
-> predefine kill condition
-> cheap G0
-> negative
-> stop
```

没有因为题目在文字上很漂亮，就继续投入 crosscoder、scale sweep、多 seed 和更多模型去“把它救回来”。

这正是候补题仓库应该起到的作用。

---

## 9. 对以后选题的具体要求

以后再做类似题目，优先满足下面几条：

1. **问题本身天然存在，而不是只因为两个 paper 能组成一个 2×2 空格。**
2. **seed paper 提供的是直接现象，而不是需要跨两三个隐含箭头推导出的直觉。**
3. **第一轮实验必须能真正 kill 核心 premise。**
4. **复杂 method 必须用于解释已经观察到的现象，而不是用来寻找现象。**
5. **如果 negative result 需要不断换 metric / layer / model 才能被绕开，就应该优先归档，而不是扩大搜索空间。**
6. **相邻工作已经很密时，G0 positive 必须非常强；弱 positive 不值得继续。**

---

## 10. 最终结论

这个题目的行为侧 premise 被成功复现，但核心 prediction 没有出现：

> **Pythia-410M 的 middle residual representation 并没有在 behavior stabilization 之后继续保持更强的 movement；相反，它的 early-to-late movement 衰减至少和 behavior 一样快，并且在多个 representation metrics 和 corpus half-samples 上都稳定成立。**

因此：

```text
ARCHIVE TOPIC 01
DO NOT RUN G1
```

保留代码、验证协议、结果和 artifacts，作为以后 Training Dynamics 题目的 reusable infrastructure 和失败案例。

详细数值见：

- [`G0_RESULTS.md`](./G0_RESULTS.md)
- [`VALIDATION.md`](./VALIDATION.md)
- [`artifacts/analysis/`](./artifacts/analysis/)
