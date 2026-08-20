# Topic 04 归档总结 — Confidence and Error Correction

## 最终状态

**ARCHIVED / KILLED AT MEASUREMENT / IDENTIFICATION GATE**

```yaml
G-1v1: FAIL
G-1v2_locked_repair: FAIL
G0_corrective_training: NOT RUN
G1_durability: NOT RUN
final_decision: ARCHIVE
hypothesis_falsified: false
```

这个项目没有得到“错误信念越坚定越容易/越难纠正”的正面或反面答案。

它在更早的一层被停止：**在当前 Qwen2.5-1.5B-Instruct × MMLU-Pro 多选题实验系统中，我们无法在足够大的样本上，把“正确答案本来有多可达”与“模型对某一个具体错误有多坚定”可靠地拆开。**

因此，正确的结论不是：

> wrong commitment does not affect correction.

而是：

> **the proposed correction hypothesis was never tested, because the required high/low-commitment comparison could not be identified cleanly at sufficient scale.**

本题按照预先锁死的停止规则归档，不再通过放宽 matching、切换更大模型、换 confidence metric、改成 free response 或加入 hidden-state measurement 来救题。

---

# 1. 最初的问题是什么

自然问题是一句话：

> **如果两个学习者离正确答案一样远，一个人坚定地相信某个具体错误，另一个人只是在多个错误之间犹豫，谁更容易被纠正？**

这个问题来自人类学习中的 hypercorrection / error-correction literature。

经典现象是：一些高置信错误在得到反馈后，反而比低置信错误更容易被纠正。一个解释是 surprise / attention：

```text
“我这么确定居然还错了”
-> corrective feedback receives extra weight
```

另一个解释则是 prior / partial knowledge：高 confidence 可能并不是独立原因，只是说明学习者已经更接近正确知识。

现代语言模型给了一个理论上很干净的实验系统：我们可以精确读取每个候选答案的概率，并在每一次相同 corrective exposure 后重新测整个学习曲线。

因此原计划不是简单比较 raw confidence，而是主动拆成两个量。

### Target accessibility

\[
a(x)=p(y^*\mid x)
\]

表示模型在纠正之前对正确答案本来有多大概率。

### Wrong-hypothesis commitment

先只在错误选项上归一化：

\[
q_j(x)=\frac{p(y_j\mid x)}{1-p(y^*\mid x)},\qquad y_j\neq y^*
\]

然后定义：

\[
c_{\max}(x)=\max_j q_j(x)
\]

直观上：

- 两个样本的 `p(correct)` 一样：正确答案同样难以访问；
- `c_max` 高：剩余错误概率主要押在一个具体 misconception 上；
- `c_max` 低：错误概率分散在多个替代答案上。

真正想检验的是：

\[
\text{matched }p(correct),\quad c_{high}\;vs.\;c_{low}
\]

在完全相同的 corrective SFT exposure 下，谁的 correction curve 更快、更慢或更容易 relapse。

---

# 2. 为什么没有直接开始训练

这个题最危险的 confound 从一开始就很明显：

```text
high wrong confidence
usually also means
low correct-answer probability
```

如果直接比较 high-confidence wrong 与 low-confidence wrong，观察到训练速度不同，无法判断是：

1. wrong belief 更坚定；还是
2. 正确答案原本离模型更远。

所以我们在 G0 之前设置了 G-1：

> **先证明存在足够多的 high/low wrong-commitment 样本，并且可以把它们在 base `p(correct)` 等关键 covariates 上匹配得足够好。**

只有这个 identification gate 通过，后面的 correction dynamics 才有解释价值。

这一步最终成为本题真正的 kill gate。

---

# 3. G-1v1：第一版 measurement 为什么失败

## 3.1 实验设置

```text
model       Qwen/Qwen2.5-1.5B-Instruct
stimuli     MMLU-Pro test, exactly K=10
items       9,981
rotation    all 10 cyclic option rotations
```

每一道题的 10 个语义选项都会轮流占据 A–J 的每个位置一次。

第一版做法：

1. 对每个 permutation 计算 A–J 条件概率；
2. 映射回 semantic option identity；
3. 对 10 个 permutation 的 semantic probabilities 做 arithmetic mean；
4. 要求同一个 semantic top-wrong 在 >=8/10 permutation 中保持 top-wrong；
5. 在 surviving initially-wrong items 中划 high / low commitment；
6. 在 category、`p(correct)`、question length、correct-answer length 上做 Hungarian matching。

## 3.2 G-1v1 结果

```text
scored items                    9,981
eligible stable wrong             716
low pool                           215
high pool                          215
matched pairs                       61
mean |Δ p(correct)|             0.00547
mean commitment separation      0.2529
```

61 对 surviving pairs 的匹配质量很好，因此最初可以排除一个简单解释：

> Hungarian matcher 本身并没有坏。

真正的问题出在进入 matcher 之前，绝大多数样本已经被 measurement definition 筛掉。

---

# 4. v1 暴露出的两个结构性 measurement defect

## 4.1 Defect A — reliability gate 与 treatment variable 机械相关

v1 要求：

```text
same semantic top-wrong in >=8/10 rotations
```

但我们真正想研究的 low commitment 本来就意味着：

```text
q1 ~= q2 ~= q3 ...
```

因此在错误选项非常接近时，轻微的 option-order perturbation 就可能让 top-1 identity 在几个错误选项之间切换。

也就是说：

\[
\text{top-wrong stability}\not\perp\text{wrong commitment}
\]

我们实际上使用 treatment variable 的一个结果来筛选 treatment 本身。

一个非常明显的证据是：v1 surviving pool 的 **low commitment cutoff 已经约为 0.795**。

但 K=10 时一共有 9 个错误选项；完全均匀的错误分布只有：

\[
1/9\approx0.111
\]

因此 v1 所谓的 low-commitment group 其实已经高度集中，真正 diffuse 的错误分布大部分被 stability gate 删除了。

## 4.2 Defect B — arithmetic mean 把 position susceptibility 伪装成 semantic uncertainty

raw result 中存在这样的题：

```text
permutation 1: model is ~90% sure option B
permutation 2: model is ~87% sure option B
permutation 3: model is ~62% sure option C
...
```

模型在每个 permutation 上都很 sharp，但随着 option position 改变，它坚定选择的 **semantic option** 会变化。

对 semantic probabilities 做 arithmetic mean 后，这类题会变成：

```text
apparently diffuse average distribution
```

于是一个真正的：

> sharp but position-sensitive response

被误解释成：

> low semantic commitment.

这不是小噪声，而是 construct validity 问题。

附近已有文献也表明 MCQ 的 option symbol / position bias 是真实且系统性的：例如 NAACL 2025 `Option Symbol Matters` 直接发现仅改变 option symbols 就能显著改变 LLM 的 MCQA 结果。2026 的 option-level psychometrics 工作也把 positional preference、response sharpness 与 distractor information 分开建模，而不是把完整选项分布直接当作纯 semantic belief。

因此 v1 的失败被判定为 **measurement failure**，而不是 hypothesis negative。

---

# 5. 为什么允许一次 G-1v2 repair

这个 repair 是在任何 correction outcome 出现之前提出的。

G0 从未运行，因此我们并不是因为“训练结果不好看”而换 metric。

而且 repair 针对的是一个明确的数学缺陷。

如果每个 rotation 下的 option logit 可以近似分解为：

\[
z_{r,j}=\alpha_j+\beta_{position(r,j)}
\]

其中：

- `alpha_j` 是 semantic score；
- `beta_position` 是 option-position nuisance；

那么在完整 balanced rotations 下，每个 semantic option 都恰好经过所有位置一次。

v2 因此定义：

\[
s_j=\frac1R\sum_r\log(p_{r,j}+\epsilon)
\]

再做：

\[
p_j^{debias}=softmax(s)_j
\]

也就是 normalized geometric mean / mean-log-prob aggregation。

在 additive position-bias model 下，共同的位置项会在最终 softmax 中抵消。

同时，v2 做了三件事：

1. **取消 top-wrong stability 作为 inclusion gate**；
2. 把 position sensitivity 单独定义为 distribution-level JS diagnostic；
3. 预先规定：如果 v2 仍 `<200` matched pairs，则直接 archive，不再进行第二次 rescue。

这次 repair 因而是合法的 measurement correction，而不是 post-hoc hypothesis shopping。

---

# 6. G-1v2 结果

v2 的 offline reaggregation 使用 v1 已经保存的全部 9,981 个 item × 10 permutation distributions，因此第一阶段完全不需要重新 inference。

结果：

```text
measurement version                         g1v2_logmean
scored items                                9,981
eligible initially-wrong                    6,668
low commitment cutoff                      0.23763
high commitment cutoff                     0.72070
low pool                                    2,001
high pool                                   2,001
matched pairs                                  130
mean |Δ p(correct)|                        0.00744
median |Δ p(correct)|                      0.00569
mean commitment separation                 0.64894
median commitment separation               0.64562
eligible median top-wrong stability          0.30  (diagnostic only)
eligible median position-susceptibility JS  0.2903
```

这个结果非常有信息量。

## 6.1 repair 的确修复了 v1 的主要筛选缺陷

v1：

```text
eligible wrong = 716
low cutoff     ~= 0.795
```

v2：

```text
eligible wrong = 6,668
low cutoff     ~= 0.238
```

说明：

- 去掉 treatment-dependent stability gate 是必要的；
- log-space reaggregation 恢复了真正的 low-commitment dynamic range；
- v1 的 61-pair failure 不能简单当作题目本身死亡。

这是 v2 repair 合理性的强证据。

## 6.2 但 v2 仍无法产生足够大的 clean comparison

high / low pool 已经各有 2,001 个，commitment separation 也非常大：

\[
\Delta c_{max}\approx0.649
\]

因此现在不再是：

> 没有 high/low commitment dynamic range。

surviving pairs 的 target accessibility matching 也很好：

\[
\operatorname{mean}|\Delta p(correct)|=0.00744
\]

所以也不能归因于 matcher 完全失效。

问题是：在同时保留原 identification requirements 时——

```text
same category
|Δ p(correct)| <= 0.02
question-length ratio <= 1.35
correct-answer-length ratio <= 1.50
```

2,001 × 2,001 的 high/low pools 最终仍只有：

\[
\boxed{130\ pairs}
\]

拥有足够的 common support。

这低于预先锁定的 hard stop：

\[
\boxed{<200\Rightarrow archive}
\]

因此实验在这里停止。

没有运行：

- balanced-family reliability audit；
- alternate-prompt reliability audit；
- predeclared 3B measurement replication；
- G0 corrective SFT；
- G1 relapse/durability。

停止发生在正确的位置。

---

# 7. 为什么 130 pairs 后必须停，而不是继续“优化 matching”

表面上还有很多能增加 pair 数的方法：

- 去掉 same-category constraint；
- 把 `p(correct)` caliper 从 0.02 放到 0.05；
- 不再匹配长度；
- 从 paired design 改成全量 regression；
- 换 3B / 7B；
- 换 MMLU / ARC；
- 改成 free-response confidence；
- 用 entropy 或 hidden states 替代 `c_max`。

但这些不是 bug fix。

它们都在改变原来的 identification strategy。

本题的科学价值来自一个非常具体的比较：

> **在正确答案原本同样 accessible、任务类型相近的情况下，只改变“错误是否集中在一个具体 hypothesis 上”。**

如果为了获得样本量而放松 `p(correct)` 或 domain comparability，那么原题最重要的 confound 又重新回来。

如果改成 regression 而不要求实际 overlap，则模型会主要依赖 extrapolation，而不是被数据支持的 high/low comparison。

如果改模型、改数据、改 response format，则不再是对当前失败的 confirmation，而是在重新寻找一个能让问题成立的 experimental system。

因此 130 不是“差 70 对而已”。

它意味着：

> **在这套自然 stimulus pool 中，核心两个变量缺少足够 common support，原来想要的 clean quasi-experiment 无法成立。**

这就是 identification failure。

---

# 8. 这次数据真正发现了什么

虽然 Topic 04 被 kill，G-1 本身留下了两个真实观察。

## 8.1 MCQ 中的“错误信念强度”不是一个可以直接从平均 option probability 读取的量

option ordering 会强烈影响 response geometry。

v2 eligible items 的：

```text
median top-wrong stability = 0.30
median position JS         = 0.2903
```

说明对于大量 item，模型的错误分布会随着 answer configuration 显著改变。

因此：

```text
MCQ option probability
!= automatically semantic belief probability
```

至少需要把 semantic preference、position susceptibility、response-channel sharpness 等因素拆开。

## 8.2 target accessibility 与 wrong commitment 很难在自然题库中独立操控

修复 measurement 后已经存在大量 high 与 low commitment items；真正不足的是 clean overlap。

这说明自然数据里：

```text
how accessible the correct answer is
```

和：

```text
how sharply the learner commits to one wrong alternative
```

并不是两个随手就能正交化的维度。

这不证明二者在理论上不可分，但说明直接把现成 benchmark 当作 quasi-experimental stimulus pool 并不够。

---

# 9. 为什么不把“position susceptibility”改成新 paper story

它确实是当前数据中最明显的自然现象之一。

但不应该用它救 Topic 04，原因有两个。

第一，它回答的是另一个问题：

> 为什么一个模型对同一道题的 semantic preference 会被 option formatting 改写？

而 Topic 04 问的是：

> wrong commitment 如何影响 corrective learning？

两者不是同一个 scientific claim。

第二，option-symbol / selection bias 已经有直接的近期文献，包括 NAACL 2025 `Option Symbol Matters`，以及更多 MCQA selection-bias / option-level psychometrics 工作。

因此把当前失败转写成：

> “我们发现 MCQ position bias 很有趣”

既是 post-hoc pivot，又缺少足够的新颖性。

如果未来真的要研究 position susceptibility，必须重新做 collision search、重新提出独立自然问题，并作为一个新的候选题注册。

---

# 10. 为什么这次不是 hypothesis falsification

这一点必须永久写清楚。

我们没有跑过任何 corrective SFT。

因此没有观察：

\[
\text{high commitment}\rightarrow\text{faster correction}
\]

也没有观察：

\[
\text{high commitment}\rightarrow\text{slower correction}
\]

更没有 evidence 支持：

\[
\text{commitment has no effect}
\]

所以不能在未来引用本项目说：

> “我们已经证明 confidence/commitment 不影响 correction。”

唯一成立的是：

> **The proposed paired design could not identify the effect cleanly enough to justify running the correction experiment.**

这是 feasibility / identification negative，不是 substantive hypothesis negative。

---

# 11. 这个题最重要的失败教训

## Lesson 1 — 对需要“拆两个天然相关变量”的题，第一步先检查 common support

这次最应该提前做的不是设计 10-cycle SFT，而是直接画：

```text
p(correct) × wrong commitment
```

的 joint distribution，并检查 high/low commitment 在同一个 `p(correct)` 区域有没有足够 overlap。

未来遇到类似问题：

```text
confidence vs knowledge
uncertainty vs difficulty
familiarity vs correctness
strategy diversity vs competence
```

第一阶段都应该先做 common-support audit。

**不要在 identification 还没成立时就设计后面的复杂 intervention。**

## Lesson 2 — reliability gate 必须尽量独立于 treatment variable

v1 最大的设计错误是用 top-wrong stability 去验证 wrong commitment reliability。

但 low commitment 本来就会导致 top-1 不稳定。

一个 measurement gate 如果机械地筛掉 treatment 的某一端，就会改变 construct 本身。

以后在设计 reliability criterion 时必须问：

> 如果 treatment 真的是 low/high，这个 gate 是否会天然更容易淘汰其中一组？

如果会，就不能把它当中性的 quality filter。

## Lesson 3 — 输出格式中的概率结构不能自动解释成语义心理量

A/B/C/D/J 的 next-token probability 同时包含：

- semantic evidence；
- symbol prior；
- position prior；
- response-format compliance；
- difficulty-triggered fallback。

把一个方便读取的 probability 当作 `belief / confidence / misconception strength` 之前，必须证明 construct validity。

**可测量不等于测到了你命名的那个东西。**

## Lesson 4 — 一个自然问题可以很有意思，但仍然不适合作为当前实验题

“坚定的错误更容易还是更难被纠正？”这个问题本身依然自然。

项目被 kill 的原因不是问题愚蠢，而是：

> 当前可行的自然数据 + measurement 无法给出足够干净的识别。

研究选题不只需要：

```text
interesting question
```

还需要：

```text
credible identification
```

两者缺一不可。

## Lesson 5 — repair 必须有次数限制

v1 确实有结构性 defect，所以一次 repair 是合理的。

但如果 v2 失败后继续：

```text
换 gate
-> 换 matching
-> 换 model
-> 换 dataset
-> 换 response format
```

就会重新进入“总能找到一个设置把题救回来”的循环。

本项目提前规定：

```text
one repair only
v2 <200 pairs -> archive
```

最终真的执行了这个规则。

这比得到 positive result 更重要。

## Lesson 6 — cheap gate 的价值是避免错误地烧后续算力

项目已经实现了完整 G0 trainer / evaluator，但最终一轮 corrective SFT 都没有跑。

这是正确的。

G-1 的作用本来就是在最便宜的地方回答：

> 如果 measurement / comparison 都站不住，后面的训练结果有没有解释价值？

答案是否定的，所以停止。

---

# 12. 项目留下的可复用资产

虽然课题归档，工程资产可以复用。

## 12.1 MCQ full-distribution scorer

已经支持：

- chat-template-consistent scoring；
- fixed-K label probability；
- balanced cyclic permutations；
- semantic remapping；
- log-space balanced aggregation；
- full response-channel diagnostics。

## 12.2 Position-susceptibility measurement

已经实现：

- per-permutation semantic distributions；
- mean-log-prob semantic debiasing；
- JS-based position susceptibility；
- alternative balanced permutation family；
- prompt/family robustness audit。

## 12.3 Controlled matching pipeline

已经实现：

- frozen base covariates；
- category-stratified Hungarian matching；
- target-accessibility caliper；
- length matching；
- discovery / confirmation split。

## 12.4 Complete negative-results record

仓库保留：

```text
results/g1/      # G-1v1
results/g1v2/    # locked repair
MEASUREMENT_REPAIR.md
VALIDATION.md
SERVER_RUNBOOK.md
```

以后可以明确知道：

- 哪个 measurement 为什么失败；
- 哪个 repair 已经尝试；
- 为什么不能继续改同一个题。

---

# 13. 未来如果重新碰到类似问题，应该怎么做

不要从 Topic 04 继续 rescue。

如果未来有新的外部证据确实值得重新研究“错误 conviction 与纠正”，至少需要一个**新的 identification system**，例如能够主动正交操控：

```text
target accessibility
×
wrong-hypothesis concentration
```

而不是在自然 benchmark 中事后匹配。

但这必须重新问两个问题：

1. 这种人工操控是否仍然对应自然的“misconception / correction”现象？
2. 是否已经变成一个过于 synthetic 的 toy problem？

因此这不是本项目的下一步，而是未来可能重新注册的新候选题。

---

# 14. 最终结论

最准确的一句话是：

> **Topic 04 was killed before corrective training because, even after one mathematically justified measurement repair, the natural stimulus pool did not provide enough clean common support to separate correct-target accessibility from commitment to a specific wrong hypothesis. The correction hypothesis itself remains untested.**

项目到此归档。
