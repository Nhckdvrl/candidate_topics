# Topic 03 归档总结 — Coverage Collapse vs. Latent Viability of Suppressed Branches

## 最终状态

**ARCHIVED / KILLED AT BEHAVIORAL PREMISE**

```yaml
seed_reproduction_audit: FAIL
R1_late_coverage_degradation: FAIL
R2_incorrect_first_fork_commitment: FAIL
G0_B_latent_probe: NOT RUN
final_decision: ARCHIVE
reason: FIRST_FORK_BEHAVIORAL_PREMISE_NOT_PRESENT
```

这个题目不再继续。

我们原本想问：

> 当 SFT 让一个原本可探索的 reasoning fork 在行为层面发生 coverage collapse 时，被压掉的正确 alternative branch 是否仍然以 branch-viability signal 的形式保留在 hidden state 中，只是没有被最终 native readout 使用？

这个问题只有在一个前提下才有实验对象：**同一条训练轨迹上必须真的出现 first-fork coverage collapse / wrong commitment。**

最终的 reproduction audit 把之前所有可能造成 false kill 的实现差异补齐后，这个前提仍然没有出现。因此没有理由继续做 probe、layer sweep、LED baseline 或 causal intervention。

---

## 1. 题目从哪里来

Seed paper 是 Nguyen et al., **Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road** (2026)。其核心动机是：

- SFT 可能提高 `pass@1`，同时降低大 `k` 的 sampled coverage；
- Graph Branching 提供一个明确的 binary first fork；
- 训练后 first-fork confidence 会极化，其中包含 confident-but-wrong branch commitment；
- inference-time prefix / diversity intervention 能恢复部分 coverage。

我们想把问题往内部机制推进一步：

> 如果行为上某个 branch 被压掉，它是真被 representation 擦除了，还是仍然以可访问的 viability signal 存在，只是在 late readout 中没有被选中？

这本来是一个比“hidden states contain alternatives”更窄、更强的 claim，因为它要求：

1. 有真实的 coverage-shrinking SFT trajectory；
2. 有 exact graph-ground-truth branch viability；
3. early readout 冻结后还能跨 checkpoint transfer；
4. matched target counterfactual 下信号必须随正确 branch 翻转；
5. 在 label-free native-vs-latent disagreement subset 上 latent readout 必须更可靠。

---

## 2. 文献审计后，真正剩下的贡献空间已经很窄

在实现前的 literature audit 中发现了几个直接 collision：

- **Latent Exploration Decoding (ICML 2026)** 已经表明 post-training 后 final-layer posterior 可以变尖，而 intermediate-layer posterior 仍保留更高探索性；因此“intermediate layers retain exploration”已经不是新贡献。
- 2025–2026 的工作已经证明 hidden activations 可以预测 alternative outcomes / reasoning paths；因此单纯 `probe AUC > 0.5` 不够。
- forced alternative-token continuation viability 也已有相邻工作。
- probe literature 还说明高 AUC 可能只是 format / lexical shortcut。

因此 Topic 03 最后只剩一个可辩护的狭窄命题：

> **在一个已知发生 coverage collapse 的 binary fork 上，early checkpoint 学到的 branch-ground-truth viability readout 是否在 late checkpoint 仍然保留，并在 matched counterfactual 与 label-free disagreement test 中比 native late commitment 更可靠？**

也正因为 claim 已经很窄，behavioral premise 必须先非常明确地成立。否则后面的复杂 probe 只会变成 mechanism-first salvage。

---

## 3. 第一版 G0 为什么不能直接判死

最初 paper-exact `2e-5` 轨迹跑了：

- Qwen2.5-0.5B；
- 200 test problems；
- 16 samples/problem；
- e01/e02/e04/e16；
- temperature 1.0, top-p 0.95。

得到：

| checkpoint | pass@1 | pass@8 | pass@16 | viable-first | first-fork entropy |
|---|---:|---:|---:|---:|---:|
| e01 | 0.2931 | 0.5406 | 0.680 | 1.000 | 0 |
| e02 | 0.2403 | 0.7763 | 0.915 | 1.000 | 0 |
| e04 | 0.2988 | 0.8504 | 0.965 | 1.000 | 0 |
| e16 | 0.3503 | 0.9011 | 0.965 | 1.000 | 0 |

原 gate 给出 `stop_or_redesign`，因为 e04→e16 的 `pass@8` 不降反升。

但审计后发现三个可能的 false-kill 来源：

1. seed paper 重点观察 `pass@32`，而 16 samples 根本测不了；
2. 漏了 e08，可能错过真实 peak；
3. seed paper appendix 写 `lr=2e-5`，但官方 `run_sft.sh` 实际使用 `1e-5`。

因此没有直接归档，而是允许一次、也是最后一次严格 reproduction audit。

这一步是必要的：**归档一个题目之前，必须先排除“我们只是没有复现对 seed phenomenon”。**

---

## 4. 最终 reproduction audit

最终 audit 同时覆盖了两种学习率解释：

```text
1e-5 = 官方 GitHub run_sft.sh 的实际参数
2e-5 = seed paper appendix 报告的参数
```

固定实验：

- Qwen2.5-0.5B；
- 200 held-out problems；
- e01/e02/e04/e08/e16；
- 64 samples/problem；
- temperature 1.0；
- top-p 0.95；
- max tokens 512；
- `pass@1 / pass@2 / pass@8 / pass@32`；
- paired bootstrap 100,000 resamples；
- 额外做真正的 **teacher-forced first-decision token probability audit**，不再只依赖 sampled branch entropy。

完整结果见：

- [`results/REPRODUCTION_AUDIT.md`](./results/REPRODUCTION_AUDIT.md)
- [`results/REPRODUCTION_AUDIT_COVERAGE.csv`](./results/REPRODUCTION_AUDIT_COVERAGE.csv)
- [`results/REPRODUCTION_AUDIT_TEACHER_FORCED.csv`](./results/REPRODUCTION_AUDIT_TEACHER_FORCED.csv)

---

## 5. R1 失败：没有 late coverage degradation

### `1e-5`

`pass@32` trajectory：

```text
e01 0.9842
e02 0.9832
e04 1.0000
e08 1.0000
e16 1.0000
```

e04/e08/e16 基本饱和且不可区分。

### `2e-5`

`pass@32` trajectory：

```text
e01 0.7471
e02 0.9606
e04 0.9908
e08 0.9914
e16 0.9844
```

表面看 e08→e16 有 `0.0070` 的小下降，但 paired 95% CI 为：

```text
[-0.006857, 0.024346]
```

跨 0，而且 practical effect 很小。

因此两套合理 reproduction 参数都没有得到可以支持研究计划的 late `pass@32` collapse。

---

## 6. R2 失败得更彻底：first fork 根本没有 wrong commitment

teacher-forced audit 是最终 kill 的关键。

两套学习率、五个 checkpoint 全部满足：

```text
output_choice_acc = 1.0
mean_p_true_viable_pair >= 0.999998
wrong_commit_rate = 0.0
strong_wrong_commit_rate = 0.0
```

sampled first branch 同样是：

```text
mean_p_viable_first = 1.0
first_branch_entropy = 0
```

也就是说模型并不是在 first fork 上“越来越自信地选错 branch”。恰恰相反，它从这个测量下几乎完美知道哪个 first branch 才能通向 queried terminal。

训练过程中 confidence 的确会进一步极化，但极化方向始终是**正确 branch**。

这直接移除了 Topic 03 需要研究的对象：

```text
不存在 suppressed correct first branch
=> 无法询问它是否还潜伏在 hidden state
```

---

## 7. 那 sampled correctness 为什么还会波动？

因为模型后面会算错。

first branch 已经 100% viable，但 `pass@1` 远低于 100%，说明大量错误发生在：

- 后续 arithmetic execution；
- 多步 substitution；
- 数值计算 / generation noise；
- 而不是 first-fork route selection。

因此当前数据中的 sampled coverage variation 和我们要解释的 branch-suppression mechanism 不是同一个对象。

继续做 latent probe 会犯一个很典型的错误：

> 先有一个漂亮的 mechanism story，再去 hidden states 里寻找某种 signal，即使 behavior 层的目标现象根本没有出现。

这正是候补题阶段应该避免的。

---

## 8. 为什么不继续跑 G0-B

即使 G0-B 最后 probe AUC 很高，也无法救这个题。

因为没有 behavioral collapse 时，probe 最多说明：

> hidden state 可以编码 graph / branch viability。

这已经被相邻工作覆盖，而且不能证明：

> 一个 behaviorally suppressed alternative 仍然 latently available。

后者需要“suppressed alternative”先真实存在。

因此：

```text
R1 FAIL + R2 FAIL
=> G0-B NOT AUTHORIZED
```

不是因为 GPU 成本，而是因为继续做已经不回答原问题。

---

## 9. 这个题具体失败在哪里

### 失败点 1：不能把 seed paper 的现象当作自动可用的实验对象

一个 published phenomenon 可以作为 seed，但如果后续研究的全部机制叙事依赖它，就必须先在自己的 exact setup 里确认：

```text
这个现象真的出现
并且出现的是我们想解释的那一部分
```

本题最初看到的是 sampled correctness / pass@k variation，但更细的 teacher-forced audit 证明 first-fork wrong commitment 根本不存在。

**教训：复制 headline metric 不够，必须验证机制研究真正依赖的 local phenomenon。**

### 失败点 2：metric-level replication 和 mechanism-level replication 是两回事

即使某些 `pass@k` 曲线有变化，也不能自动解释成 first-fork coverage collapse。

本题说明：

```text
pass@k movement
!=
first-fork wrong commitment
```

同一个 overall metric 可以由 downstream execution noise 产生。

**教训：机制题必须验证 causal bottleneck / decision point 本身，而不是只看 aggregate outcome。**

### 失败点 3：复现歧义要一次性解决，不要无限 rescue

我们允许了一次 reproduction repair，因为存在明确、外部可验证的差异：

- 16 vs 64 samples；
- 缺少 e08；
- paper `2e-5` vs official code `1e-5`；
- sampled branch entropy vs teacher-forced token confidence。

这些修正都不是看结果后随意找参数，而是为了忠实对齐 seed paper。

修正后仍失败，因此停止。

**教训：允许一次有独立依据的 reproduction correction；修正完成后仍失败，就不要继续搜索设置直到现象出现。**

### 失败点 4：复杂 mechanistic pipeline 必须建立在便宜、明确的 behavior premise 上

这个题后面原本需要：

- matched target flip；
- target-blind control；
- frozen early readout；
- cross-checkpoint transfer；
- discovery/confirmation split；
- label-free disagreement selection；
- LED/logit-lens baseline；
- 最好再加 causal intervention。

这些设计本身并非错误，但只有在 behavior phenomenon 已经很稳时才值得付成本。

如果 G0-A 都没有 object，继续堆这些 gate 只会把一个不存在的现象包装得更复杂。

**教训：复杂方法应该解释一个已经清晰存在的现象，而不是替不存在的现象寻找 representation story。**

---

## 10. 对以后选题最重要的可迁移教训

这次最值得保留的是一个新的前置原则：

> **Mechanism research requires mechanism-level phenomenon replication.**

不要只问：

```text
seed paper 的最终 metric 能不能大致复现？
```

还要问：

```text
我们要解释的那个具体 decision point / failure mode / state transition
在自己的系统里到底存不存在？
```

如果不存在，就不要直接跳到 hidden-state analysis。

可以把以后类似题目的验证顺序固定为：

```text
1. reproduce aggregate phenomenon
2. localize the claimed mechanism-level event
3. verify it occurs at useful frequency and effect size
4. only then study representation / intervention / causal mechanism
```

本题在第 2 步失败，因此终止是正确的。

---

## 11. 留下的正资产

虽然题目被 kill，仍然留下了一些可复用资产：

- Graph Branching exact parser 与 first-fork reconstruction；
- paper/code parameter mismatch audit；
- multi-checkpoint pass@k sampling pipeline；
- paired bootstrap trajectory comparison；
- teacher-forced branch-probability extractor；
- matched target-flip / target-blind counterfactual构造；
- frozen cross-checkpoint probe protocol；
- 一个很重要的 falsification pattern：**先验证 mechanism-level object，再做 representation analysis。**

这些代码可以复用，但不应继续包装成 Topic 03 的 positive story。

---

## 12. 最终一句话

> **Topic 03 不是因为 probe 失败而死，而是因为 probe 想解释的 first-fork suppression 在严格复现后根本没有出现。**

因此最终决定：

```text
ARCHIVED — FIRST_FORK_BEHAVIORAL_PREMISE_NOT_PRESENT
```

除非未来出现一个新的、独立且可靠的数据/模型设置，明确复现了高频率的 wrong first-fork commitment 与 late coverage collapse，并形成新的注册问题，否则不要复活本题。
