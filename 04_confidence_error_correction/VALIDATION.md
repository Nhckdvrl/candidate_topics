# Validation contract — Topic 04

**Freeze date:** before the first G0 correction run.

This document separates:

- **measurement feasibility** (G-1),
- **discovery** (G0-D),
- **locked confirmation** (G0-C),
- **robustness/replication** (G0-R),
- **durability** (G1).

The purpose is to prevent the failure mode seen in earlier candidate topics: searching a large grid of metrics/settings, selecting a promising cell, and treating it as if it were a preregistered phenomenon.

---

# 1. Scientific question

Among initially wrong items with matched accessibility of the correct target:

> **Does stronger commitment to one specific wrong hypothesis change the speed or durability of corrective learning?**

The study does **not** assume the direction in advance.

---

# 2. Primary stimulus pool

## 2.1 Primary

**MMLU-Pro test split, exactly 10 answer options.**

The split is used as an experimental stimulus pool. No benchmark-generalization claim is made.

Reasons to prefer K=10:

- richer wrong-distribution geometry;
- enough items for tight matching;
- fixed K avoids entropy/concentration comparability problems;
- lower reported prompt-format sensitivity than original MMLU.

## 2.2 External replication pools

Only after the primary confirmation:

- MMLU (K=4),
- ARC-Challenge (usually K=4),
- OpenBookQA (K=4),
- MedMCQA (K=4).

Do not mix variable-K items into the primary MMLU-Pro analysis.

---

# 3. G-1 — measurement feasibility

## 3.1 Base model

Primary:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Optional measurement replication before G0 if resources are abundant:

```text
Qwen/Qwen2.5-3B-Instruct
```

The same scoring prompt/template and metrics must be used.

## 3.2 Candidate schema

```json
{
  "id": "mmlu_pro:123",
  "dataset": "mmlu_pro",
  "category": "physics",
  "question": "...",
  "choices": ["...", "..."],
  "answer": 4
}
```

Requirements:

- unique correct answer;
- primary pool has exactly 10 choices;
- no duplicate IDs;
- no answer leakage introduced by preprocessing.

## 3.3 How option probabilities are measured

Use the instruct model's **chat template**.

The user message contains:

```text
Question text

Options:
A. ...
B. ...
...

Choose the single best option. Reply with only its letter.
```

The scorer computes the conditional log probability for each allowed answer label and renormalizes over the answer-label set.

Important implementation requirements:

1. use `tokenizer.apply_chat_template(..., add_generation_prompt=True)`;
2. do not re-tokenize `prompt + candidate` in a way that changes the prompt boundary;
3. if all answer labels are single tokens, gather them from one next-token distribution;
4. otherwise fall back to exact candidate-sequence scoring;
5. normalize only over the permitted labels.

## 3.4 Option-position control

For K options, use K cyclic rotations:

```text
0 1 2 ... K-1
1 2 3 ... 0
...
```

Thus every semantic choice occupies every answer-label position exactly once.

For each rotation:

1. score label probabilities;
2. map them back to semantic choice identities.

Then average semantic probabilities across rotations.

Primary semantic-stability gate for K=10:

```text
the same semantic top-wrong option must be top-wrong in >= 8/10 rotations
```

For K=4 replication:

```text
>= 3/4
```

## 3.5 Variables

Let averaged semantic probabilities be \(p_i\).

### Target accessibility

\[
a = p_{\text{correct}}
\]

### Wrong distribution

\[
q_j=\frac{p_j}{1-a},\quad j\neq\text{correct}
\]

### Primary commitment

\[
c_{\max} = \max_j q_j
\]

### Robustness commitment

\[
c_H=1-\frac{H(q)}{\log(K-1)}
\]

### Additional fixed diagnostics

Record, but do not promote post hoc to the primary metric:

- top-wrong semantic identity;
- top-wrong probability;
- wrong top-1 / top-2 margin;
- normalized wrong entropy;
- overall answer entropy;
- target rank;
- option-position agreement;
- question token count;
- correct-answer token count.

## 3.6 Initially wrong gate

Primary items satisfy:

```text
argmax averaged semantic probability != correct answer
```

Do not select items from sampled verbal answers. Selection is entirely from the frozen probability measurement above.

## 3.7 Pair construction

Define high/low commitment using the upper/lower 30% of `c_max` among eligible wrong items.

Pair one high item to one low item with:

```text
same category if feasible
|p_correct_high - p_correct_low| <= 0.02
question token length ratio <= 1.35
correct-answer token length ratio <= 1.50
same K
```

Use optimal / minimum-cost matching within category when possible, not outcome-aware manual pairing.

The matching cost may depend only on frozen **pre-training** covariates.

## 3.8 G-1 pass criteria

Strong pass:

```text
>= 600 matched pairs
mean |Δ p_correct| <= 0.010
median |Δ p_correct| <= 0.010
mean c_max separation >= 0.10
>= 90% pairs same category
```

Minimal pass:

```text
>= 300 matched pairs
mean |Δ p_correct| <= 0.015
mean c_max separation >= 0.08
```

Measurement failure:

- < 200 pairs;
- position stability fails broadly;
- wrong commitment collapses to a narrow range;
- high/low groups remain systematically unmatched on target accessibility.

If G-1 fails, **do not inspect correction dynamics**. Fix/abandon the measurement first.

## 3.9 Prompt robustness audit

Before G0, rerun a deterministic 20% sample under the predeclared alternate prompt:

```text
Which option is correct? Return only the option letter.
```

Pass if:

- Spearman correlation of `c_max` between primary/alternate prompts >= 0.70;
- semantic top-wrong identity agrees >= 75%;
- pair membership is not catastrophically unstable.

This is a measurement audit only. The primary prompt remains fixed.

---

# 4. G0 design — corrective learning

## 4.1 Why one exposure per cycle

A **correction cycle** means every selected semantic item receives exactly one supervised corrective exposure.

Primary MMLU-Pro uses 10 cycles because K=10. The training-data builder rotates option position so that, across 10 cycles, each semantic answer occupies each label position exactly once.

This prevents "cycle" from secretly meaning different numbers of exposures for different items.

## 4.2 Training target

Input:

```text
question + rotated options
```

Assistant target:

```text
Answer: <rotated label>. <correct answer text>
```

Loss is applied only to assistant tokens.

Why include answer text:

- supervision is semantic, not only a letter;
- the correct content remains invariant across rotations.

Why retain the rotated label:

- the model also learns the current option mapping;
- fixed-letter memorization is impossible because the same semantic answer rotates.

## 4.3 G0 data split

Pairs, not individual items, are split.

Primary:

```text
70% discovery
30% locked confirmation
```

Stratify by category.

### Important

Run discovery and confirmation as **separate fine-tuning jobs from the same base checkpoint**.

The confirmation subset is not used to choose:

- direction;
- learning rate;
- metric;
- number of cycles;
- threshold;
- prompt.

## 4.4 Primary training recipe

Initial locked recipe:

```text
model             Qwen/Qwen2.5-1.5B-Instruct
optimizer         AdamW
learning rate     1e-5
weight decay      0.0
precision         bf16
cycles            10
max length        1024
gradient clipping 1.0
scheduler         constant
training          full-parameter
```

Use three independent order/random seeds if possible:

```text
17, 29, 43
```

Each seed can run on a separate GPU/node. Do not use slow cross-node DDP.

### LR sanity condition

The primary `1e-5` is not changed based on the high-vs-low effect.

If aggregate learning is obviously broken:

```text
mean p(correct) barely changes by cycle 10
```

or instantly saturated:

```text
>95% of all items are top-1 correct after cycle 1
```

then the whole G0 run is an **optimization-invalid run**, not evidence for/against the scientific hypothesis.

In that case, use an unrelated calibration subset to compare exactly:

```text
5e-6, 1e-5, 2e-5
```

Choose a recipe based only on aggregate learning geometry, then restart G0 from scratch and document the change.

Do not choose LR based on which gives the desired group difference.

---

# 5. G0 evaluation

Evaluate:

```text
cycle 0 (base)
cycle 1
...
cycle 10
```

using the same permutation-robust semantic scorer from G-1.

For each item and cycle record:

```text
p_correct
p_old_wrong
top1_correct
correct_rank
answer_entropy
```

The `old_wrong` identity is frozen from G-1 and never redefined after training.

---

# 6. Primary endpoint and statistics

## 6.1 Primary endpoint

For item \(i\), define correction gain relative to its own frozen base state:

\[
G_i=\frac{1}{10}\sum_{e=1}^{10}
\left[p_{i,e}(\text{correct})-p_{i,0}(\text{correct})\right]
\]

Primary paired effect:

\[
\Delta_G
=
G_{\text{high-commitment}}
-
G_{\text{low-commitment}}
\]

Raw probability AUC is reported descriptively, but **AUC gain is primary** because it measures learning from each item's own starting point.

within matched pairs.

Report:

- mean paired difference;
- 95% pair-cluster bootstrap CI;
- each seed separately;
- pooled mean across seeds.

## 6.2 Secondary endpoints fixed in advance

### Immediate correction

\[
\Delta_1=p_1(\text{correct})-p_0(\text{correct})
\]

### Early correction

Mean gain over cycles 1–2.

### Late correction

Mean \(p(\text{correct})\) over cycles 8–10.

### Behavioral correction time

`T_top1`:

```text
first cycle where correct option becomes semantic top-1
and remains top-1 at the next evaluation
```

This replaces an arbitrary `p >= .5` threshold.

### Original-error suppression

\[
S=p_0(y_{\text{old wrong}})-p_{10}(y_{\text{old wrong}})
\]

### Continuous model

Predeclared regression:

```text
AUC_correct
  ~ wrong_concentration
  + base_p_correct
  + wrong_concentration * base_p_correct
  + question_token_count
  + correct_answer_token_count
  + category fixed effects
```

Cluster SE / bootstrap by matched pair.

The interaction is secondary. Do not search additional nonlinear terms unless clearly labeled exploratory.

---

# 7. Discovery → confirmation decision

## 7.1 Directional result

A candidate directional effect proceeds to confirmation if discovery shows:

- same sign in >= 2/3 seeds;
- paired bootstrap CI for pooled \(\Delta_G\) excludes 0;
- absolute mean \(|\Delta_G| >= 0.02\).

The 0.02 threshold is a **screening threshold**, not a universal scientific constant.

## 7.2 Locked confirmation

Confirmation succeeds if:

- same direction as discovery;
- confirmation 95% CI excludes 0;
- effect is not driven by one category or one seed;
- primary prompt-position diagnostics remain healthy.

If discovery is strong and confirmation flips sign or collapses near zero:

```text
KILL the directional claim.
```

Do not rescue with another threshold, entropy metric, prompt, or hand-picked domain.

## 7.3 Equivalence-style null

If discovery suggests near-zero effect, do not claim "commitment does not matter" from `p > .05`.

A potentially interesting **accessibility-dominance** result requires:

1. a prespecified smallest effect of interest (screening default ±0.02 AUC);
2. the 90% CI on \(\Delta_G\) lies entirely inside [-0.02, +0.02];
3. the same equivalence pattern holds in locked confirmation;
4. preferably replicate on a second model/dataset.

Only then is "target accessibility dominates wrong commitment" worth serious analysis.

---

# 8. Predeclared alternative natural phenomena

The user goal is rapid scientific triage, not forcing one directional hypothesis. The following patterns may motivate a **new natural question** if they are strong and replicated.

## 8.1 Early–late reversal

Example:

```text
high commitment learns faster at cycle 1–2
but ends lower / relapses more by cycle 8–10
```

Natural question:

> Does surprise accelerate initial correction while entrenched memory preserves the old error long-term?

This is especially relevant to human hypercorrection / return-of-error findings.

## 8.2 Accessibility × commitment interaction

Example:

```text
commitment matters only when p(correct) is moderate,
not when the target is nearly inaccessible
```

Natural question:

> Does conviction matter only after the learner already has partial access to the correction?

This connects directly to the prior-knowledge account of hypercorrection.

## 8.3 Correct-target learning and misconception suppression dissociate

Example:

```text
p(correct) rises similarly,
but the original wrong answer decays at different rates
```

Natural question:

> Is learning a correction different from unlearning the misconception it replaces?

This can be pursued even if final accuracy is similar.

## 8.4 Domain-specific effect

Only take seriously if:

- visible in >=2 seeds;
- replicated on held-out pairs;
- not found by scanning dozens of domains for the best result.

A strong domain interaction may suggest that structured misconceptions differ from diffuse ignorance.

---

# 9. G1 — durability / return of error

Run only after a reproducible G0 result or a strong early–late pattern.

## 9.1 Corrected set

Freeze items that are semantic top-1 correct at cycle 10.

## 9.2 Interference phase

Starting from the corrected checkpoint, train on a fixed unrelated instruction/filler corpus.

Predeclare budgets, e.g.:

```text
0, 250, 1000, 4000 optimizer steps
```

Do not include the original correction questions or obvious paraphrases.

## 9.3 Outcomes

At each budget evaluate:

```text
p_correct
p_original_wrong
top1 identity
```

Primary durability event:

```text
original wrong semantic option becomes top-1 again
```

This is stronger than simply losing the corrected answer.

Interesting pattern:

```text
high commitment corrects quickly
but old errors return more often after interference
```

This would parallel a classic distinction between immediate hypercorrection and persistence of entrenched errors.

---

# 10. G2 — external replication

Priority:

1. Qwen2.5-3B-Instruct on the same MMLU-Pro protocol;
2. original MMLU / MedMCQA as fixed-K=4 domain replication;
3. different model family if available.

For K=4 use four cyclic rotations and the `>=3/4` semantic-stability gate.

Do not pool K=10 and K=4 commitment values without normalization and a predefined cross-K analysis.

---

# 11. Final kill rules

The topic should be archived rather than rescued if:

1. G-1 measurement is unstable.
2. Pair matching cannot separate commitment from accessibility cleanly.
3. Discovery does not survive locked confirmation.
4. The sign is learning-rate or prompt-template specific without a stable explanation.
5. "Null" results are too imprecise for equivalence.
6. The effect is entirely a label-position artifact.
7. A recent direct paper closes the exact gap.

No post-hoc:

- hidden-state probe;
- layer sweep;
- alternative "confidence" measure selected by outcome;
- arbitrary category cherry-picking;
- model shopping for a positive sign.

A genuinely new external observation can justify a new topic, but it should be registered as a new hypothesis rather than used to rewrite this one.
