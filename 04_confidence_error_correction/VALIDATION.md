# Validation contract — Topic 04

This file defines the **first scientific test**, not a full-paper experiment plan.

## Primary question

Among initially wrong multiple-choice items with matched correct-answer accessibility, does stronger concentration on one specific wrong answer predict different corrective-learning dynamics?

## G-1: freeze the measurement before SFT

### Candidate item schema

JSONL:

```json
{"id":"mmlu_xxx","dataset":"mmlu","question":"...","choices":["...","...","...","..."],"answer":2}
```

Requirements:

- `len(choices) >= 4`;
- a unique correct option;
- no answer leakage in the question;
- no examples used to tune the scoring prompt after inspection of group outcomes.

### Balanced position control

For K=4, use four cyclic permutations: `0123`, `1230`, `2301`, `3012`.

Each semantic option occupies each answer label exactly once. Map label probabilities back to semantic choices, then average.

The primary item is retained only when the same semantic wrong answer is top-wrong in >= 3/4 permutations.

### Variables

Let averaged semantic option probabilities be `p`.

```text
target_accessibility = p[correct]
wrong_concentration = max(p[wrong]) / (1 - p[correct])
wrong_entropy_norm = H(p[wrong] / sum(p[wrong])) / log(K-1)
```

Primary conviction variable: `wrong_concentration`.

Robustness variable: `1 - wrong_entropy_norm`.

### Matching

Create high/low conviction groups from the upper/lower tercile of `wrong_concentration`, then greedily match within:

- same dataset/domain when possible;
- `|p_correct_high - p_correct_low| <= 0.03`;
- question token length ratio <= 1.25.

The confirmation analysis must also use `p_correct` as a continuous covariate.

### G-1 pass condition

Proceed only if:

- >= 300 pairs are available for the main pilot, or >= 200 pairs for a reduced pilot;
- matched mean absolute `p_correct` difference < 0.015;
- neither group is dominated by a single dataset;
- semantic top-wrong stability >= 75% by construction.

If this fails, **do not invent a different confidence score after seeing SFT results**.

## G0: corrective learning

Recommended first run:

```text
model       Qwen/Qwen2.5-1.5B-Instruct
items       400-600 matched pairs if available
training    full-parameter SFT preferred
epochs      8 correction cycles
seeds       3 shuffle/training seeds
evaluation  after every correction cycle
```

A correction cycle means every selected semantic item has received the same number of supervised exposures.

During training, rotate option order across exposures. The target is the **correct answer content**, not a permanently fixed label token.

### Primary endpoints

For item i at exposure cycle e:

```text
p_correct(i,e)
p_old_wrong(i,e)
```

Predeclare:

```text
AUC_correct(i) = mean_e p_correct(i,e)
T50(i)         = first stable e with p_correct >= .5
Suppression(i) = p_old_wrong(i,0) - p_old_wrong(i,last)
```

### Primary comparison

The main estimand is the coefficient / matched difference associated with **wrong conviction after conditioning on base target accessibility**.

A simple first analysis:

```text
AUC_correct ~ wrong_concentration + base_p_correct + dataset + length
```

and a pairwise bootstrap over matched pairs.

Do not use the best of many thresholds as the headline result.

### Positive controls

1. `base_p_correct` should predict correction speed at least weakly; otherwise scoring/training may be broken.
2. Training loss must decrease and aggregate training-set accuracy must rise.
3. Position-permutation agreement should remain high enough that label artifacts are not dominating.

### Locked confirmation

Before choosing any new metric from G0, hold out either 25% of matched pairs or a second dataset.

Freeze confidence definition, accessibility caliper, SFT recipe and primary endpoint, then one-shot confirm the direction/effect on the holdout.

## G1: durability

Only after G0 confirmation.

1. Identify items corrected by the final G0 checkpoint.
2. Train the model on unrelated filler text/instructions for a fixed budget.
3. Evaluate at several fixed filler budgets.
4. Score both correct-answer probability and **return probability of the original wrong answer**.

Primary outcome: `return_old_error = old_wrong becomes top option again`.

This is deliberately stronger than simply losing the corrected answer.

## Interpretation guardrails

- A result on prompt-level revision is not evidence about weight-level corrective learning.
- Raw max wrong probability is not the primary independent variable.
- Do not equate model probability with human subjective confidence without qualification.
- The human literature motivates a general learning question; this study does not require human-like mechanisms.
- A cross-model replication is required before any broad claim about "learning systems".
