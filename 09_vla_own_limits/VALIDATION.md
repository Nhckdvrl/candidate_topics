# Validation contract

## Scientific object

We are not evaluating a new confidence method. We are testing an interpretation of an already observed phenomenon.

Known:

```text
hidden representation -> eventual success
```

Unknown:

```text
is the decoded signal mainly D(state),
or does it contain C(policy, state)?
```

The first experiment is designed so that `state` is literally shared inside each comparison.

## G0 — can the question be identified naturally?

### Data

Use released same-family pi0.5 LIBERO checkpoints:

```text
2k, 3k, 9k
```

Run all checkpoints on the **same** task / environment seed panel using the official evaluation path. Discovery and confirmation state seeds must be disjoint.

Record:

```text
task, seed, checkpoint, success
```

No hidden states are needed yet.

### Why two-way crossover is mandatory

Suppose 9k beats 2k on every disagreement state. Then a downstream readout can appear “self-aware” by outputting a constant prior that 9k is better. That does not identify state-dependent competence.

The usable contrast requires both:

```text
2k wins on some identical states
9k wins on other identical states
```

The discovery script ranks checkpoint pairs by:

```text
min(A_wins, B_wins)
```

not total disagreement.

### Frozen G0 stop rule

For the first behavioral panel, require at least **15 natural crossover states in each direction** for one same-family checkpoint pair. This threshold is about identifiability / sample support, not significance.

If no pair reaches it:

```text
STOP_NO_NATURAL_CROSSOVER
```

Do not create perturbations or train special checkpoints to rescue the topic.

If a pair reaches it, freeze that pair before G1 confirmation.

## G1 — paired relative success signal

### Representation

Use one predeclared hidden location rather than searching layers. Primary candidate: pi0.5 action-expert hidden state around **layer 11**, because prior COAST analysis already localizes success/failure structure there.

Before collection, audit the exact released checkpoint code so “layer 11” refers to the same tensor used by the prior work. If that mapping cannot be reproduced, mark the run technical-blocked rather than silently choosing another layer.

### Shared decoder

Fit one linear success decoder across both frozen checkpoints on discovery states:

```text
q = w^T h + b
```

The same `w,b` is applied to both checkpoints.

Never fit one separately calibrated probe for A and another for B in the primary test: separate probes can inject policy identity through their supervision and make cross-policy scores incomparable.

### Confirmation contrast

On independent crossover states:

```text
relative_score(s) = q_A(s) - q_B(s)
winner(s) = 1 if A succeeds and B fails, else 0
```

Primary metric:

```text
AUROC(relative_score, winner)
```

Secondary calibration-free metric:

```text
balanced accuracy of sign(relative_score)
```

Both outcome directions must be represented.

### Interpretation

A generic state-only feature shared by A and B contributes equally inside the pair and cancels in the score difference. A constant “checkpoint B is better overall” bias cannot solve bidirectional crossover.

A material positive result therefore means the representation changes in a state-dependent way that tracks **which policy will succeed from that same state**.

### G1 bar

Do not continue to mechanism work for a tiny above-chance effect. The intended bar is:

```text
relative AUROC >= 0.70
and bootstrap 95% lower bound > 0.60
```

on independent confirmation crossover states.

If the result is near chance or weak:

```text
KILL_SELF_KNOWLEDGE_INTERPRETATION
```

Do not sweep layers, probe classes, or confidence definitions to rescue it.

If it is strong:

```text
PASS_POLICY_SPECIFIC_SUCCESS_SIGNAL
```

Only then is layerwise mechanism localization justified.

## Stochastic-policy note

pi0.5 uses a generative action process. The local agent must first reproduce the released evaluation stack and determine exactly how inference RNG is seeded. The same published inference protocol must be used for every checkpoint.

Do **not** quietly average many reruns until the crossover pattern looks cleaner. If repeated rollouts are needed to define a stable success probability, that change must be made before G1 and applied uniformly to all checkpoints.

## Explicit anti-complexity rule

The topic gets one paired identification strategy.

If interpreting the result starts to require a growing list of nuisance controls (camera strata, task geometry, action entropy, hand-picked failure taxonomies, several probe heads, layer search), stop and reconsider the question rather than adding gates.
