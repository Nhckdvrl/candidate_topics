# Is Negative Behavioral Adaptation Intrinsically Harder for LLMs?

**Status:** `PROVISIONAL SURVIVOR — ROUND 02`

This is not a registered numbered Topic. It survived the first phenomenon / confound / collision audit and is retained for a matched G-0 experiment.

---

## One-sentence question

> **When the target behavior, exposure, feedback strength, and test are matched, are LLMs intrinsically worse at learning to suppress an action from negative experience than at learning to select an action from positive experience?**

Short form:

> **Can LLMs learn “do this” much more easily than “don’t do this again”?**

The goal is not generic agent memory. The object is a very specific **positive acquisition vs negative inhibition asymmetry**.

---

## Seed phenomenon

The direct seed is Qin et al. (ACL 2026 Best Resource Paper), **[ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](https://aclanthology.org/2026.acl-long.1301/)**.

The benchmark evaluates whether experience changes later behavior without an explicit reminder. Across 17 models it reports a striking asymmetry:

```text
inhibitory adaptation: 17.6%
preference adaptation: 75.0%
mean gap: 57.4 points, p < 0.001
```

The gap persists across model families. Particularly severe examples include jargon avoidance at roughly 4%.

This is an unusually strong phenomenon-first starting point: the abnormal behavior has already been demonstrated across many modern systems.

---

## Why the seed does not yet identify “inhibition” as the cause

This is the key attack that keeps the proposed project from being a benchmark extension.

In the seed paper's Figure 4, the inhibition and preference conditions are compared using **different task families**:

```text
jargon avoidance   vs directory preference
API distrust       vs filetype preference
context behavior   vs brevity preference
```

Therefore two things change simultaneously:

```text
feedback sign / suppress-vs-select
AND
task object / pretrained default / response form
```

A 57-point gap can therefore arise because:

- negative behavioral adaptation is genuinely harder;
- the inhibitory tasks oppose much stronger pretrained defaults;
- the preference tasks happen to have simpler output spaces;
- some inhibitory tasks require semantic negation or warning behavior;
- the task families differ in prior probability, surface form, or evaluation difficulty.

The original paper establishes the **anomaly**, but its broad causal interpretation remains underidentified.

This is exactly the kind of situation where one clean matched contrast is better than another large benchmark.

---

## Exact scientific contrast

Construct an initially arbitrary binary behavior with no natural preferred answer:

```text
A vs B
```

Use the same action vocabulary and the same interaction template in both conditions.

### Positive condition

Experience teaches:

```text
A → success
B → neutral / failure
```

Test whether the model selects `A`.

### Negative condition

Experience teaches:

```text
A → failure
B → neutral / success
```

Test whether the model avoids `A` and selects `B`.

Across counterbalanced items, swap the identities of A/B and the surface labels.

The critical comparison is:

```text
P(select learned-positive action)
vs
P(avoid learned-negative action)
```

with every other property matched.

---

## Why this is scientifically interesting

In associative learning and cognitive science, **conditioned inhibition** is a real and historically difficult construct: a cue can come to predict the absence of an otherwise expected outcome and suppress responding. Theoretical debates have long asked whether inhibition is a distinct learning process or can be explained by ordinary associative competition.

For LLMs, the practical version is basic:

- a useful long-horizon system must not only acquire successful habits;
- it must also stop repeating behaviors that experience shows are bad;
- explicit memory of “that failed” is not equivalent to automatic behavioral suppression on the next relevant case.

If a large matched inhibition gap exists, this is a stable capability question, not a bug in one agent framework.

---

## Collision audit

### Direct collision: ImplicitMemBench

It reports the asymmetry but does not isolate feedback sign with matched task pairs. This proposal is a direct identification follow-up, not a new benchmark competing with it.

### Learning-from-failure / reflection work

A large agent literature — including Reflexion and more recent structured-reflection / negative-trajectory training — asks how to improve performance after failure.

That literature is adjacent but usually changes the **learning mechanism** (reflection, memory, SFT/RL, trajectory filtering) and evaluates downstream success.

The proposed question comes before method:

> under matched in-context experience, is negative suppression itself weaker than positive acquisition?

If the answer is no, there is no reason to build an inhibition-specific method.

### Negative vs positive evidence literature

Human learning and language-acquisition work has compared positive and negative evidence for decades. It supplies conceptual precedent but does not answer the LLM behavioral-adaptation question.

### Collision boundary

Kill this candidate if a prior LLM study already uses **identity-swapped, same-action, same-outcome-magnitude positive/negative conditioning pairs** and demonstrates the same asymmetry across modern model families.

Do not preserve novelty by moving to one particular tool, jargon type, or agent environment.

---

## Main conceptual attacks

### Attack 1 — negation-language confound

If the negative condition literally says “do not choose A,” a lower score may simply measure negation processing.

**Defense:** feedback should be conveyed through the **experienced consequence**, not an explicit negative instruction. For example, after action A the environment emits a matched failure outcome.

### Attack 2 — pretrained-prior confound

Suppressing familiar behavior is harder than selecting an arbitrary new behavior even if inhibition per se is normal.

**Defense:** G-0 must use arbitrary, balanced action symbols / tool names with empirical baseline choice close to 50/50. A separate later experiment can deliberately introduce strong priors.

### Attack 3 — outcome asymmetry

“success” and “failure” text may have different salience or token statistics.

**Defense:** counterbalance consequence wording and, where possible, use symmetric abstract outcome tokens whose valence is established within the episode.

### Attack 4 — test can be solved by explicit recollection

The model may narrate the prior episode instead of forming anything resembling automatic adaptation.

**Defense:** retain the seed paper's `Learning → Interference → Test` structure and first-attempt scoring; do not ask “what happened before?” at test time.

### Attack 5 — this may be ordinary recency / copying

A model might simply copy the most recent successful label.

**Defense:** balance presentation order, include interference, and ensure the positive/negative comparison uses identical event order distributions.

---

## Cheapest decisive G-0

### Dataset

Do **not** begin by expanding ImplicitMemBench.

Build a tiny controlled factorial set, around `40–80` base situations, each automatically expanded into counterbalanced variants:

```text
feedback sign:       positive / negative
action identity:     A / B
presentation order:  AB / BA
surface naming:      multiple arbitrary label pairs
```

The same item generator should produce both positive and negative versions.

### Models

Use several strong open / accessible model families rather than one giant sweep. The phenomenon should not depend on one vendor.

### Primary statistic

For each matched pair:

```text
Δ_inhibition = positive-adaptation accuracy - negative-adaptation accuracy
```

Report paired differences and model-consistency before any mechanistic analysis.

### Strong positive

A substantial negative-adaptation deficit remains after:

- matched action space;
- matched baseline preference;
- matched event count/order;
- identity swaps;
- symmetric consequence designs.

### Kill line

Kill the “fundamental inhibition bottleneck” framing if:

- the 57-point seed gap largely vanishes under matched pairs;
- sign effects reverse across equivalent surface realizations;
- the effect is explained by baseline action priors;
- only explicit natural-language negation produces a gap.

A null result is still publishable only if it convincingly shows that a high-profile benchmark-level asymmetry was a **task-family confound**; otherwise stop rather than invent a narrower mechanism.

---

## Natural second stage only if G-0 survives

If a genuine matched gap exists, then ask what controls it by varying one factor at a time:

1. strength of pretrained action prior;
2. number of positive/negative experiences;
3. delay / interference length;
4. whether an explicit memory statement is available;
5. inference vs lightweight finetuning.

Only after these behavioral facts should representation / activation work be considered.

---

## Why this is not generic Agent / Agentic RL

The project does not need a long-horizon agent benchmark, tool harness, RL algorithm, or frontier-scale training.

The scientific object is closer to a cognitive learning experiment:

```text
same behavior
same experience structure
one factor: acquire vs inhibit
one outcome: first subsequent action
```

Agent systems are merely one motivation for why the answer matters.

---

## Interestingness test

Assume the cleanest positive result:

> **Even when actions and outcomes are perfectly matched, modern LLMs robustly convert positive experience into future selection but fail to convert negative experience into future suppression.**

That would be a distinctive, simple, model-agnostic limitation with direct implications for experience-based adaptation.

Assume the opposite result:

> **The dramatic 17.6% vs 75% benchmark asymmetry disappears when the tasks are matched.**

That is also scientifically useful because it overturns a strong causal reading of a recent Best Resource Paper's headline diagnostic and tells the field not to treat “inhibition” as the underlying mechanism.

---

## Current verdict

`KEEP — HIGH PRIORITY`

This candidate currently has an especially attractive G-0 because the experiment is small, the falsification line is immediate, and the anomaly comes directly from a strong recent paper rather than from our imagination.