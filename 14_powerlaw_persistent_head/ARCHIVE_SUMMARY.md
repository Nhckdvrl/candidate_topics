# Topic 14 archive summary

## Final status

**ARCHIVED — valid frozen full G0 returned `KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT`; G1 not run.**

Topic 14 asked whether the power-law advantage in compositional learning depends on a **temporally persistent head**: must the same skills remain high-frequency long enough to scaffold the rest, or is local asymmetry at each moment already sufficient?

The project reached a technically clean and scientifically interpretable negative. The motivating power-law phenomenon was not weak — it was extremely strong — but the proposed temporal-persistence explanatory axis was not stable across the locked replications.

The correct conclusion is therefore:

> **In the locked 4-hop S5 regime, power-law asymmetry strongly helps learning, but its main benefit does not generally require persistent head identity.**

This is a substantive null for the proposed mechanism, not a prerequisite or measurement failure.

[Full frozen G0 result](./G0_RESULT.md)

---

## Scientific question

> When power-law skill frequencies make a compositional task learnable, does the benefit come from a persistent implicit curriculum in which the same head skills are privileged for a long time, or from instantaneous/local distributional asymmetry?

The seed work reported a strong power-law advantage and stage-wise head-to-tail learning dynamics. Topic 14 isolated the temporal interpretation of that story.

The valuable positive result would have been especially sharp:

> give two learners the exact same finite training minibatches, but order them so one learner experiences long persistent head phases while the other rapidly alternates head identity; if learning changes dramatically, curriculum is a property of temporal organization, not only of the data histogram.

That made Topic 14 worth testing: the question was natural and the first decisive contrast was unusually clean.

---

## Final identification

For each seed, the clean G0 constructed one common 1000-step uniform warmup branch checkpoint and restored the exact same model + AdamW state for all four arms.

Power-law minibatches were deterministically keyed by:

```text
(seed, map_id, occurrence_id)
```

Slow order:

```text
A0 A1 ... A(P-1) B0 B1 ... B(P-1)
```

Fast order:

```text
A0 B0 A1 B1 ...
```

Thus Slow and Fast received the **same actual finite training minibatch multiset**. Matched quantities included:

- learner initialization at the branch point;
- AdamW state;
- batch contents and labels;
- A/B maps;
- number of A/B batches;
- total compute;
- post-branch constant LR;
- frozen uniform evaluation panel.

Only temporal ordering / head persistence changed.

Across the five locked replication seeds, the base rank-to-skill mapping also changed according to a predeclared seed rule. This prevented one arbitrary S5 skill assignment from being mistaken for a general mechanism.

This design is important for the archive interpretation: the project did not fail because the intervention was non-identifying or because different arms saw different data.

---

## Frozen full G0

All 5 seeds and all 20 arms completed and passed integrity checks.

### Prerequisite

The clean-regime static power-law advantage was enormous:

```text
median Static - Uniform exact-AUC = +0.9300
positive seeds                    = 5/5
```

So the scientific object was clearly present. There is no need to invoke the separate paper-faithful reproduction diagnostic to explain the negative result.

### Primary persistence contrast

```text
seed 0   Slow - Fast = -0.0325
seed 1   Slow - Fast = +0.7106
seed 2   Slow - Fast = +0.0095
seed 3   Slow - Fast = +0.0340
seed 4   Slow - Fast = -0.0395

median                 +0.0095
near-zero seeds         4/5   (|gap| <= 0.06)
```

The preregistered final decision was:

```text
KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT
```

G1 was not run.

---

## Why this is an informative null

This is not a weak or underpowered null in the way many failed candidates are.

Three facts make it unusually informative.

### 1. The prerequisite was extremely strong

Static power law beat uniform by a median AUC of `0.93`. Therefore Slow/Fast were evaluated in exactly the regime where power-law asymmetry mattered dramatically.

The result cannot be summarized as "nothing learned" or "the seed phenomenon failed to reproduce."

### 2. Four independent paired replications are individually near zero

The small median is not produced by averaging several huge positive and negative effects. Four of the five locked seeds satisfy the explicit per-seed near-zero criterion.

That is precisely the pattern needed to support the statement that persistent head identity is usually not doing much in this regime.

### 3. The intervention changed only time order

Slow and Fast saw the same finite data multiset from the same learner state under the same optimizer/LR/evaluation contract.

So the negative directly constrains the temporal-persistence explanation rather than a loose proxy for it.

---

## The seed-1 outlier

Seed 1 showed a very large Slow advantage (`+0.7106`). It should be preserved in the record, not hidden.

But it should also **not** be used to continue Topic 14.

The full replication design deliberately varied the arbitrary rank-to-skill assignment across seeds. If a general persistence mechanism were large enough to support a project, it should not disappear in four of five such assignments.

The most plausible status of seed 1 is therefore:

> an assignment-/optimization-specific interaction that may be real, but is not evidence for the project-level law that persistent head identity generally explains the power-law benefit.

Following seed 1 now would require asking which maps, head subsets, algebraic properties, or schedules produce persistence. That is outcome-conditioned search.

There is one legitimate future path: if an **independent external observation** suggests a specific structural property of the frequent skill set — for example generator/subgroup coverage — should control whether a persistent head can scaffold the tail, register that as a new topic with a frozen structural hypothesis. Do not reopen Topic 14 by sweeping maps until the outlier repeats.

---

## What the result says about the seed-paper mechanism

The result narrows the interpretation of stage-wise power-law learning.

A simple reading of "head skills learn first, then tail skills accelerate" is that the same head must stay privileged long enough to form a scaffold. Topic 14 tested that temporal reading directly and did not find a robust effect.

The more defensible interpretation after G0 is:

> **distributional asymmetry can strongly change optimization even when the identity of the privileged skills changes rapidly; a persistent named head is not generally necessary for the main benefit.**

This does not decide every possible mechanism behind power-law learning. It does eliminate one particularly natural and experimentally separable interpretation.

---

## Why the project stops here

The topic had the properties we want from a candidate:

- natural one-sentence scientific question;
- strong established prerequisite phenomenon;
- simple causal manipulation;
- same-data/different-order identification;
- no hidden-state/probe dependence;
- a positive result would have been genuinely surprising;
- a null result answers the question rather than erasing the experimental object.

It nevertheless stops because the proposed explanatory variable did not generalize across the locked replications.

Continuing by sweeping persistence lengths, alpha, maps, head definitions, architectures, or group-structure descriptors would no longer be testing the registered question. It would be searching for a regime where a failed general mechanism becomes true.

This is exactly the kind of negative result the candidate-selection process should accept cleanly.

---

## Transferable lessons

1. **A very strong prerequisite makes a mechanistic null more valuable.** When the base phenomenon has a huge effect but the proposed explanatory manipulation is near zero, the negative constrains the explanation rather than merely exposing a weak testbed.
2. **Randomize arbitrary identities when the claim is meant to be general.** If a mechanism depends on a randomly assigned mapping, replicate across mappings rather than only across model initialization. Otherwise one favorable assignment can masquerade as a general law.
3. **Use robust aggregation to prevent one spectacular seed from creating a project.** Here the arithmetic mean would be dominated by seed 1, while the frozen median plus per-seed near-zero requirement correctly reflected the replication pattern.
4. **An outlier is not automatically a new phenomenon.** Following a single post-hoc outlier is justified only when there is an independently motivated variable that predicts it. Otherwise it is regime search.
5. **A good topic can fail cleanly.** Topic quality and hypothesis truth are different. Topic 14 had a natural question and an unusually strong identification; the hypothesis simply did not hold broadly enough.
6. **Do not run mechanism-characterization experiments after the mechanism failed G0.** The persistence-timescale sweep was correctly reserved for a replicated G0 effect and therefore was not run.

---

## Reopen condition

Do not reopen Topic 14 by trying:

- more random A/B mappings;
- alternate head fractions;
- different alpha values;
- different persistence lengths simply to find a separation;
- architectures/models chosen after seeing this null;
- secondary metrics chosen because seed 1 looks interesting.

Reopen only if independent evidence creates a new, specific prediction for **when** head identity should matter. That should be registered as a new topic, not treated as continuation of the failed general-persistence claim.

---

## Preserved assets

Keep the complete validation implementation, locked config, design history, tests, G1 launcher, and server outputs for reproducibility.

The G1 launcher remains in the archived folder only as a record of the preregistered follow-up; the archive decision explicitly closes it under the current hypothesis.
