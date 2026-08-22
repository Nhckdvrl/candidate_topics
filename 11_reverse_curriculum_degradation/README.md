# 11 — Does a Model Forget Its Curriculum Backwards?

**Status:** REGISTERED CANDIDATE — CHEAP G-0 FIRST

## Natural question

> Are capabilities acquired later during pretraining also the first to become fragile when the trained model is structurally degraded?

A stronger formulation is:

> **Is model degradation approximately reverse development?**

This asks whether developmental order leaves a robustness hierarchy inside a trained model, rather than being only a transient property of learning curves.

## Seed evidence

### Seed A — language models exhibit a reproducible implicit curriculum

*What do Language Models Learn and When? The Implicit Curriculum Hypothesis* (COLM 2026) studies a large collection of elemental and compositional skills across multiple models and model families and reports highly consistent skill-emergence ordering, including a broad tendency for atomic skills to emerge before compositional skills.

This provides a frozen external quantity:

`E_i = emergence rank of skill i`.

The project should reuse this ordering rather than rediscovering a convenient emergence threshold.

### Adjacent evidence

Earlier learning-curve work has reported that earlier-learned token predictions can also be more stable during continued pretraining. Recent work additionally links altered early skill acquisition to downstream compressibility. These findings motivate the question but do not answer the proposed within-model, skill-level relation between **emergence order** and **structural degradation order**.

## Exact question

For skills that are all robustly mastered by the dense final model, ask whether later emergence predicts lower robustness under a single frozen degradation axis.

The project deliberately avoids continual fine-tuning / interference, because that introduces new-task exposure, optimization order, learning rate, rehearsal, and distribution-shift confounds.

## Avoid the double-threshold trap

Do **not** define both an arbitrary emergence threshold and an arbitrary degradation threshold.

Instead:

1. take skill emergence ranks from the seed study / a locked reproduction;
2. retain only skills satisfying one predeclared dense-model mastery floor;
3. apply one fixed degradation family, initially global magnitude pruning without retraining;
4. evaluate each skill over the entire predetermined sparsity sweep;
5. summarize robustness by the normalized area under the retained-performance curve.

For skill `i`:

`R_i = AUC_s [ performance_i(s) / performance_i(0) ]`.

Primary test:

`Spearman(E_i, R_i)`.

If later-emerging skills degrade earlier, the expected relationship is negative.

## Strong paired G-0

The cheapest and most interpretable first test uses dependency pairs already present in the implicit-curriculum taxonomy:

`A, B -> composite(A,B)`.

Restrict to cases where components and composite are all strongly mastered by the dense model. Then ask:

> Across the same pruning sweep, is the later-emerging composite systematically less robust than its already-mastered components?

This paired contrast is more important than a large heterogeneous cross-skill correlation.

## G-0 protocol

1. Freeze a small set of high-mastery atomic/compositional dependency groups.
2. Freeze sparsity levels, e.g. a monotonic grid from dense to clearly degraded.
3. Evaluate every skill at every level.
4. Plot normalized degradation curves together with the seed emergence order.
5. Report paired component-vs-composite robustness and the full rank correlation.

No re-training, no hidden representations, no per-skill threshold tuning.

## Interpretation

### Reverse-development pattern

If later-acquired/compositional skills consistently fail earlier under structural damage despite comparable dense mastery, pretraining developmental order appears to leave a persistent robustness hierarchy.

### No relationship

Then the implicit curriculum may describe acquisition dynamics without defining the model's later structural resilience. This is still a direct answer to the proposed relationship; do not rescue it by adding other corruption families after the fact.

### Reversed relationship

If later skills are more robust, that would be especially surprising and would argue against a simple developmental-age account of model organization.

## Kill line

The paired atomic/composite contrast is the first gate.

Kill the topic if a locked set of high-mastery dependency pairs shows no coherent ordering under the fixed pruning sweep. Do not add difficulty matching, token-length matching, category-specific thresholds, quantization variants, or alternative pruning algorithms to manufacture a correlation.

## What would make the result worth being excited about?

The interesting result is not merely "harder skills break first." It would need to show a reproducible ordering tied to **developmental acquisition rank** within a model:

> **the model loses capabilities in approximately the reverse order in which they emerged.**

Only a large, simple paired effect deserves follow-up across model families.
