# 11 — Does Functional Necessity Predict Causal RL Adaptation Leverage?

**Status:** REGISTERED CANDIDATE — CHEAP G-0 FIRST

## Natural question

> If a component is especially necessary for a learned capability, is that also the component where changing parameters most effectively improves that capability?

For reasoning models:

> Are the transformer layers that are causally necessary for mathematical reasoning also the layers with the greatest causal adaptation leverage under RLVR?

This separates two notions that are often conflated:

- **functional necessity**: how much inference performance is damaged when layer `l` is removed/ablated;
- **causal adaptation leverage**: how much reasoning improvement can be obtained when RL is allowed to update only layer `l`.

The topic does **not** equate raw weight-change magnitude with learning leverage, and does **not** claim that reasoning literally "lives" in any single layer.

## Seed evidence

### Seed A — layer necessity is highly non-uniform and stable

*Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training* (2025) reports that mathematical reasoning is disproportionately sensitive to a small set of layers, and that this layer-importance structure remains broadly stable across base, instruction-tuned, RL-tuned, and distilled variants.

This provides:

`I_l = inference-time functional necessity of layer l`.

### Seed B — single-layer RL leverage is also highly non-uniform

*Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training* (2026) reports that updating only one transformer layer can recover a large fraction of full-parameter RL gains, with strong layer-to-layer variation and especially high leverage in some middle layers.

This provides:

`C_l = causal RL adaptation leverage of layer l`.

### Important literature collision already known

Prior work such as *Superficial Self-Improved Reasoners Benefit from Model Merging* (EMNLP 2025) has compared reasoning-layer importance with **parameter-change magnitude** and observed that reasoning-critical layers need not be the layers that move most during self-improvement.

That is related but does not answer the causal question here. The topic is only interesting if `C_l` is measured by actually restricting learning to layer `l`, not by observing how much a layer happened to move during full training.

## Exact question

For a matched model/task family, compare the complete layer-wise curves:

- `I_1 ... I_L`: locked inference ablation damage;
- `C_1 ... C_L`: locked single-layer RL gain.

Primary question:

> Does layer-wise functional necessity predict layer-wise causal adaptation leverage?

## G-0: one figure, no layer fishing

Use a model/task setup for which a complete single-layer-RL curve exists or can be reproduced cheaply.

1. Freeze one layer-ablation protocol before running the sweep.
2. Sweep **every** transformer layer once to obtain `I_l`.
3. Align with the complete single-layer RL curve `C_l`.
4. Plot `I_l` against `C_l` and show both full depth-wise curves.

Primary summary:

`Spearman rho(I, C)`.

A small predeclared top-k overlap may be descriptive only.

There is no discovery over "best" layers: the complete depth sweep is the experiment.

## Interpretation

### Strong positive association

Reasoning-critical computation is also the most plastic / highest-leverage target for RL adaptation.

### Strong dissociation

Where reasoning is most functionally fragile is not where learning can most efficiently improve it. This separates **computation** from **adaptation leverage**.

### Negative association

The most indispensable parts of the reasoning computation may be comparatively resistant to useful adaptation, while less indispensable layers act as control/routing/adaptation interfaces.

## Kill line

This is intentionally a cheap G-0.

Kill the topic if the complete matched-model curve shows no large, stable, interpretable relation or dissociation worth explaining. Do **not** rescue it by searching alternative layer subsets, thresholds, tasks, or ablation definitions after seeing the result.

A strong result should be visible in the full curve before any mechanism work.

## What would make the result worth being excited about?

Not "layer 17 is important." The interesting result would be a reproducible architectural principle across independent model families:

> **functional necessity and causal learning leverage are systematically aligned or systematically dissociated.**

Only after G-0 establishes a large clean relationship should replication or mechanism analysis begin.
