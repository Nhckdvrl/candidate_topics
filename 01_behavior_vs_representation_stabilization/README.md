# Behavior Stabilization vs. Representation Stabilization

## Status

Candidate topic. **High priority for Training Dynamics**, but only worth continuing if a cheap pilot shows a clear temporal decoupling between output behavior and internal representations.

---

## 1. Background

A recent Training Dynamics line suggests that different notions of "model change" can evolve on very different time scales.

### Seed paper 1: output-space stabilization

**Establishing a Scale for KL Divergence in Language Models Across Various Settings** (Findings ACL 2026) studies language-model training trajectories in output-distribution space. On Pythia checkpoints, the paper finds that model behavior measured through likelihood/KL geometry stabilizes relatively early, even though the model's weights continue to move substantially.

The key established phenomenon is therefore:

> **behavior can become nearly stable before parameter training has stopped.**

Paper: https://aclanthology.org/2026.findings-acl.1163/

### Seed paper 2: representation dynamics

**Evolution of Concepts in Language Model Pre-Training** (ICLR 2026) and **Crosscoding Through Time** (ACL 2026) show that internal features can be tracked across training checkpoints. Features can emerge, persist, change, or disappear over pretraining, and some of these feature changes are behaviorally or causally relevant.

- Evolution of Concepts: https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html
- Crosscoding Through Time: https://github.com/bayazitdeniz/crosscoding-through-time

These two lines leave a very natural missing comparison:

> **When the model's output behavior has already stabilized, have its internal representations stabilized as well?**

The topic is not simply "study representation dynamics". That has already been done. The one-step question is to compare the **time of behavioral stabilization** with the **time of representational stabilization** within the same training trajectory.

---

## 2. What we want to study

Let

- `t_behavior` = the point at which successive checkpoints become nearly indistinguishable in output-distribution space;
- `t_repr` = the point at which internal representations/features stop changing substantially.

The central question is:

> **How are `t_behavior` and `t_repr` related?**

Possible regimes include:

1. `t_behavior << t_repr`: behavior becomes stable while the inside of the model keeps reorganizing;
2. `t_repr << t_behavior`: the representation basis stabilizes first, while later training mainly changes readout/calibration;
3. `t_behavior ≈ t_repr`: internal and external convergence are synchronized;
4. there is no single global order, but different layers/features stabilize at systematically different times.

The most interesting result would be a period of **behaviorally silent representational reorganization**: output behavior changes very little, yet internal features continue to emerge, disappear, or change their organization.

---

## 3. Exact measurements

### 3.1 Behavioral change

Follow the KL-trajectory setup from the seed paper.

For successive checkpoints `t` and `t + Δ`, evaluate a fixed set of held-out texts and estimate:

`D_behavior(t) = KL(p_t || p_{t+Δ})`

The pilot only needs a stable relative trajectory; the goal is to locate the late-training regime where output-distribution movement has become very small.

### 3.2 Representation change: cheap pilot

Before training crosscoders or SAEs, use a cheap geometry-level screen.

For the same fixed texts and token positions, collect hidden states from a few layers and compute representation similarity between adjacent checkpoints, e.g. linear CKA:

`D_repr^l(t) = 1 - CKA(H_t^l, H_{t+Δ}^l)`

Suggested layers for the pilot:

- 25% depth
- 50% depth
- 75% depth
- final layer

CKA is only a screening tool. If a clean decoupling appears, the full study should move to feature-level measurements.

### 3.3 Representation change: full-paper measurement

Use cross-checkpoint crosscoders / sparse features to track:

- feature emergence;
- feature disappearance;
- feature persistence;
- feature reassignment/change;
- optionally, causally relevant feature turnover using RelIE-style measurements.

The important quantity is not generic parameter drift, but **meaningful feature turnover after behavioral stabilization**.

---

## 4. Minimal validation experiment

### Model

Start with **Pythia-410M**.

Pythia is ideal because it provides dense public pretraining checkpoints and is already used by the seed literature.

### Checkpoints

A first sparse trajectory is enough:

`1k, 2k, 4k, 8k, 16k, 32k, 48k, 64k, 80k, 96k, 112k, 128k, 143k`

### Data

Use a fixed set of **1,000 held-out Pile texts**.

For each checkpoint:

1. compute sequence log-likelihoods for all 1,000 texts;
2. save hidden states at four selected layers;
3. sample a fixed set of token positions, e.g. 16 positions per text;
4. compute adjacent-checkpoint KL movement;
5. compute adjacent-checkpoint CKA movement.

### Pilot outputs

The pilot should produce only three essential plots:

1. **Behavioral movement vs training step**
   - `D_behavior(t)`
2. **Representation movement vs training step**
   - `D_repr^l(t)` for several layers
3. **Behavioral movement vs representational movement**
   - each training interval as one point

Do not hand-pick a stabilization step in advance. If the curves support it, estimate `t_behavior` and `t_repr` using a change-point or piecewise-regression analysis.

---

## 5. Decision rule

### Continue

Continue only if at least one of the following appears clearly and reproducibly:

- behavior enters a stable regime while representation drift remains substantial;
- representation stabilizes much earlier than behavior;
- different layers show a strong and systematic stabilization order.

If this happens, move from CKA to crosscoder/feature-level analysis and replicate across model scales and seeds.

### Stop

Stop if:

- behavioral and representational movement simply decay together with no interesting separation;
- the apparent gap is tiny, unstable across text samples, or highly sensitive to the similarity metric;
- only parameter-space drift remains while feature-level representations are already stable.

Do **not** turn a weak CKA difference into a paper claim.

---

## 6. Why this topic may matter

If model behavior stops changing much earlier than its internal representations, then late pretraining is doing something that ordinary benchmark/output evaluation cannot see. The scientific question becomes:

> **What does a language model keep learning after its behavior appears to have stabilized?**

That would connect output-space training dynamics with interpretable feature dynamics without inventing a new benchmark or a new training algorithm.
