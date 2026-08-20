# 01 — Behavior Stabilization vs. Representation Stabilization

## Status

**ARCHIVED / KILLED after G0.**

The behavioral premise reproduced cleanly, but the required behavior–representation temporal decoupling did not appear. Representation movement stabilized at least as fast as behavioral movement across matched cosine drift, standardized drift, projected CKA, and 30 deterministic corpus half-samples. No G1 crosscoder experiment was run.

See:

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — complete summary of what was tested, why the topic failed, and lessons for future topic selection;
- [`G0_RESULTS.md`](./G0_RESULTS.md) — exact experimental results;
- [`VALIDATION.md`](./VALIDATION.md) — preregistered validation logic and kill criteria;
- [`artifacts/analysis/`](./artifacts/analysis/) — committed analysis outputs and plots.

The original scientific question was:

> **After global output-distribution movement reaches a low late-training regime, does feature-level representational learning continue in a way that cannot be explained by trivial residual-coordinate drift?**

Final answer for the tested Pythia-410M middle-residual setup: **no evidence for the required decoupling; archive the topic.**

---

## 1. Background

### Seed 1 — output-distribution stabilization

**Establishing a Scale for KL Divergence in Language Models Across Various Settings** (Findings ACL 2026) studies Pythia checkpoints in a geometry built from sequence log-likelihoods. It shows that movement in output-distribution space becomes much smaller during training even while parameters continue to move.

Paper: <https://aclanthology.org/2026.findings-acl.1163/>

The key measurement is a local KL proxy derived from a double-centered checkpoint × text log-likelihood matrix:

```text
2 KL(p_i, p_j) ≈ ||q_i - q_j||² / N
```

The relevant comparison is between checkpoints separated by a **fixed training horizon**. Varying the checkpoint gap would confound movement rate with elapsed training.

### Seed 2 — representation / feature dynamics

**Evolution of Concepts in Language Model Pre-Training** and **Crosscoding Through Time** show that representations and sparse features can emerge, persist, consolidate, or disappear across pretraining checkpoints.

- Evolution of Concepts: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html>
- Crosscoding Through Time: <https://github.com/bayazitdeniz/crosscoding-through-time>

Crosscoding Through Time explicitly works with checkpoint pairs/triplets and a fixed residual-stream hook, then uses sparse crosscoders and RelIE-style attribution to study feature life cycles.

### Important nearby work / novelty constraint

**PolyPythias** already studies Pythia training phases, linguistic representations, and representational shifts across many pretraining runs. Therefore a paper cannot stop at “CKA changes later than benchmark performance” or “hidden states keep moving.”

The interesting gap was instead:

> **Can global function-space convergence hide continued interpretable feature learning?**

G0 did not provide the required positive residual-geometry premise, so the project stops before feature-level follow-up.

---

## 2. What we tested

Let

- `D_behavior(t, Δ)` measure local output-distribution movement;
- `D_repr(t, Δ)` measure movement of matched residual representations;
- `Δ = 1,000 training steps` for every pair.

The intended positive pattern was:

```text
behavior movement falls strongly
while
meaningful representation movement remains elevated
```

Instead, representation movement fell at least as fast as behavior movement.

---

## 3. Validation design

### G0-A — behavioral premise

Model:

```text
EleutherAI/pythia-410m
```

Fixed-horizon pairs:

```text
2k   → 3k
5k   → 6k
10k  → 11k
20k  → 21k
50k  → 51k
100k → 101k
142k → 143k
```

Every pair has:

```text
Δ = 1,000 steps
```

Data:

- 1,000 fixed UTF-8 byte chunks;
- approximately 1,024 bytes per chunk;
- generated once from `NeelNanda/pile-10k` using reservoir sampling;
- exact corpus hash reused by every checkpoint;
- no tokenizer truncation.

Behavior metrics:

1. raw local KL proxy in bits/byte;
2. robust KL sensitivity check with lower-tail clipping and top-3% movement-outlier trimming;
3. 95% cluster bootstrap CIs over examples.

Result: **PASS**. Robust behavior movement fell to `0.05367x` of its early value.

### G0-B — representation screen

One explicit residual-stream location:

```text
middle GPT-NeoX block, layer 12, resid_pre
```

For each text, four deterministic interior token positions were compared across the same fixed-1k pairs.

Metrics:

1. matched cosine drift;
2. pooled-standardized drift;
3. projected linear CKA as a geometry control.

Results:

```text
robust behavior late/early ratio = 0.05367
cosine drift late/early ratio   = 0.03019
standardized late/early ratio   = 0.03428
CKA movement late/early ratio   = 0.000370
```

Representation therefore did **not** retain more late-training movement than behavior. In 30 deterministic random half-samples of 500 texts, cosine or standardized drift retained more movement than robust behavior in **0 / 30** runs.

Result: **FAIL**.

---

## 4. Final decision

The preregistered negative case was:

```text
behavior stabilizes
+
representation stabilizes similarly or faster
=> kill topic
```

That is what happened.

```text
G0-A: PASS
G0-B: FAIL
G1: NOT RUN
FINAL: ARCHIVED
```

Do not restart the project by sweeping more layers, models, metrics, or crosscoder settings unless a genuinely new external observation changes the scientific premise.
