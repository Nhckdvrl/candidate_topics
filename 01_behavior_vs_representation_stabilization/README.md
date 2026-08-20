# 01 — Behavior Stabilization vs. Representation Stabilization

## Status

**Candidate topic — audited G0 implementation ready.**

The broad question, “do representations keep changing after behavior stabilizes?”, is too close to existing Pythia training-dynamics work such as PolyPythias and concept-evolution studies. The version worth testing is narrower:

> **After global output-distribution movement reaches a low late-training regime, does feature-level representational learning continue in a way that cannot be explained by trivial residual-coordinate drift?**

The purpose of the current code is not to prove a paper claim. It is to kill or keep this topic quickly.

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

The interesting gap is instead:

> **Can global function-space convergence hide continued interpretable feature learning?**

If G0 only finds generic activation drift, the topic is not strong enough.

---

## 2. What we want to know

Let

- `D_behavior(t, Δ)` measure local output-distribution movement;
- `D_repr(t, Δ)` measure movement of matched residual representations;
- `Δ = 1,000 training steps` for every pair.

We test whether the **relative decay** of the two quantities differs:

```text
behavior movement falls strongly
while
meaningful representation movement remains elevated
```

The strongest eventual story would be:

```text
global behavioral convergence
        ↓
hides continued sparse-feature emergence / reorganization
        ↓
those late features encode local or task-relevant functional changes
```

That final feature-level step is not part of G0. G0 only decides whether it is worth paying for crosscoder experiments.

---

## 3. Audited validation design

The validation is deliberately split into two gates.

### G0-A — reproduce the behavioral premise first

Do **not** inspect representation results until this passes.

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

This avoids the previous invalid comparison where checkpoint intervals grew from 1k to 16k.

Data:

- 1,000 fixed UTF-8 byte chunks;
- approximately 1,024 bytes per chunk;
- generated once from `NeelNanda/pile-10k` using reservoir sampling;
- the exact corpus is hashed and reused by every checkpoint;
- no silent tokenizer truncation is allowed.

Behavior metrics:

1. raw local KL proxy in bits/byte;
2. robust KL sensitivity check:
   - lower 2% log-likelihood clipping per checkpoint;
   - remove the top 3% examples by maximum pairwise LL-change score;
3. 95% cluster bootstrap CIs over examples.

G0-A passes only when:

- the fixed-horizon KL trajectory clearly enters a lower late-training regime;
- raw and robust curves tell the same qualitative story;
- the result is not driven by a handful of pathological text chunks.

If this premise does not reproduce, stop and debug the behavior measurement. Do not interpret representations.

### G0-B — cheap representation screen

Only after G0-A passes.

To keep the screen fast, the default uses one explicit residual-stream location:

```text
middle GPT-NeoX block, resid_pre
```

This matches the type of hook used by Crosscoding Through Time more closely than relying on an ambiguous `hidden_states[k]` tuple.

For each text we sample four deterministic interior token positions and compare matched states across the same fixed-1k checkpoint pairs.

We report three complementary metrics:

1. **matched cosine drift**
   - sensitive to direct movement of the same state in the same lineage;
2. **pooled-standardized drift**
   - dimensionless matched displacement after pairwise feature standardization;
3. **projected linear CKA**
   - a rotation/scale-tolerant geometry control.

CKA is **not** a falsifier by itself. A representation can rotate while preserving CKA, and sparse features can change even when global second-order geometry remains similar.

Bootstrap:

- cosine / standardized drift: 500 cluster bootstraps by text;
- CKA: a small bootstrap after fixed random projection to 128 dimensions, because CKA is only a control in G0.

---

## 4. Why these measurements are deliberately redundant

A useful result must survive multiple explanations.

### Case A — only direct drift stays high, CKA ≈ 1

Likely interpretation:

```text
coordinate / basis / scale drift
```

This is weak and does **not** justify a paper.

### Case B — behavior stabilizes and matched + standardized drift remain high

Potentially interesting. Continue to G1, especially if the effect is reproducible under corpus resampling and additional layers.

### Case C — all representation metrics decay with behavior

Broad topic is likely dead. Given PolyPythias and existing representation-dynamics work, do not force a weak separation into a claim.

### Case D — systematic layer ordering appears

Potentially interesting, but only after replicating the middle-layer signal and then expanding to a small layer sweep.

---

## 5. G1 only if G0 survives

Do **not** train a large crosscoder grid immediately.

Pick only three snapshots around the transition found by G0, for example:

```text
before behavioral stabilization
near stabilization
late stable regime
```

Then reuse the Crosscoding Through Time machinery to ask whether sparse features still:

- emerge;
- disappear;
- persist;
- change causal relevance.

The real paper-level claim requires feature-level evidence. Geometry-level movement is only a screening signal.

The topic becomes genuinely strong only if we can show something like:

> global sequence-distribution movement is already small, but interpretable late features still reorganize and encode local functional changes that the global KL summary hides.

---

## 6. Quick start

```bash
cd 01_behavior_vs_representation_stabilization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Engineering smoke test

```bash
NUM_EXAMPLES=100 BATCH_SIZE=16 GATE=behavior ./run_pilot.sh
```

### G0-A — behavior premise

```bash
NUM_EXAMPLES=1000 BATCH_SIZE=16 GATE=behavior ./run_pilot.sh
```

Inspect:

```text
artifacts/analysis/behavior_metrics.csv
artifacts/analysis/behavior_outliers.csv
artifacts/analysis/behavior_constant_horizon.png
artifacts/analysis/behavior_summary.json
```

### G0-B — representation screen

Only after G0-A is convincing:

```bash
NUM_EXAMPLES=1000 BATCH_SIZE=16 POSITIONS_PER_TEXT=4 GATE=representation ./run_pilot.sh
```

Inspect:

```text
artifacts/analysis/representation_metrics.csv
artifacts/analysis/representation_constant_horizon.png
artifacts/analysis/behavior_vs_representation.csv
artifacts/analysis/behavior_vs_representation.png
artifacts/analysis/representation_summary.json
```

To run both gates in one shot after the pipeline has been smoke-tested:

```bash
GATE=all ./run_pilot.sh
```

---

## 7. Decision rule

| Result | Decision |
| --- | --- |
| Fixed-horizon KL premise does not reproduce | stop; implementation / corpus issue |
| KL stabilizes; all representation measures stabilize similarly | stop topic |
| KL stabilizes; only raw/direct activation movement remains | weak; likely coordinate drift; stop unless stronger evidence appears |
| KL stabilizes; matched + standardized geometry movement remains systematically elevated | continue to G1 |
| G1 also finds late sparse-feature turnover | topic is alive |
| Late features have local/task-functional effects despite low global KL | strong paper direction |

The repository intentionally contains no hard-coded “significance = continue” threshold. With 1,000 examples, tiny effects can be statistically significant. We care about **effect-size separation, robustness, and interpretability**, not p-value hunting.
