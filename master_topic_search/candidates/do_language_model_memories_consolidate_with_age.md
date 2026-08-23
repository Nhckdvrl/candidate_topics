# Do Language-Model Memories Consolidate With Age?

**Status:** `PROVISIONAL SURVIVOR — ROUND 03`

This is not a registered numbered Topic. It survived the first seed-paper / collision / identification audit and is retained for a matched behavioral G-0.

The title uses “consolidate” only as an intuitive shorthand. The first experiment makes a narrower claim about **acquisition age / training-order recency and later overwrite resistance**. It does not assume biological memory mechanisms.

---

## One-sentence question

> **For facts that are equally well remembered at test time, does when they were learned predict how easily later training can overwrite them?**

Short form:

> **Are older learned facts harder—or easier—to rewrite than equally strong recent facts?**

---

## Direct seed paper

The seed is Krasheninnikov, Turner & Krueger, ICLR 2026, **“Fresh in Memory: Training-Order Recency Is Linearly Encoded in Language Model Activations.”**

The paper creates a controlled training history by sequentially fine-tuning a model on six disjoint but otherwise similar entity datasets.

It establishes a striking representation-level phenomenon:

- activation centroids for the six entity groups lie in their exact training order along an approximately linear direction;
- linear probes distinguish early-vs-late learned entities at roughly 90% accuracy and generalize to unseen entities;
- the model can be trained to explicitly report an unseen entity's training stage at roughly 80% accuracy;
- the signal is not explained by simple differences in activation magnitude, loss, or confidence;
- re-exposure moves old information toward the “recent” end of the direction;
- training-order information remains partly detectable even after later mixed training removes the explicit sequential curriculum.

So the surprising fact is already established:

```text
model activations carry information about when knowledge was acquired
```

The seed explicitly notes implications for conflicting data and knowledge modification, but does not establish whether the timestamp has a behavioral consequence for memory plasticity.

---

## Labmate-style one-step rotation

The seed asks:

> **Can the model's internal state reveal when information was learned?**

The proposed rotation asks:

> **Does when information was learned predict how that information behaves under later contradictory learning?**

In the Yano-style language:

```text
representation contains property X
        ↓
does X correspond to a real behavioral distinction?
```

No new representation is invented. We take an already-demonstrated training-order variable and ask whether it matters for one natural quantity: **overwriteability**.

---

## Why the answer would matter

A model that has learned two facts equally well can still, in principle, store them in states with different plasticity.

Two pictures are possible.

### Picture A — training age is readable but behaviorally inert

```text
early learned fact  ─┐
                     ├─ same update budget → same overwrite dynamics
recently learned fact┘
```

Then the `Fresh in Memory` direction is primarily a historical residue or metadata-like trace. Interesting for interpretability, but not a determinant of future learning.

### Picture B — training age predicts memory stability

```text
same current accuracy / confidence
but
different acquisition time
        ↓
different susceptibility to contradictory update
```

Then learning history matters even after current task performance is matched. That would be a fundamental continual-learning fact: **present competence does not fully specify future plasticity**.

This can inform how conflicting information, corrections and updates interact with previously learned knowledge without proposing a new editing algorithm.

---

## Collision audit

### 1. Fresh in Memory itself

Very close seed, but representation-focused. It shows that acquisition time is encoded and can be read; it does not test matched-fact overwrite resistance as a function of acquisition age.

### 2. Catastrophic forgetting / post-training forgetting

There is a large literature on catastrophic forgetting, including recent 2026 item-level studies that map which pretrained capabilities are lost during post-training.

These works establish that different items have different forgetting susceptibility. But the current Round-03 audit did not find a study where items have **known, experimentally controlled acquisition ages** and are then matched for current strength before measuring contradictory-update susceptibility.

### 3. Knowledge editing

Knowledge editing directly studies changing facts, but usually asks:

```text
Can method M edit fact f accurately / locally / robustly?
```

This proposal does not compare editing methods. A fixed ordinary update procedure is merely the measurement instrument for asking:

```text
Does acquisition age predict how much update is required?
```

### 4. Human memory consolidation analogy

Older cognitive-science literatures ask whether recent and remote memories differ in susceptibility to interference and reconsolidation. That provides a useful conceptual comparison but must not be imported as an assumed mechanism.

### Collision boundary

Kill this candidate if a prior LLM study is found that already controls factual acquisition order, matches present fact strength, and measures later overwrite / interference resistance as a function of that order.

Do not preserve novelty by narrowing to one model family, one editing method or one entity type.

---

## Main conceptual attacks

### Attack 1 — old facts simply received more indirect reinforcement

In natural pretraining, “older” facts often have more opportunities for later re-exposure. That makes age inseparable from exposure count.

**Defense:** G-0 should begin in the seed paper's controlled sequential-training setup, where each stage can have matched data volume and update budget.

Natural pretraining is a later extension, not the first identification experiment.

### Attack 2 — early facts may already be weaker because of forgetting

If D1 accuracy is lower than D6 accuracy, D1 may be easier to overwrite simply because it is weaker now.

**Defense:** only compare facts / subsets with matched baseline correctness, answer margin / log probability, and loss as closely as practical. Report results conditioned on current strength.

### Attack 3 — “overwrite” can mean many things

A new value can temporarily suppress the old answer without deleting the previous representation.

**Defense:** keep the first claim behavioral and operational:

> update budget required to make the contradictory answer dominate under a fixed test protocol.

Do not claim physical deletion of memory.

### Attack 4 — training stage may proxy arbitrary dataset identity

The seed already counterbalances datasets across independent runs, but our overwrite result must do the same.

**Defense:** rotate which entity group occupies each stage across seeds / curricula and analyze stage rather than dataset identity.

### Attack 5 — more recent facts may simply be closer in parameter space to the final model

That may actually be the explanation, but it does not invalidate the phenomenon. The first experiment asks whether a stable behavioral age effect exists; explanation comes later.

---

## Cheapest decisive G-0

### Experimental object

Reuse the **Fresh in Memory** controlled sequential entity-learning setup as closely as possible. The public codebase underlying the work makes this attractive.

Train one small open model sequentially on `D1 … D6`, with disjoint entity groups and matched training budgets.

### Step 1 — establish current knowledge

At the final checkpoint, retain facts that are correctly answered and construct matched subsets across stages by:

- correctness;
- answer log-probability / margin;
- loss;
- optionally paraphrase robustness.

If matched subsets cannot be obtained because early-stage facts have degraded too much, adjust the base training protocol before proceeding rather than statistically hiding the mismatch.

### Step 2 — identical contradictory update

For each selected fact or small balanced batch, branch from the **same final model** and train on an equally formatted contradiction:

```text
old: entity X → attribute A
new: entity X → attribute B
```

Use exactly the same number / format of examples and optimizer settings.

### Primary observable

For fact `f`:

```text
flip_steps(f) = minimum update steps until B reliably dominates A
```

or an equivalent continuous area-under-update-curve metric.

Then ask whether:

```text
training stage / recency projection
        ↔
overwrite resistance
```

after conditioning on current fact strength.

### Strong positive

Across counterbalanced curricula, acquisition stage predicts the contradictory-update curve even among facts with similar final correctness/confidence.

The direction itself is empirical:

- older facts may be more stable;
- or recent facts may be more stable.

Do **not** preregister a biological “older = consolidated” direction as necessary for survival. The scientific question is whether age produces a reproducible plasticity gradient at all.

### Kill line

Kill / downgrade if:

- overwrite dynamics are indistinguishable across stages after matching current strength;
- the effect reverses arbitrarily with dataset identity or curriculum seed;
- stage effects vanish once baseline log-probability / loss is controlled;
- only a specialized knowledge-editing algorithm reveals the effect.

The candidate should live or die on ordinary training dynamics.

---

## Optional second-stage causal test — only after G-0 survives

The seed paper shows that **re-exposure moves an old entity group along the recency direction**.

This creates a tempting intervention:

```text
old fact
→ re-expose
→ recency representation becomes “newer”
→ does overwriteability change?
```

But re-exposure also changes exposure count / strength, so this is not a clean first experiment. It should only be attempted after the basic behavioral age effect is established, with controls for extra training.

---

## Why this is different from Candidate A

Candidate A, **When Does a Fact Become Recallable?**, asks about the developmental transition:

```text
encoding → usable recall
```

during acquisition.

This candidate asks what happens **after knowledge is already learned**:

```text
acquisition age / training history → future plasticity under conflict
```

They use related learning-dynamics tools but answer distinct questions. If both survive G-0, they may later inform a broader account of knowledge lifecycle; they should not be fused before either phenomenon is established.

---

## Interestingness test

Assume the cleanest positive result:

> **Two facts can be equally retrievable now, yet the one learned earlier requires systematically more—or systematically less—training to overwrite solely because of its acquisition history.**

That would make training history a genuine state variable for future learning, not merely a probe-readable timestamp.

A clean null is equally clarifying: it would show that the striking ICLR 2026 temporal direction can remain behaviorally inert with respect to later knowledge modification.

---

## Why this fits the advisor-style search

It has a particularly direct seed-paper structure:

```text
strong recent paper proves X is encoded
        ↓
one natural question:
does X affect behavior?
        ↓
one matched update experiment
```

It is small enough for a rapid pilot, does not rely on frontier APIs, and should remain meaningful as models change.

---

## Current verdict

`KEEP — PROVISIONAL TOP-5 CANDIDATE`

Among the new Round-03 leads this is the more laboratory-style one-step question. The main danger is overlap with the very broad forgetting / knowledge-editing literature, so the project must retain its exact identifying object: **known acquisition age, matched present strength, fixed contradictory update**.