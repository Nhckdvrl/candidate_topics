# G1 — Is the Readable Ranking State Causally Used?

**Status: FROZEN BEFORE FRESH-SEED TEST**

## Why G1 exists

G0 passed on the exact Qwen3-8B / seed-0 / five-shot setting:

- full probe accuracy: `0.996875`
- full generation accuracy: `0.817500`
- hard probe accuracy: `0.961240`
- hard generation accuracy: `0.682171`
- hard `probe-correct / generation-wrong`: `38 / 129`

Thus the experimental object is real: immediately before generation, a simple
linear classifier can recover the correct ranking on almost every error.

But G0 still establishes **availability, not causal use**.

G1 asks one narrower question:

> **If we minimally change the readable ranking coordinate while leaving the
> rest of the residual state fixed, does the model's generated choice change?**

A positive result says that the ranking coordinate participates in the native
output computation. A clean null says that the linearly readable ranking state
is not itself the causal readout used by generation at the tested stage.

---

## Important post-G0 observation and independence rule

Inspection of the locked seed-0 G0 errors revealed an unexpected pattern: the
hard critical cases appear to overwhelmingly choose the scientific-notation
operand when generation is wrong. This observation was made **after seeing the
G0 test set**. It is therefore exploratory and cannot be confirmed on seed 0.

G1 must use a fresh generated test set for all new claims.

```text
fresh confirmation seed = 20260824
setting                 = int_sci_compare
model                   = exact G0 Qwen3-8B revision
prompt                   = exact G0 balanced 5-shot prompt
```

Use the upstream generator unchanged. Preserve raw data. For inferential counts,
exact displayed duplicate `(a,b)` pairs are counted once (first occurrence),
and exact numerical ties are excluded rather than resampled. Report both raw
and unique counts.

Do not inspect fresh-test results and then alter the intervention.

---

# G1-P0 — Fresh-object replication

Before any intervention, run the frozen G0 behavior/probe on the fresh seed
using the **seed-0-trained probe weights**. Do not refit the probe on fresh test.

The project proceeds to intervention only if the fresh unique hard subset has:

1. at least `100` hard examples;
2. frozen-probe hard accuracy `>= 0.90`;
3. at least `25` unique `probe-correct / generation-wrong` examples;
4. critical rate `>= 0.20`;
5. invalid generation `< 5%`.

Failure is `STOP_G1_NONREPLICATION`. Do not search a new seed/model/prompt.

The fresh set also provides a confirmatory descriptive test of the exploratory
notation pattern. Among hard generation errors that exactly equal one of the two
input operands, report the proportion that select the scientific-notation
operand. A notation-mechanism follow-up is allowed only if this rate is `>=0.80`.
This notation threshold does **not** affect the main rank-causality test.

---

# G1-P1 — Freeze the causal layer

G0 selected the final layer for maximum decoding accuracy, but intervening only
at the final block output leaves little downstream computation in which the
changed state can act. G1 therefore uses a predeclared saturation rule based
only on the already-frozen seed-0 validation curve:

> **Choose the earliest layer whose seed-0 validation ranking-probe accuracy is
> at least 0.99.**

From G0 this is:

```text
zero-based transformer block = 19
one-based layer              = 20
seed-0 validation probe acc  = 0.990625
```

Call this `L_sat`.

This rule was chosen before fresh-seed evaluation. No layer sweep is allowed in
G1. Train/freeze the logistic ranking probe `(w_rank, b_rank)` at `L_sat` on the
original seed-0 training set with the same G0 recipe.

The intervention site is the **last prompt token** at the output of transformer
block 19. During generation, modify only the prefill pass; do not reapply the
intervention to every generated token.

---

# G1-P2 — Minimal rank reflection

For a hidden state `h` at `L_sat`, define the frozen ranking probe logit

```text
m(h) = w_rank^T h + b_rank
```

The minimal-L2 perturbation that reflects the state across the probe's decision
hyperplane is

```text
h_flip = h - 2 * m(h) / ||w_rank||^2 * w_rank
```

This changes the sign of the readable ranking variable while preserving every
component orthogonal to `w_rank`.

There is no steering coefficient and no strength search.

## Primary population

Use fresh-seed **unique hard examples that are originally both probe-correct and
generation-correct**. These cases let us test whether changing the readable
ranking state can causally disrupt a decision that the model normally gets
right.

## Primary behavioral outcome

After `h -> h_flip`, greedily generate with the same decoding contract and ask:

```text
did an originally correct answer become the opposite input operand?
```

Report separately:

- any correct -> wrong flip;
- exact flip to the opposite operand;
- invalid / neither-operand output.

The exact-opposite-operand flip is the main interpretable effect.

---

# G1-P3 — Norm-matched random null

For each example, let

```text
delta_i = ||h_flip - h||
```

Use `8` fixed Gaussian random directions with seeds

```text
20260831 ... 20260838
```

At `L_sat`, orthogonalize each random direction against `w_rank`, normalize it,
and apply an equal-norm perturbation `delta_i` at the same token.

Thus the null:

- changes the residual by exactly the same per-example L2 norm;
- acts at the same layer and token;
- leaves the linear ranking coordinate unchanged by construction.

Do not select a convenient random seed. Pool/report all eight.

---

# G1 primary gate

Define

```text
F_rank = exact-opposite-operand flip rate under rank reflection
F_null = mean exact-opposite-operand flip rate over the 8 random controls
DeltaF = F_rank - F_null
```

Bootstrap over unique examples, keeping the 8 null interventions nested within
each example.

### CAUSAL USE

Declare `RANK_DIRECTION_CAUSAL` only if all hold:

1. the intervention flips the frozen probe sign on `>= 0.99` of evaluated cases;
2. `DeltaF >= 0.20`;
3. paired bootstrap 95% CI lower bound for `DeltaF` is `> 0`;
4. among rank-reflection outputs that change, at least `80%` are one of the two
   original operands rather than invalid/novel numeric strings.

### STRONG NULL

Declare `READABLE_BUT_NOT_CAUSALLY_USED_AT_LSAT` if:

```text
DeltaF <= 0.05
and 95% CI upper bound <= 0.10
```

This is a scientifically meaningful result. Do **not** rescue it by trying
other layers, tokens, coefficients, nonlinear probes, SAEs, or a different
model.

### Otherwise

`INCONCLUSIVE_DO_NOT_TUNE`

---

# G1-P4 — Conditional notation-competition follow-up

This section is allowed only if the fresh-seed replication confirms that at
least `80%` of exact-operand hard generation errors select the scientific
operand.

The question then becomes:

> **Does a competing notation-side representation causally dominate the
> already-correct ranking signal?**

At the same frozen `L_sat`, train on seed-0 train a linear classifier predicting
whether the scientific-notation operand is in position A or B. The notation
assignment is randomized by the upstream generator and is independent of the
ranking label.

Let its weight be `w_not`. Remove its component parallel to `w_rank`:

```text
w_not_orth = w_not - proj_w_rank(w_not)
```

Normalize this direction and fit a one-dimensional train/validation threshold
for scientific-side classification. No fresh-test fitting is allowed.

On fresh hard critical cases, **neutralize** the notation coordinate to that
frozen threshold while exactly preserving the ranking projection. Compare with
same-norm random directions orthogonal to `w_rank`.

Primary outcomes:

- wrong -> correct rescue rate;
- reduction in scientific-operand choice rate;
- invalid output rate.

This is a conditional mechanism test, not a rescue route for a failed rank
causality test. If the notation intervention is null, do not expand to a large
subspace search in the same project stage.

---

# Novelty boundary

The EACL 2026 seed already establishes representation/behavior dissociation and
probe-aware finetuning, but explicitly states that it did not intervene on the
representation at inference time to test how output changes.

A nearby July 2026 ICML Mechanistic Interpretability Workshop paper,
**Geometry of Ordinal Representations in Language Models**, studies geometry and
activation patching for ordinal variables including numeric magnitude. Its
question is representation geometry and task-relevant manifold information; it
does not study the mixed-notation same-instance `probe-correct / generation-wrong`
dissociation or whether the correct ranking state controls the generated choice.

Therefore G1 must keep its claim at the causal-use/readout boundary, not claim
that activation patching of numerical representations is itself new.

---

# Required artifacts

Save under `20_numeracy_representation_access/artifacts/g1/`:

```text
fresh_data_audit.json
fresh_baseline_records.jsonl
fresh_baseline_summary.json
rank_probe_lsat.npz (or equivalent small weights/metadata)
rank_reflection_records.jsonl
random_null_records.jsonl
rank_causal_summary.json
notation_summary.json             # only if conditional P4 is run
```

Write `G1_RESULTS.md` with:

- exact model revision/environment;
- fresh seed and checksums;
- raw and deduplicated counts;
- frozen `L_sat` verification;
- fresh-object replication metrics;
- notation-choice confirmation statistic;
- `F_rank`, all eight null flip rates, `F_null`, `DeltaF`, bootstrap CI;
- invalid/novel-output rate;
- frozen verdict;
- conditional notation result if eligible.

Do not commit model weights, hidden-state dumps, or large caches.
