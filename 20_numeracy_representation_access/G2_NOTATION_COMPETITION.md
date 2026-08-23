# G2 — Does Scientific-Notation Form Override an Already-Correct Magnitude Decision?

**Status: FROZEN AFTER DISCOVERY + INDEPENDENT CONFIRMATION, BEFORE G2 TEST**

## Why G2 exists

Topic 20 G0 established a same-prompt dissociation on Qwen3-8B: the numerical ranking is usually linearly decodable even when generation chooses the wrong operand.

The original rank-causality G1 did **not** run. Its fresh-object P0 stopped because the seed-0-trained layer-20 probe obtained `124/138 = 0.898551` on the fresh hard subset, narrowly below the preregistered `0.90` threshold. That historical result remains `STOP_G1_NONREPLICATION`; G2 does not alter that verdict.

However, G0 inspection revealed a different, sharper phenomenon:

> when Qwen3-8B is wrong on hard mixed-notation comparisons, it appears to select the operand written in scientific notation regardless of which operand is actually larger.

This was exploratory on seed 0 and therefore could not count as confirmation.

The frozen G1 protocol preregistered a fresh descriptive confirmation threshold **before** seed `20260824` was evaluated. On that independent seed:

```text
raw test                         = 1600
unique displayed pairs           = 1598
unique hard                       = 138
hard generation errors            = 60
exact-operand hard errors          = 60
errors choosing scientific operand = 55 / 60 = 0.916667
```

The preregistered notation threshold was `>= 0.80`, so the phenomenon independently confirmed.

Thus G2 is not a rescue of the failed G1 rank gate. It is a new mechanism question supported by a discovery/confirmation sequence:

```text
seed 0       : exploratory discovery
seed 20260824: independent confirmation
seed 20260825: locked causal test
```

The seed paper studies mixed-notation comparison and answer-position bias, but does not report this scientific-operand error attractor. Its 5-shot prompt deliberately alternates correct answer position to reduce position bias, which is the same prompt used here.

---

## Natural question

Scientific notation and ordinary notation are mathematically equivalent encodings of magnitude. A competent comparison procedure should therefore treat notation as irrelevant once magnitude is recovered.

Yet Topic 20 now has evidence for the opposite pattern:

> **the model often has a correct readable ranking while its output is pulled toward whichever operand is written in scientific notation.**

G2 asks:

> **Is a notation-side representation causally competing with the already-available ranking signal at the decision stage?**

A positive result would identify a concrete access/readout failure: the model has the relevant magnitude relation, but a task-irrelevant surface-form feature wins the output competition.

---

# G2-P0 — Locked fresh causal dataset

Use a new untouched generated test set:

```text
fresh causal seed = 20260825
setting           = int_sci_compare
model             = exact Qwen3-8B G0/G1 snapshot
prompt            = exact balanced official 5-shot prompt
```

Generate with the unchanged upstream `construct_data.py`. Preserve the raw test and checksum before evaluation.

For inferential counts:

- exclude exact numerical ties;
- count exact displayed `(a,b)` duplicates once, first occurrence;
- do not regenerate until a convenient seed appears.

No seed20260825 result may be used to redesign the intervention.

## G2 object gate

The fresh seed must contain at least:

```text
N_unique_hard >= 100
N_hard_exact_operand_generation_errors >= 30
```

and among those exact-operand hard errors:

```text
scientific_operand_choice_rate >= 0.80
```

This is the phenomenon-level prerequisite for a notation-attractor causal test.

The frozen seed-0 ranking probe at `L_sat` is reported descriptively, but **G2 does not require a new arbitrary 0.90 point threshold**. The causal population itself is restricted to examples whose frozen ranking probe is correct, so every evaluated rescue case explicitly has the representation required by the scientific claim.

If the notation attractor fails the fresh seed20260825 object gate, stop:

```text
STOP_G2_NOTATION_NONREPLICATION
```

Do not search another seed/model/prompt.

---

# G2-P1 — Frozen layer and two linear coordinates

Use the same already-fixed causal site as G1:

```text
L_sat                    = layer 20
zero-based block          = 19
intervention token        = final prompt token
intervention pass         = prefill only
```

No layer or token sweep.

## Ranking coordinate

Fit the same seed-0 logistic ranking probe on the original seed-0 train split:

```text
y_rank = 1[a > b]
```

Freeze `(w_rank, b_rank)`.

## Notation-side coordinate

On the **same seed-0 train split**, fit a logistic classifier:

```text
y_not = 1[scientific-notation operand is A]
```

The upstream generator randomly chooses which operand is converted to scientific notation, so notation side is not the ranking label.

Let its weight be `w_not`.

Remove the component parallel to the ranking direction:

```text
w_not_orth = w_not - proj_w_rank(w_not)
```

Normalize:

```text
u_not = w_not_orth / ||w_not_orth||
```

This gives a notation direction that, by construction, does not change the linear ranking coordinate.

Before test, freeze the scalar decision threshold `tau_not` for `z = u_not^T h` using seed-0 train/validation only. No seed20260825 fitting.

Required representation checks on seed-0 validation:

```text
notation-side accuracy >= 0.95
abs(cosine(u_not, w_rank)) <= 1e-5
```

If these fail, stop before causal test because the intended independent notation coordinate was not identified cleanly.

---

# G2-P2 — Primary population

Use only seed20260825 unique hard cases satisfying all:

1. frozen ranking probe is correct;
2. baseline generation is wrong;
3. baseline output exactly equals one of the two input operands;
4. baseline output is the scientific-notation operand;
5. therefore the correct answer is the ordinary-notation operand.

These are the clean cases implied by the confirmed phenomenon:

```text
correct ranking is readable
BUT
output follows scientific notation
```

Require at least:

```text
N_primary >= 25
```

Otherwise:

```text
STOP_G2_INSUFFICIENT_CAUSAL_SUPPORT
```

---

# G2-P3 — Notation neutralization intervention

For each primary hidden state `h` at `L_sat`, compute:

```text
z_not = u_not^T h
```

Move only the notation coordinate to its frozen neutral decision threshold:

```text
h_neutral = h - (z_not - tau_not) * u_not
```

Because `u_not` is orthogonal to `w_rank`, this intervention preserves the linear ranking projection by construction.

There is **no strength coefficient** and no search.

Apply the intervention only to the final prompt token during the prefill pass. Do not reapply on generated tokens.

Primary behavioral outcome:

```text
wrong scientific operand -> correct ordinary operand
```

Also report:

- any change from baseline;
- scientific-operand choice after intervention;
- invalid / neither-operand output;
- frozen ranking-probe logit before vs after intervention.

The ranking logit must remain numerically unchanged up to implementation tolerance.

---

# G2-P4 — Norm-matched orthogonal random null

Use `8` fixed Gaussian random directions:

```text
20260901 ... 20260908
```

Orthogonalize every random direction against **both**:

```text
w_rank
u_not
```

normalize it, and for each example apply the same L2 perturbation norm as its notation-neutralization intervention.

Thus the random null:

- acts at the same layer/token/prefill pass;
- has exactly the same per-example perturbation magnitude;
- preserves the ranking coordinate;
- does not move directly along the identified notation coordinate.

Do not select a convenient null seed.

---

# G2 primary gate

Define:

```text
R_not  = wrong->correct rescue rate under notation neutralization
R_null = mean wrong->correct rescue rate across 8 random controls
DeltaR = R_not - R_null
```

Bootstrap over unique primary examples with random-control seeds nested within each example.

## NOTATION COMPETITION CAUSAL

Declare:

```text
NOTATION_COMPETITION_CAUSAL
```

only if all hold:

1. notation coordinate is successfully moved to `tau_not` on `>= 0.99` of examples;
2. ranking logit preservation succeeds on `>= 0.99` of examples;
3. `DeltaR >= 0.20`;
4. paired bootstrap 95% CI lower bound for `DeltaR` is `> 0`;
5. invalid/neither-operand rate under notation neutralization is `< 0.10`;
6. among changed valid outputs, at least `80%` move to the correct ordinary operand.

## STRONG NULL

Declare:

```text
NOTATION_READABLE_BUT_NOT_CAUSAL_AT_LSAT
```

if:

```text
DeltaR <= 0.05
and bootstrap 95% CI upper bound <= 0.10
```

provided manipulation checks pass.

## Otherwise

```text
INCONCLUSIVE_DO_NOT_TUNE
```

Do not rescue with another layer, token, coefficient, multi-dimensional subspace, model, or prompt in this project stage.

---

# Why this is not circular

The notation direction is learned from **which input side is rendered in scientific notation**, not from correctness or model errors.

The causal outcome is a different variable: whether removing that direction rescues a wrong numerical comparison while preserving the ranking projection.

Nothing about fitting `w_not` forces generation to prefer the scientific operand or forces neutralization to correct an error.

---

# Interpretation

A positive G2 would support the narrow mechanism:

> Qwen3-8B can contain the correct mixed-notation ranking at the decision state, yet a linearly separable notation-side signal causally biases the generated operand; neutralizing that signal rescues errors without erasing the ranking coordinate.

This would convert the broad `representation vs access` dissociation into a concrete competition mechanism.

A null would show only that the simple one-dimensional notation coordinate at `L_sat` is not the causal culprit; it would not erase the confirmed behavioral notation attractor.

---

# Novelty boundary

The EACL 2026 seed paper already establishes mixed-notation difficulty, high internal numerical decodability, answer-position bias under one-shot prompting, and improvement from probe-aware finetuning.

It does not report the independently confirmed fact used here: under the balanced 5-shot hard regime, Qwen3-8B's exact-operand errors overwhelmingly select the operand written in scientific notation.

Nearby work on ordinal representation geometry and numeric activation patching means that `activation intervention on numbers` is not itself novel. The claim must remain the specific **surface-notation-vs-correct-ranking competition** mechanism.

---

# Required artifacts

Save under `20_numeracy_representation_access/artifacts/g2/`:

```text
fresh_seed20260825_test.jsonl
fresh_seed20260825_test.sha256
fresh_data_audit.json
fresh_baseline_records.jsonl
fresh_baseline_summary.json
rank_probe_lsat.npz
notation_probe_lsat.npz
notation_representation_checks.json
notation_neutralization_records.jsonl
random_null_records.jsonl
notation_causal_summary.json
```

Write `G2_RESULTS.md` with exact environment, checksums, representation checks, object gate, population size, `R_not`, all eight null rates, `R_null`, `DeltaR`, bootstrap CI, manipulation checks, invalid rate, and frozen verdict.
