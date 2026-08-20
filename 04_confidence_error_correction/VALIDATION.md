# Validation contract — Topic 04 G-1v2

**Freeze:** G-1v2 is the single allowed measurement repair after the preregistered G-1v1 failure. No G0 correction outcome has been observed.

---

# 1. Scientific question

Among initially wrong items with matched accessibility of the correct target:

> **Does stronger commitment to one specific wrong hypothesis change the speed or durability of corrective learning?**

The direction is not assumed.

---

# 2. Why G-1v1 failed

G-1v1 scored 9981 exact-K=10 MMLU-Pro items and produced only 61 matched pairs, below the hard `<200` stop threshold.

The failure happened **before G0**.

Post-run audit identified two design defects:

1. `top_wrong_stability >= 0.8` was not treatment-neutral. A diffuse wrong distribution should have unstable top-1 identity under small perturbations, so this gate selected against the low-commitment construct.
2. arithmetic averaging of mapped probability vectors did not isolate semantic preference from option-position susceptibility.

The v1 `results/g0/STOPPED.md` remains valid: G0 was correctly not run.

---

# 3. G-1v2 primary measurement

## 3.1 Stimulus and model

Primary:

```text
dataset   MMLU-Pro test, exactly K=10
model     Qwen/Qwen2.5-1.5B-Instruct
```

One predeclared model-scale replication is allowed:

```text
Qwen/Qwen2.5-3B-Instruct
```

No further model search is allowed to rescue a failed v2.

## 3.2 Balanced permutation family

Primary family A:

```text
cyclic
```

Every semantic choice occupies every label position once.

Reliability family B on a deterministic 20% subset:

```text
hashed_cyclic
```

For each item, deterministically hash-shuffle the base semantic order, then take all K cyclic shifts. It is another complete balanced family but not the same set of ordered option lists.

## 3.3 Semantic distribution

For mapped semantic probability vector `p_r` from permutation `r`:

\[
s_j=\frac1R\sum_r\log(p_{r,j}+\epsilon)
\]

and

\[
p^{debias}_j = softmax(s)_j.
\]

This `p^{debias}` is the **only primary G-1v2 semantic distribution**.

The old arithmetic mean is retained only as a v1 diagnostic.

## 3.4 Why log-space aggregation is justified

Assume an additive position nuisance in label logits:

\[
z_{r,j} = \alpha_j + \beta_{\operatorname{position}(r,j)}.
\]

Because the balanced set sends every semantic choice through every position exactly once:

\[
\frac1R\sum_r \beta_{\operatorname{position}(r,j)}
\]

is constant across `j`. The normalization term is also common across `j`, so averaging log probabilities and renormalizing recovers the semantic term under this model.

A deterministic unit test in `tests/test_g1v2_math.py` verifies this exact cancellation synthetically.

This is a prespecified nuisance-removal argument, not a metric chosen for a favorable correction result.

---

# 4. G-1v2 variables

Correct-target accessibility:

\[
a=p^{debias}_{correct}.
\]

Wrong distribution:

\[
q_j=\frac{p^{debias}_j}{1-a},\quad j\ne correct.
\]

Primary wrong commitment:

\[
c_{\max}=\max_j q_j.
\]

Robustness-only commitment:

\[
c_H=1-\frac{H(q)}{\log(K-1)}.
\]

Position susceptibility:

\[
S_{pos}=\frac1R\sum_r JS(p_r\|p^{debias}).
\]

`S_pos` is a separate nuisance/phenotype. Do not redefine low commitment as high positional instability.

The following are diagnostics only:

- `top_wrong_stability`;
- modal top-wrong identity;
- arithmetic-mean v1 metrics;
- wrong top-1/top-2 margin;
- answer entropy;
- target rank.

---

# 5. Response-channel diagnostics

For single-token answer labels, save for every permutation:

\[
M_{label}=\sum_{\ell\in A..J}p_{vocab}(\ell)
\]

and whether the unconstrained greedy next token is one of A..J.

Aggregate per item:

```text
mean_label_mass
min_label_mass
greedy_is_allowed_label_rate
```

Interpretation:

- conditional A-J probabilities are the experimental choice distribution;
- low label mass means this distribution is highly conditional on an artificial constraint;
- if the model broadly refuses the requested answer channel, do not treat sharp A-J ratios as clean belief evidence.

Before G0, manually inspect the audit subset if:

```text
median mean_label_mass < 0.50
or
median greedy_is_allowed_label_rate < 0.80
```

These are diagnostic warning thresholds, not post-hoc exclusion rules. If the requested response format is broadly not obeyed, stop and judge the measurement invalid rather than filtering favorable items.

---

# 6. Existing-data zero-GPU gate

The v1 `base_scores.jsonl` already contains all mapped `permutation_probs`.

First run:

```bash
python code/reaggregate_g1v2.py \
  --input results/g1/base_scores.jsonl \
  --output results/g1v2/base_scores_reaggregated.jsonl
```

Then pair using no top-wrong stability filter:

```bash
python code/build_matched_pairs.py \
  --input results/g1v2/base_scores_reaggregated.jsonl \
  --pairs-output results/g1v2/matched_pairs.jsonl \
  --eligible-output results/g1v2/eligible_wrong.jsonl \
  --report-output results/g1v2/matching_report.json \
  --require-k 10 \
  --p-caliper 0.02 \
  --question-length-ratio 1.35 \
  --answer-length-ratio 1.50 \
  --high-quantile 0.70 \
  --low-quantile 0.30 \
  --discovery-fraction 0.70 \
  --seed 20260821
```

Do not use `--susceptibility-caliper` in the primary v2 screen. We do not know a justified threshold yet; reliability is assessed independently.

## 6.1 Pair common-support gate

Strong:

```text
>= 600 pairs
mean |Δ p_correct| <= .010
median |Δ p_correct| <= .010
mean c_max separation >= .10
```

Minimal:

```text
>= 300 pairs
mean |Δ p_correct| <= .015
mean c_max separation >= .08
```

Hard failure:

```text
< 200 pairs
```

If `<200`, **KILL Topic 04**. Do not change quantiles, K, dataset, caliper, or commitment definition.

If `200–299`, run reliability audit but do not proceed to G0 unless there is a preregistered reason to accept a reduced-power pilot. Default decision remains stop.

---

# 7. Independent reliability audits

Only if the zero-GPU common-support screen is not a hard failure.

Select a deterministic 20% item subset by hash of item ID.

Run three measurements on exactly the same subset:

```text
A: primary prompt   + cyclic
B: primary prompt   + hashed_cyclic
C: alternate prompt + cyclic
```

For A vs B and A vs C require:

```text
Spearman c_max        >= .70
Spearman p_correct    >= .90
median semantic JS    <= .05
```

Exact semantic top-wrong identity agreement is diagnostic only.

If either audit fails, **KILL Topic 04**.

Do not replace the reliability gate with a new one after seeing the values.

---

# 8. 3B replication

`Qwen/Qwen2.5-3B-Instruct` was allowed before G0 outcomes existed.

Run the exact same G-1v2 protocol if:

- 1.5B v2 passes; or
- 1.5B is borderline but not a hard failure and there is enough compute to determine whether poor semantic commitment is scale-specific.

The 3B result is reported regardless of direction.

If both 1.5B and 3B fail measurement reliability/common support, **KILL Topic 04**.

Do not escalate to 7B/14B.

---

# 9. G0 is unchanged in scientific logic

Only after G-1v2 passes.

Use the v2 matched pairs and the same `p^{debias}` scorer for cycle 0 through cycle 10.

One semantic item receives exactly one corrective exposure per cycle. High and low groups train together.

Primary outcome remains per-item correction gain AUC:

\[
G_i=\frac1{10}\sum_{e=1}^{10}(p_{i,e}(y^*)-p_{i,0}(y^*)).
\]

Primary contrast:

\[
\Delta_G=G_{high}-G_{low}.
\]

Discovery/confirmation remain 70/30 and start independently from the same base model.

Discovery continuation criterion:

```text
>= 2/3 seeds same direction
pooled pair-bootstrap 95% CI excludes 0
|mean Δ_G| >= .02
```

Freeze before confirmation.

A failed confirmation kills the directional claim. No hidden-state or model rescue.

---

# 10. Predeclared interpretable G0 outcomes

Only after a valid v2 measurement:

1. high commitment corrects faster → hypercorrection-like plasticity;
2. high commitment corrects slower → entrenchment;
3. early advantage but late reversal / relapse → uptake differs from durable replacement;
4. equivalence after accessibility control → accessibility dominates commitment;
5. correct-target growth and original-error suppression dissociate → learning the correction and suppressing the misconception may be distinct processes.

These outcomes must not be used to justify a bad G-1 measurement.

---

# 11. No-rescue rule

If G-1v2 fails, Topic 04 is archived.

Do not rescue by:

- lowering the old 8/10 stability gate;
- using 7B/14B because 1.5B/3B fail;
- changing from K=10 to K=4;
- changing to free-response confidence;
- selecting a new confidence metric from the same data;
- adding hidden-state probes;
- loosening accessibility matching;
- choosing only domains with more pairs.

A future free-response misconception study would be a **new topic**, not Topic 04 continuation.
