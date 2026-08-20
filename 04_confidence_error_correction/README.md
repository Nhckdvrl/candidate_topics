# Topic 04 — Confidence and Error Correction

**Status: ARCHIVED / KILLED AT MEASUREMENT / IDENTIFICATION GATE**

> **If two learners are equally far from the correct answer, does being strongly committed to one wrong answer make corrective learning easier or harder?**

The question itself remains scientifically meaningful, but this candidate is **closed**.

The project never reached corrective SFT. It failed twice at the prerequisite measurement / identification gate:

```text
G-1v1                    FAIL — 61 matched pairs
G-1v2 locked repair      FAIL — 130 matched pairs
G0 corrective training   NOT RUN
G1 durability            NOT RUN
```

The locked G-1v2 rule was:

```text
<200 matched pairs -> archive Topic 04
```

That rule was triggered, so no further rescue was attempted.

For the full scientific history, results, failure analysis, reusable assets, and lessons:

**[ARCHIVE_SUMMARY.md](./ARCHIVE_SUMMARY.md)**

---

## What the project was trying to identify

For a K-way multiple-choice item with correct answer `y*`, the project separated:

### Correct-target accessibility

\[
a(x)=p(y^*\mid x)
\]

from concentration on a specific wrong hypothesis:

\[
q_j(x)=\frac{p(y_j\mid x)}{1-p(y^*\mid x)},\qquad y_j\neq y^*
\]

\[
c_{\max}(x)=\max_j q_j(x).
\]

The intended comparison was:

```text
same / tightly matched p(correct)
+ clearly different wrong commitment
-> identical corrective SFT
-> compare correction curves
```

This was meant to distinguish whether a strongly held misconception changes learning dynamics independently of how accessible the correct answer already is.

---

## G-1v1

Primary system:

```text
model    Qwen/Qwen2.5-1.5B-Instruct
data     MMLU-Pro test, exactly K=10
items    9,981
```

v1 used arithmetic averaging over 10 balanced option rotations and required the same semantic top-wrong in >=8/10 rotations.

Result:

```text
eligible stable wrong             716
low pool                           215
high pool                          215
matched pairs                       61
mean |Δ p(correct)|             0.00547
mean commitment separation      0.2529
```

The surviving pairs were well matched, but the measurement itself had two structural defects:

1. the top-wrong stability gate mechanically removed genuinely low-commitment items;
2. arithmetic averaging mixed semantic uncertainty with option-position susceptibility.

Therefore v1 was classified as a **measurement failure**, not a hypothesis result.

---

## The one allowed repair: G-1v2

Before any correction outcome existed, the project allowed one mathematically motivated repair.

Balanced permutations were aggregated in log-probability space:

\[
s_j=\frac1R\sum_r\log(p_{r,j}+\epsilon)
\]

\[
p_j^{debias}=softmax(s)_j.
\]

This normalized geometric-mean construction removes an additive position main effect under the balanced-permutation model.

At the same time:

- `top_wrong_stability` became diagnostic only;
- position susceptibility was separated using distribution-level JS divergence;
- response-channel diagnostics were added;
- the same repaired scorer was prepared for any eventual G0 evaluation.

The repair and its justification are preserved in **[MEASUREMENT_REPAIR.md](./MEASUREMENT_REPAIR.md)**.

---

## G-1v2 final result

The offline repair was run on all saved permutation distributions.

```text
scored items                                9,981
eligible initially-wrong                    6,668
low commitment cutoff                      0.23763
high commitment cutoff                     0.72070
low pool                                    2,001
high pool                                   2,001
matched pairs                                  130
mean |Δ p(correct)|                        0.00744
median |Δ p(correct)|                      0.00569
mean commitment separation                 0.64894
median position-susceptibility JS           0.2903
```

The repair clearly restored the low-commitment end of the construct: eligible wrong items increased from 716 to 6,668 and the low cutoff dropped from about 0.795 to about 0.238.

However, even with large high/low pools and strong commitment separation, only **130** pairs survived the locked clean-comparison constraints.

This is below the predeclared `<200` hard stop.

Therefore the correct interpretation is:

> **The natural stimulus pool did not provide enough common support to separate correct-target accessibility from wrong-hypothesis commitment at the scale required for a credible paired correction experiment.**

It is **not** evidence that commitment has no effect on correction.

---

## What was deliberately not run

After G-1v2 failed, the project stopped before:

- second balanced-family reliability audit;
- alternate-prompt reliability audit;
- 3B replication;
- G0 corrective SFT;
- G1 relapse / durability.

No attempt was made to rescue the topic by:

- loosening the `p(correct)` caliper;
- removing comparability constraints;
- changing to 7B / 14B;
- switching datasets;
- using free-response confidence;
- changing the commitment metric;
- adding hidden-state probes.

Those would constitute a new identification strategy, not a bug fix.

---

## Repository record

```text
04_confidence_error_correction/
├── ARCHIVE_SUMMARY.md
├── README.md
├── VALIDATION.md
├── SERVER_RUNBOOK.md
├── MEASUREMENT_REPAIR.md
├── results/
│   ├── g1/      # v1 results
│   └── g1v2/    # locked repair + STOPPED.md
├── tests/
└── code/
```

`VALIDATION.md` remains as the frozen historical protocol. The code and results are retained for reuse, but **Topic 04 itself should not be revived without a genuinely new external observation and a newly registered identification strategy.**
