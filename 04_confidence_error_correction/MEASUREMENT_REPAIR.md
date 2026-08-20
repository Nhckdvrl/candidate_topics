# Topic 04 measurement repair record

## Final status

**CLOSED — G-1v2 FAILED THE LOCKED HARD STOP. TOPIC ARCHIVED.**

The project never reached corrective SFT. This document records why one repair was allowed after G-1v1 and why no further repair is allowed after G-1v2.

---

## Decision after G-1v1

**Verdict at the time: measurement failure; one repair allowed.**

G-1v1 correctly stopped before G0 because only 61 matched pairs survived the preregistered `<200` stop rule.

The result did **not** test the correction hypothesis. No corrective SFT was run.

### Observed v1 numbers

```text
n_scored_input                    9981
n_eligible_wrong                   716
wrong_concentration low cutoff    0.7947
wrong_concentration high cutoff   0.9358
n_low_pool                         215
n_high_pool                        215
n_pairs                             61
mean |Δ p_correct|                0.00547
mean commitment separation        0.2529
```

The fact that the v1 "low" cutoff was already ~0.795, despite nine wrong options, showed that the stability gate had removed most genuinely diffuse distributions.

---

## Why v1 was a measurement defect rather than a hypothesis result

### Defect 1 — treatment-dependent stability gate

G-1v1 required the same semantic top-wrong option in >=8/10 rotations.

But low wrong commitment means multiple wrong options are close. Their top-1 identity is therefore expected to swap under small option-order perturbations.

The gate selected on a consequence of the treatment itself.

### Defect 2 — arithmetic mean confounded semantic uncertainty and position susceptibility

Some raw items were highly confident on almost every permutation but changed *which semantic option* received the confidence when option positions moved.

Arithmetic averaging turned:

```text
sharp but position-sensitive
```

into:

```text
apparently semantically diffuse
```

These are different phenomena.

Because both defects were identified before any G0 correction outcome existed, one measurement repair was allowed without changing the scientific hypothesis.

---

## G-1v2 repair

The locked repair did six things:

1. removed top-wrong stability as an inclusion gate;
2. aggregated balanced permutations in log-probability space;
3. retained position susceptibility as a separate JS-divergence diagnostic;
4. prepared an independent balanced-permutation-family and alternate-prompt reliability audit;
5. added full-vocabulary answer-label mass / response-channel diagnostics;
6. used the same repaired scorer for any eventual G0 checkpoint evaluation.

### Why mean log-probability

For an additive nuisance model

\[
z_{r,j} = \alpha_j+\beta_{position(r,j)}
\]

and a complete balanced permutation set, every semantic option sees every position exactly once.

Then averaging log probabilities across rotations leaves semantic score `alpha_j` plus constants shared across all choices. Renormalizing removes those constants.

A deterministic unit test verifies exact recovery in the synthetic additive-bias case.

---

## Locked hard stop before running v2

The repair was explicitly defined as the **only** allowed repair.

The project would be archived if:

- `<200` matched pairs remained after offline v2 reaggregation;
- balanced-family reliability failed;
- prompt reliability failed;
- response-channel diagnostics showed the choice probabilities were not a meaningful answer channel.

If v2 failed, there would be no additional model/metric/dataset rescue.

---

## G-1v2 result

The saved 9,981 × 10 permutation distributions were reaggregated offline with the repaired log-space measurement.

```text
measurement_version                         g1v2_logmean
n_scored_input                              9981
n_eligible_wrong                            6668
wrong_concentration low cutoff             0.237628
wrong_concentration high cutoff            0.720695
n_low_pool                                  2001
n_high_pool                                 2001
n_pairs                                      130
mean |Δ p_correct|                         0.007440
median |Δ p_correct|                       0.005687
mean commitment separation                 0.648937
median commitment separation               0.645623
eligible median top-wrong stability        0.30      # diagnostic only
eligible median position JS                0.290292
```

The repair clearly fixed the main v1 construct problem:

```text
eligible wrong: 716 -> 6668
low cutoff:     ~0.795 -> ~0.238
```

So v1 had indeed been over-filtering the low-commitment end.

However, after restoring the full commitment range, only 130 high/low pairs had enough common support under the frozen clean-comparison constraints.

This is below the locked threshold:

```text
130 < 200
```

Therefore G-1v2 returned **FAIL / ARCHIVE**.

---

## Final interpretation

This is not a negative result about corrective learning.

No corrective training was run, so the project cannot conclude whether concentrated wrong commitment:

- accelerates correction;
- slows correction;
- produces an early/late reversal;
- or has no effect.

The actual negative result is an identification result:

> **Even after one mathematically justified repair, the natural MMLU-Pro stimulus pool did not provide enough clean high/low commitment comparisons at matched correct-target accessibility to justify the planned correction experiment.**

The surviving pairs were tightly matched and strongly separated in commitment, which argues against a trivial matcher bug. The bottleneck is insufficient common support under the intended scientific comparison.

---

## Why no second rescue is allowed

Possible ways to create more pairs include:

- loosening `p(correct)` matching;
- dropping category/length comparability;
- switching from matched pairs to extrapolative regression;
- moving to larger models;
- switching datasets or response formats;
- defining a new confidence/commitment metric.

Each of these changes the identification strategy after the failure is known.

They would therefore constitute a new exploratory project, not a repair of Topic 04.

The topic is archived here.

See **[ARCHIVE_SUMMARY.md](./ARCHIVE_SUMMARY.md)** for the complete scientific summary, lessons, and reusable assets.
