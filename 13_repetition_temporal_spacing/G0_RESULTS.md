# G0 Results — Topic 13 Temporal Spacing

## Final verdict

**`NO_EVIDENCE_SPACING_IN_LOCKED_TEST`**

The preregistered four-trial confirmation reproduced the repetition-damage prerequisite in every trial, but the clean clustered-vs-even spacing manipulation did not show a stable direction across independent training/repeated-pool replications.

This is the meaningful negative result defined by the Topic 13 protocol.

## Locked confirmation results

| Trial | fresh | random | random − fresh | clustered | even | clustered − even |
|---|---:|---:|---:|---:|---:|---:|
| 20260822 | 7.382136 | 7.398458 | +0.016322 | 7.414454 | 7.415987 | −0.001534 |
| 20260823 | 7.402416 | 7.422812 | +0.020395 | 7.445307 | 7.434549 | +0.010758 |
| 20260824 | 7.403583 | 7.422048 | +0.018465 | 7.431175 | 7.430169 | +0.001005 |
| 20260825 | 7.393898 | 7.407173 | +0.013275 | 7.410883 | 7.420017 | −0.009134 |

## Gate interpretation

### Prerequisite: repetition damage

`random - fresh` is positive in all four locked trials:

```text
+0.016322
+0.020395
+0.018465
+0.013275
```

So the selected regime robustly instantiated the motivating repetition-damage phenomenon. The spacing result is therefore interpretable; this is not a setup-reproduction failure.

### Primary spacing contrast

`clustered - even` changes sign across trials:

```text
-0.001534
+0.010758
+0.001005
-0.009134
```

The frozen protocol required a consistent direction across valid trials for a spacing claim. That condition fails.

No post-hoc schedule, model, repeat pool, threshold, or alternate spacing metric is authorized to rescue the hypothesis.

## Scientific interpretation

Under this exact fixed-multiset, cross-optimizer-update design, temporal spacing of identical repeated documents is **not supported as a stable causal driver** of the observed repetition damage.

This does not prove that no spacing effect can ever exist in language-model training. It does kill the registered Topic 13 formulation as a paper candidate: the clean locked manipulation failed to produce the robust direction the story required while the prerequisite repetition damage was present.

## Reusable lesson

> **When the prerequisite phenomenon is stable but the proposed explanatory axis reverses direction across locked replications, stop. A real seed phenomenon does not justify tuning the explanation until it works.**
