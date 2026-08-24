# Topic 13 Archive Summary — Temporal Spacing of Repeated Pretraining Data

## Final decision

**ARCHIVED / `NO_EVIDENCE_SPACING_IN_LOCKED_TEST`**

Topic 13 asked whether duplicate-data damage depends not only on the repeated-data multiset but also on the temporal distance between identical exposures across optimizer updates.

The final G0 deliberately removed the earlier within-update duplicate confound: all compared schedules used the same repeated documents, multiplicities, non-repeat slots, total tokens, optimizer steps, model initialization within trial, and at most one repeat slot per optimizer step. The four confirmation trials also varied the repeated-document pool and rotated GPU assignment.

## What happened

The motivating repetition-damage prerequisite was robust:

```text
random - fresh
20260822  +0.016322
20260823  +0.020395
20260824  +0.018465
20260825  +0.013275
```

But the primary spacing contrast was not directionally stable:

```text
clustered - even
20260822  -0.001534
20260823  +0.010758
20260824  +0.001005
20260825  -0.009134
```

The registered protocol explicitly defined this pattern as `NO_EVIDENCE_SPACING_IN_LOCKED_TEST`.

## Why this is terminal

This is not a failed reproduction: repetition damage was present in all four trials. The registered explanatory axis—cross-update spacing of identical repeated documents—failed to show a stable causal direction under the clean matched design.

Continuing by changing schedule geometry, model family, repeated pool, effect metric, or thresholds would be post-hoc search for a positive result.

The broad scientific possibility that training temporal organization matters is not disproved. The **specific Topic 13 paper story is stopped**.

## Reusable lesson

> **Do not confuse a robust seed phenomenon with evidence for the proposed explanation. If the seed survives and the explanatory manipulation changes sign across locked replications, archive the explanation instead of tuning it.**

See [`G0_RESULTS.md`](./G0_RESULTS.md) for the exact confirmation table.
