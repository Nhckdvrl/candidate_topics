# 13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?

**Status: `ARCHIVED / NO_EVIDENCE_SPACING_IN_LOCKED_TEST`**

## Scientific question

> If two language-model training runs see exactly the same repeated documents the same number of times, can held-out generalization change only because identical copies reappear at different temporal distances across optimizer updates?

Topic 13 was designed as a causal follow-up to the robust phenomenon that internal data repetition can damage language-model pretraining.

The final re-audited G0 removed the original within-update duplicate confound. Within each trial, `random`, `clustered`, and `even` used:

- the same repeated document IDs;
- the same multiplicity for every repeated document;
- the same non-repeated document at every non-repeat slot;
- the same repeat-slot positions;
- the same total tokens and optimizer steps;
- at most one repeat slot per optimizer step;
- the same initialization across conditions.

The confirmation used four locked trials (`20260822`–`20260825`) with different repeated pools and balanced GPU assignment.

## Final result

The repetition-damage prerequisite reproduced in **4/4** trials:

```text
random - fresh
+0.016322
+0.020395
+0.018465
+0.013275
```

The primary spacing contrast was inconsistent:

```text
clustered - even
-0.001534
+0.010758
+0.001005
-0.009134
```

The sign changes across independent locked replications. Under the preregistered decision logic, the verdict is:

```text
NO_EVIDENCE_SPACING_IN_LOCKED_TEST
```

This is a scientific negative for the registered spacing explanation, **not** a setup/reproduction failure, because the motivating repetition damage remained robust.

No alternate schedule, model, repeated pool, threshold, or post-hoc spacing metric is authorized to rescue the topic.

## Archive files

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — final decision and transferable lesson.
- [`G0_RESULTS.md`](./G0_RESULTS.md) — exact four-trial table and gate interpretation.
- `configs/g0.json`, `schedule.py`, `train.py`, `analyze.py`, `run_g0.py` — frozen implementation retained for provenance.
- `tests/` — structural tests for the matched schedule and analysis.

## Reusable lesson

> **A robust motivating phenomenon does not make its proposed explanation true. When the explanatory contrast reverses direction across locked replications, stop rather than search for a schedule that produces the preferred sign.**
