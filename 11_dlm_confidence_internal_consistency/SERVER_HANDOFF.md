# Topic 11 — archived server handoff

**Do not run G-1 or rerun G-0 to search for a positive result. Topic 11 is archived.**

The frozen v3 G-0 completed with valid prerequisites and the final verdict:

```text
KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL
```

Key result:

```text
Arithmetic prerequisite gap = 0.426361
Semantic-alias gap          = 0.214693
Eligible mirrored pairs     = 140

Primary confidence_result_middle:
Delta_consistency = -0.000003
95% CI            = [-0.000055, 0.000025]
locked floor      = 0.010
```

The protocol worked; the project-level retroactive/global-consistency hypothesis did not.

Read:

- `G0_RESULT.md` for the full frozen table;
- `ARCHIVE_SUMMARY.md` for the scientific conclusion and lessons;
- `AUDIT.md` for the pre-run identification logic.

The large `confidence_full` consistency effect is not a rescue: that metric includes the manipulated suffix. The frozen primary deliberately scored unchanged Step-2/Step-3 result tokens before the future consistency intervention, and that effect was essentially zero.

Only engineering archaeology or reproduction should use the preserved code. Do not change metrics, thresholds, pooling, token regions, model, prompt, or dataset to preserve the same scientific claim.

A future agent should reopen Topic 11 only if a genuinely new external result motivates a **new separately registered question** with a distinct identification strategy.
