# Topic 24 — G0 result (CloseDoor, physical push, 2026-08-24/25)

**Verdict: `WBC_LEVEL_DOMINATES`.** Of the recovery `fresh` shows over
`actuator_replay`, essentially all of it is attributable to the WBC seam;
the VLA-level contribution is not statistically distinguishable from zero.

## Frozen prerequisites — all passed

```text
structural_violations         []
fidelity_control (force=0)    fresh = vla_replay = actuator_replay = 0.933   pass
push_effective (150N)         median base displacement 0.102 m >= 0.02 m     pass
matched configs per cell      29 or 30 of 30, every cell >= 24                pass
```

The replay instruments hold inside the actual G0 code path, not just in the
isolated P0/P0b prototypes: on the unperturbed control column, `fresh`,
`vla_replay` and `actuator_replay` all land on exactly the same success rate
(28/30), with zero gap in either direction.

### One deliberate deviation from a literal 30/30, reported rather than hidden

Data collection was stopped 2 rows short of the full 630-row grid
(628/630) after a very long overnight run whose remaining rows were entirely
inside a single already-99.7%-complete worker. This is not a silent
shortcut: the frozen gate is `MIN_MATCHED_CONFIGS = 24` per cell precisely so
that a small amount of incomplete data cannot invalidate the panel. The only
cell affected is `150N/right`, which has 29 of 30 matched configs — still far
above the floor, and the verdict does not turn on that one config.

## The pooled numbers

```text
                    success rate
fresh                0.397
vla_replay           0.380
actuator_replay      0.078

delta_high (VLA-level online feedback)        0.017   95% CI [-0.051, 0.084]
delta_low  (WBC/reference-generation feedback) 0.302   95% CI [ 0.246, 0.358]
```

`delta_high`'s confidence interval straddles zero and its point estimate is
far below the pre-registered minimum worthy effect (0.10): the VLA's own
online replanning contribution is not established. `delta_low`'s interval is
entirely positive and its point estimate clears the same bar by 3x: the
WBC/reference-generation seam is carrying the recovery.

Read against the P0b finding recorded before this G0 ran — that the arms and
hands below the VLA seam are open-loop interpolation, so `delta_low` can only
ever carry locomotion/balance state feedback — this sharpens rather than
weakens the result: whatever is saving the task here is leg/base-level
balance correction, not arm-level corrective reaching, and the VLA is not
shown to be doing that reaching either.

`actuator_replay`'s pooled residual is 0.078: almost nothing survives on
servo/actuator dynamics/mechanics/task-slack alone. Nearly every surviving
success needed at least one of the two feedback layers to engage.

## Per-cell grid — every force/direction reported, none selected after the fact

| force | direction | configs | fresh | vla_replay | actuator_replay | delta_high | delta_low |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | (control) | 30 | 0.933 | 0.933 | 0.933 | 0.000 | 0.000 |
| 50 | left | 30 | 0.833 | 0.533 | 0.000 | **+0.300** | +0.533 |
| 50 | right | 30 | 0.700 | 0.800 | 0.267 | **-0.100** | +0.533 |
| 100 | left | 30 | 0.267 | 0.033 | 0.000 | **+0.233** | +0.033 |
| 100 | right | 30 | 0.400 | 0.600 | 0.200 | **-0.200** | +0.400 |
| 150 | left | 30 | 0.067 | 0.000 | 0.000 | **+0.067** | +0.000 |
| 150 | right | 29 | 0.103 | 0.310 | 0.000 | **-0.207** | +0.310 |

### The reversal that must be reported, not smoothed over

`delta_high` is negative on 3 of 6 perturbed cells — every `right`-push cell.
`vla_replay` *outperforms* `fresh` there: sticking to the pre-push recorded
plan and letting only the WBC react to the disturbed state does better than
letting the live VLA re-observe and re-plan. The pooled `delta_high` (0.017)
is close to zero because the `left`-push cells go the other way and the two
roughly cancel; that cancellation is exactly why the bootstrap CI straddles
zero, and it is a real cross-cell pattern, not just noise averaging out.
`delta_low` is positive and large on every single perturbed cell without
exception, left or right.

No thumb was on this scale: the six cells above are the entire frozen force
grid, reported in the order the design specifies, before any of them was
read.

## Bootstrap

Clustered over the 30 physical configs (each config's whole force panel
resampled as a unit), 10,000 resamples, seed `20260824`:

```text
delta_high   point 0.017   95% CI [-0.051, 0.084]
delta_low    point 0.302   95% CI [ 0.246, 0.358]
```

Records: [`records/g0_closedoor.jsonl`](records/g0_closedoor.jsonl) (628 rows),
[`records/g0_result.json`](records/g0_result.json).

## Method opening this licenses

Given `WBC_LEVEL_DOMINATES`: the immediate lever is compute allocation. If a
released VLA's replanning is not shown to help under a lateral push at these
magnitudes, paying for a full high-level re-plan on every disturbance is not
obviously justified — a policy that learns when the WBC alone can absorb the
error and only wakes the VLA otherwise is a concrete, testable follow-on. The
`right`-push reversal additionally opens a sharper question: under what
disturbance geometry does live VLA replanning actively hurt relative to
holding the pre-disturbance plan, and is that a property of this checkpoint's
training distribution or of decoupled-WBC architectures generally.

## Incident record: two false-positive automation failures during collection

Two revisions of the background completion-watcher script used to orchestrate
this run independently produced false positives that killed live policy
servers mid-collection, crashing 8 of 11 in-flight workers each time. No data
was lost — every worker's output file is append-only and `--resume` skips
already-recorded rows — but both incidents are recorded because the failure
mode is the same one this candidate exists to guard against, turned against
our own tooling: an automated judgment that "the process is done" was trusted
without being proven, and it had kill authority.

1. An SSH-reachability check that treated a failed remote query as "zero
   processes" rather than "unknown," combined with a `pkill` cleanup that
   kills every matching process on a host rather than only the ones actually
   finished.
2. A revised version with a "two consecutive clean zero-readings" guard still
   produced a single false trigger through a mechanism not fully root-caused
   under time pressure.

The fix was not a third attempt at a smarter liveness check: the final
watcher has no kill authority anywhere in the script, and triggers only on a
purely local signal (total row count across output files, no SSH involved).
This mirrors [`../FAILURES_AND_LESSONS.md`](../FAILURES_AND_LESSONS.md) lesson
16 (prove the instrument before trusting it) applied to infrastructure: an
automated action with destructive authority needs a completion signal that
cannot be wrong in the direction that causes damage, not merely one that is
usually right.
