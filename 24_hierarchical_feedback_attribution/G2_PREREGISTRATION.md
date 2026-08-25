# Topic 24 — G2 pre-registration: does the navigation-channel reversal hold across the force grid?

> Written and committed **before any G2 result was read**.

## What G1 left open

G1 factored the VLA command at a single operating point (100N, both
directions) and found the reversal G0 first noticed lives specifically in the
navigation/base channel:

```text
N_100,left  = LR - RR = +0.200   95% CI [ 0.067, 0.367]
N_100,right = LR - RR = -0.200   95% CI [-0.367,-0.033]
R_100 = N_100,left - N_100,right = 0.400
```

That is one force out of the three G0 already froze and ran (`50N`, `100N`,
`150N`). Whether this is a property of 100N specifically, or holds across the
force grid, is unanswered. Chasing *why* the navigation channel is
directionally miscalibrated before answering that would mean building an
explanation from a single operating point — the same selection risk this
candidate has avoided so far.

## What's new here

Only `50N` and `150N` need new `LR`/`RL` collection. `100N` is not re-run —
G1's own new rows are reused unchanged, alongside G0's `RR`/`LL` at all three
forces. Total new data:

```text
30 configs x 2 directions x 2 forces x 2 conditions = 240 new rollouts
```

Runner, hook and structural proof-of-fire checks are unchanged from G1
(`g1_channel_factorization_runner.py --force-n {50,150}`); only the force
argument differs.

## A corrected verdict predicate, used going forward

G1's evaluator (`g1_core.py`) called a reversal whenever *any* navigation
effect cleared the minimum-worthy bar and no upper-body effect did — it never
checked that the `left` and `right` navigation effects were actually opposite
in sign. That was audited and recorded in `G1_RESULTS.md` after the fact; it
did not change G1's conclusion (which rests on the raw `N_100,left`/
`N_100,right` numbers, independent of the verdict enum), but the predicate
itself was looser than intended.

G2 uses `g2_core.py`, built directly on the single clean contrast the G1 audit
identified:

```text
N_f,d = S(LR) - S(RR)          upper-body held at replay throughout
R_f   = N_f,left - N_f,right
```

A reversal is established at force `f` only if `N_f,left` and `N_f,right` are
each **independently significant** (bootstrap CI excludes zero, `|point| >=
0.10`) **and opposite in sign**. Verified against the existing G1 100N data
before any new rollout was collected: reproduces `R_100 = 0.400`,
`reversal_established = True`.

## Frozen stop rules

```text
PREREQUISITE_FAIL_STRUCTURAL       any row not at {50,100,150}N; a channel
                                   claimed replayed/live not matching its
                                   overwrite count; RR contacted the server;
                                   a hybrid row never queried the live VLA
INSUFFICIENT_MATCHED_CONFIGS       any force/direction has <24 of 30 complete
                                   RR/LR pairs
```

## Frozen verdicts

```text
REVERSAL_CONFIRMED_ACROSS_FORCE_GRID       established at all 3 forces
REVERSAL_CONFIRMED_AT_SOME_FORCES_ONLY     established at 1 or 2, not all 3
REVERSAL_NOT_ESTABLISHED_OUTSIDE_100N      established at 0 forces besides 100N
```

None of these is treated as a failure. `REVERSAL_CONFIRMED_AT_SOME_FORCES_ONLY`
or `..._NOT_ESTABLISHED_OUTSIDE_100N` are results, not stops: they would say
the phenomenon is real but force-magnitude-dependent, which is itself
informative and reportable, not grounds to re-run at a fourth force in search
of confirmation.

## What is explicitly deferred

Not run in G2, on purpose: any investigation of *why* the navigation channel
is directionally miscalibrated (inspecting raw `navigate_cmd` values, camera
framing, training-data coverage). That question only becomes appropriately
timed after the force grid answers whether the phenomenon generalizes past a
single operating point. Cross-task validation (a second SIMPLE task with
strong upstream Ψ0 competence, matching P0/P0b done fresh) is the step after
that, not before it.

## Files

- `g1_channel_factorization_runner.py` — unchanged, run with `--force-n 50`
  and `--force-n 150`.
- `g2_merge_records.py` — folds G0's `RR`/`LL` across all three forces with
  G1's existing 100N rows and the new 50N/150N rows into one panel.
- `g2_core.py` — the corrected frozen decision procedure.
- `tests/test_g2_core.py` — pure-logic tests (`11 passed`).
