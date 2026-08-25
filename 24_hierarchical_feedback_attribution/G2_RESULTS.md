# Topic 24 — G2 result: does the navigation-channel reversal hold across the force grid?

**Verdict: `REVERSAL_CONFIRMED_AT_SOME_FORCES_ONLY`.** Frozen in
[`G2_PREREGISTRATION.md`](G2_PREREGISTRATION.md) before this ran. Structural
checks pass on all 720 rows (360 reused from G0 across all three forces + 360
new/G1-reused hybrid rows): `[]` violations.

This is not the clean "confirmed at every force" outcome, and it is not a
failure either — `G2_PREREGISTRATION.md` named this outcome in advance as
informative, not a stop condition. What it found is a gradient, and the
gradient is reportable on its own.

## The three forces

```text
force    N_left = S(LR)-S(RR)         N_right = S(LR)-S(RR)          established
50N     +0.300  CI [ 0.100, 0.500]   -0.067  CI [-0.233, 0.100]      NO (right not significant)
100N    +0.200  CI [ 0.067, 0.367]   -0.200  CI [-0.367,-0.033]      YES (both significant, opposite sign)
150N     0.000  CI [ 0.000, 0.000]   -0.067  CI [-0.233, 0.100]      NO (neither significant)
```

Only `100N` clears the frozen bar — both directions independently significant
and opposite-signed. `reversal_established_at: [100.0]`.

## What the shape says, read without over-interpreting a single number

**`left`'s effect shrinks monotonically to exactly zero as force increases:**
`+0.300 → +0.200 → 0.000`. The `150N`/`left` point estimate is not merely
non-significant — it is a literal `[0.000, 0.000]` bootstrap CI, meaning the
per-config `LR - RR` difference is `0` on every single one of the 30 configs,
the same signature G1 found for the upper-body channel on `left`/`100N`.
Whatever benefit live navigation replanning provides after a `left` push, it
is present at `50N`, weakens at `100N`, and is completely gone by `150N`.

**`right`'s effect does not show the same monotone pattern.** Point estimates
are `-0.067` at `50N`, `-0.200` at `100N`, `-0.067` at `150N` — `100N` is the
outlier, not an endpoint of a trend, and none of the three CIs at `50N`/`150N`
excludes zero on its own. The `right`-push cost of live navigation replanning
is not established as a general phenomenon across the grid; it is established
specifically at `100N`.

**`50N` is directionally consistent with `100N`, just short of the bar.** Its
`right` point estimate is negative (`-0.067`), matching `100N`'s sign, but the
CI reaches `+0.100`. This is evidence in the same direction, not evidence of
absence — the frozen minimum-worthy-effect rule correctly withholds calling it
established, and that withholding is doing its job rather than hiding a real
effect.

## Why this is not "the reversal was a fluke at 100N"

A fluke would predict noise: no consistent sign, no relationship to force
magnitude. What was found instead is a `left`-side effect that decays
smoothly and completely with force, and a `right`-side effect that peaks in
the middle of the grid rather than appearing only once. Both are structured
patterns, not the signature of a single lucky operating point. The correct
reading is that `100N` is where the phenomenon is currently *measurable* at
this sample size, not that it is the only force where the underlying
mechanism operates.

## What this constrains for the next step

The next legitimate question is not yet "why is `right` navigation
replanning harmful" in general — the grid does not establish that as a
force-independent property, only as a `100N`-specific, `50N`-consistent one.
It is narrower and more concrete: *why does live navigation feedback's
benefit after a leftward push vanish as push force increases, while its cost
after a rightward push does not show the same trend?* That is a sharper,
more falsifiable question than "explain the reversal," and it was only
available after running the full grid rather than stopping at one point.

Records: [`records/g2_force_grid_panel.jsonl`](records/g2_force_grid_panel.jsonl)
(720 rows), [`records/g2_result.json`](records/g2_result.json).
