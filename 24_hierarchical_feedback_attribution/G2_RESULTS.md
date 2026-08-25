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

**`left`'s effect on binary task success shrinks monotonically to exactly zero
as force increases:** `+0.300 → +0.200 → 0.000`. The `150N`/`left` point
estimate is not merely non-significant — it is a literal `[0.000, 0.000]`
bootstrap CI, meaning the per-config `LR - RR` difference is `0` on every
single one of the 30 configs, the same signature G1 found for the upper-body
channel on `left`/`100N`.

What this establishes is narrower than it might sound: **live vs. replayed
navigation has exactly zero causal effect on binary task success across
these 30 configs at 150N/left.** It does not establish that the navigation
channel "stops working" or is inert at this force — live navigation may still
be issuing a different `navigate_cmd`, moving the base along a different
trajectory, or changing contact timing; none of that was measured here, and
none of it needs to be zero for the *outcome* effect to be zero. This
distinction matters because this candidate exists partly to catch exactly the
opposite mistake — treating unchanged behavior as proof a mechanism did
nothing. That is why P0b was run as a separate gate before Topic 24 was even
registered: P0's identical replay fidelity under no disturbance did not by
itself establish the WBC seam was live, and had to be checked directly at the
command level (see
[`../embodied_topic_search/prototypes/feedback_source_attribution/P0B_RESULTS.md`](../embodied_topic_search/prototypes/feedback_source_attribution/P0B_RESULTS.md)).
The same logic applies here: a zero *outcome* effect is not evidence of a zero
*mechanism* effect unless the mechanism itself was checked.

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

## Structured force dependence beyond the binary reversal verdict

The full grid does not establish a force-general reversal. It reveals
structured heterogeneity:

```text
left:  +0.300 → +0.200 → 0.000
right: -0.067 → -0.200 → -0.067
```

The monotone `left`-side pattern is descriptive rather than a pre-registered
trend claim — no trend test was frozen in advance, and three points is not
enough to statistically rule out a fluke shape. Only `100N` independently
establishes the bidirectional reversal; `50N` and `150N` are consistent with
it (`left` still positive at `50N`; `right` still negative-signed at both) but
neither clears the frozen bar on its own. The `right`-side pattern in
particular is not monotone at all — `100N` is a local extreme within the
three points measured, not an endpoint of a trend — so no directional
claim about `right` beyond `100N` is being made here.

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
