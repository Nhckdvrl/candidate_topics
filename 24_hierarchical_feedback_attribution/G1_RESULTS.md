# Topic 24 — G1 result: which VLA command channel causes the reversal?

**Verdict: `NAVIGATION_CHANNEL_CAUSES_REVERSAL`.** Frozen in
[`G1_PREREGISTRATION.md`](G1_PREREGISTRATION.md) before this ran. Structural
checks pass on all 240 rows (120 reused from G0 + 120 new): `[]` violations.
This verdict string was produced by the frozen evaluator; a post-result audit
found its predicate looser than the natural-language definition of "reversal"
— see [Post-result audit](#post-result-audit-evaluator-predicate). It does not
change the empirical conclusion below, which rests on a single controlled
contrast independent of the evaluator's verdict logic.

## The cleanest single contrast

Hold the upper-body channel fixed at replay throughout; the only thing that
changes is whether the navigation/base channel is replayed or live:

```text
N_100,d = S(LR) - S(RR)     upper-body held constant, navigation is the only treatment

N_100,left  = 0.233 - 0.033 = +0.200   95% CI [ 0.067, 0.367]
N_100,right = 0.400 - 0.600 = -0.200   95% CI [-0.367,-0.033]

R_100 = N_100,left - N_100,right = 0.400
```

Same checkpoint, same task, same 100N magnitude — only the push direction
differs, and the causal value of live navigation replanning goes from +20pp to
-20pp. Both CIs independently exclude zero. This is the headline result; the
verdict enum below organizes the fuller 2x2 but the science stands on this one
line.

## The core numbers

```text
              RR      LR      RL      LL
left        0.033   0.233   0.033   0.267
right       0.600   0.400   0.600   0.400
```

On `right`, the pooled `RR` and `RL` rates are **equal** (0.600 = 0.600) and so
are `LR` and `LL` (0.400 = 0.400). Holding the upper-body channel fixed,
switching navigation between replayed and live moves the pooled success rate
the full 0.2 every time, in both directions. The reverse — holding navigation
fixed and switching upper-body — is addressed carefully below: equal *pooled*
rates are not the same claim as *zero effect on every config*, and the two
directions turn out to differ in exactly that respect.

```text
                                   left                          right
nav_effect_upper_replayed   +0.200  CI [ 0.067, 0.367]    -0.200  CI [-0.367,-0.033]
nav_effect_upper_live       +0.233  CI [ 0.100, 0.400]    -0.200  CI [-0.400, 0.000]
upper_effect_nav_replayed    0.000  CI [ 0.000, 0.000]     0.000  CI [-0.100, 0.100]
upper_effect_nav_live       +0.033  CI [ 0.000, 0.100]     0.000  CI [-0.100, 0.100]
```

`nav_effect_upper_replayed` (i.e. `N_100,d` above) independently clears the
minimum worthy effect with a CI excluding zero in *both* directions on its
own — this is the contrast the headline rests on. `nav_effect_upper_live`
replicates the same direction and magnitude (`+0.233` left, `-0.200` right)
but its `right` CI reaches to `0.000` exactly rather than excluding it, so by
the frozen pre-registered rule it does not independently establish the
effect — it is a directionally consistent replication, not a second
independent confirmation. Every upper-body effect is at or under the 0.10
floor.

Read against G0's own per-cell table: `RR` and `LL` here are G0's `vla_replay`
and `fresh` at 100N, so this is not a new phenomenon — it is the same reversal
G0 found, now traced to a specific half of the command.

## What "the reversal" actually is

The sign flip is not a property of live VLA replanning in general. It is the
navigation/base channel specifically, and it flips sign with the push
direction: live navigation feedback *helps* under a `left` push
(`nav_effect_upper_replayed = +0.200`) and *hurts* under a `right` push
(`nav_effect_upper_replayed = -0.200`) — same magnitude, opposite sign, same
task, same checkpoint, only the push direction differs. Whatever the VLA's
navigation channel does after a lateral shove, it is directionally
appropriate for a leftward correction and directionally wrong for a rightward
one.

The correct claim is narrower than "the upper-body channel does nothing," and
the data itself draws the line precisely. On `left`, the per-config `RL - RR`
difference is exactly `0` on all 30 configs — not just an average that comes
out to zero, but zero row-by-row, which is why the bootstrap CI is exactly
`[0.000, 0.000]`. On `right`, the same per-config comparison is `+1` config,
`-1` config, `28` unchanged — real, opposite-signed movement that happens to
cancel, which is exactly why the CI is `[-0.100, +0.100]` rather than
`[0, 0]`. A CI of `[0,0]` and a CI of `[-0.1,+0.1]` are not the same claim,
and averaging them into "upper-body has no effect" would repeat the exact
mistake this candidate exists to catch: G0's pooled `delta_high` looked like
"VLA replanning doesn't matter" until per-cell inspection showed it was two
real opposite-signed effects cancelling. The honest statement here is:

> No average upper-body contribution above the pre-registered minimum worthy
> effect (0.10) was established at 100N, in either direction. On `left` this
> reflects a genuine absence of effect on every one of the 30 configs; on
> `right` it is consistent with a small amount of real, oppositely-signed
> per-config heterogeneity averaging toward zero, not with the channel being
> inert.

Recall from P0b that the arms and hands below the VLA seam are already known
to be open-loop with respect to the WBC. This result is compatible with that
— no *average* VLA-level upper-body effect was established either — but it
does not repeat P0b's stronger claim (which was a structural, per-tick proof)
at the level of outcomes.

## What this sharpens about the G0 reading

G0 already showed the WBC/reference-generation seam dominates pooled
recovery, and that `delta_high` pooled near zero because of a left/right
cancellation rather than being uniformly small. G1 shows that cancellation is
not diffuse across the whole VLA command — it is concentrated in the
navigation/base channel specifically, and that channel's own effect is *not*
small in either direction (`N_100,left = +0.20`, `N_100,right = -0.20`, both
CI-excluding-zero on the single `LR - RR` contrast). The correct headline is
not "VLA replanning barely matters"; it is:

> **Live navigation replanning has direction-dependent causal value: it
> improves recovery after leftward disturbances and degrades recovery after
> rightward disturbances.** No comparably-sized average upper-body
> contribution was established at this operating point.

## Structural checks

Every one of the 120 new rows shows the intervention fired exactly as
claimed: a channel marked replayed has an overwrite count equal to its step
count; a channel marked live has an overwrite count of zero; every hybrid row
queried the live policy server; every `RR` row (reused from G0) has
`server_queries == 0`. `push_tick` matches the G0 tape for every config the
disturbance is applied to the same event G0 measured the reversal under.

Records: [`records/g1_channel_panel.jsonl`](records/g1_channel_panel.jsonl)
(240 rows), [`records/g1_result.json`](records/g1_result.json).

## Post-result audit: evaluator predicate

The frozen evaluator (`g1_core.py`) returned `NAVIGATION_CHANNEL_CAUSES_REVERSAL`.
A post-result read of the code found its predicate looser than the natural-
language definition of "reversal":

```python
nav_real = any(real(k, d) for k in (nav_effect_upper_replayed, nav_effect_upper_live)
                for d in (left, right))
upper_real = any(real(k, d) for k in (upper_effect_nav_replayed, upper_effect_nav_live)
                  for d in (left, right))
if nav_real and not upper_real:
    verdict = NAVIGATION_CHANNEL_CAUSES_REVERSAL
```

`nav_real` only asks whether *any* of the four navigation-channel effect
estimates clears the bar — it never checks that the `left` and `right`
effects are opposite in sign. A dataset where `left = +0.20` and `right =
0.00` would trigger the identical verdict string. The evaluator was not
literally testing for a reversal; it was testing "does the navigation channel
have an established average effect, and does the upper-body channel not."

**This does not drive the empirical conclusion above.** The reversal itself is
established directly and independently of this verdict logic, by the single
controlled contrast at the top of this document: `N_100,left = +0.200` and
`N_100,right = -0.200`, each with a bootstrap CI that excludes zero on its
own. The verdict enum organizes the fuller 2x2 into a name; the science does
not depend on that name's predicate being exactly right, and here it happens
to have been looser than intended without changing which conclusion the data
support.

This is being left as a recorded audit rather than a silent patch to
`g1_core.py`, consistent with [`FAILURES_AND_LESSONS.md`](../FAILURES_AND_LESSONS.md)
lesson 16: an instrument's correctness is proven, not assumed, and a fix
applied after seeing the result must be visible as a fix, not folded
invisibly into what looks like the original pre-registration. `g2_core.py`
(for the force-grid completion) uses a corrected predicate that explicitly
requires opposite-signed, independently-established effects across
directions before calling a reversal.

## What this opens next

Not "why does navigation replanning go wrong under a `right` push." That
question is real, but chasing it now — inspecting `navigate_cmd` values,
camera framing, training-data coverage — would mean building an explanation
after seeing exactly one result at exactly one operating point, the same
selection risk this whole candidate exists to avoid.

The disciplined next step is to finish the experiment already frozen but not
yet run: G0's force grid has three levels (`50N`, `100N`, `150N`); G1 only
factored the channel at `100N`. `50N` and `150N` need the same `LR`/`RL`
collection (`RR`/`LL` again reused from G0) to know whether the reversal
found here is a property of this specific force magnitude or holds across the
grid. Only after that is answered does asking *why* become the right next
move rather than a premature one. See `G2_PREREGISTRATION.md`.
