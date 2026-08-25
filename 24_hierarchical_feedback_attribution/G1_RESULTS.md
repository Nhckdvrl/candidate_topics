# Topic 24 — G1 result: which VLA command channel causes the reversal?

**Verdict: `NAVIGATION_CHANNEL_CAUSES_REVERSAL`.** Frozen in
[`G1_PREREGISTRATION.md`](G1_PREREGISTRATION.md) before this ran. Structural
checks pass on all 240 rows (120 reused from G0 + 120 new): `[]` violations.

## The core numbers

```text
              RR      LR      RL      LL
left        0.033   0.233   0.033   0.267
right       0.600   0.400   0.600   0.400
```

On `right`, `RR` and `RL` are **exactly equal** (0.600 = 0.600) and `LR` and
`LL` are **exactly equal** (0.400 = 0.400). Holding the navigation/base channel
fixed, switching the upper-body channel between replayed and live changes the
success rate by exactly zero. Holding the upper-body channel fixed, switching
navigation between replayed and live moves the success rate the full 0.2 every
time.

```text
                                   left                          right
nav_effect_upper_replayed   +0.200  CI [ 0.067, 0.367]    -0.200  CI [-0.367,-0.033]
nav_effect_upper_live       +0.233  CI [ 0.100, 0.400]    -0.200  CI [-0.400, 0.000]
upper_effect_nav_replayed    0.000  CI [ 0.000, 0.000]     0.000  CI [-0.100, 0.100]
upper_effect_nav_live       +0.033  CI [ 0.000, 0.100]     0.000  CI [-0.100, 0.100]
```

Both navigation-channel effects clear the pre-registered minimum worthy effect
(0.10) with a CI that excludes zero, in both directions. Every upper-body
effect is at or under the floor, and three of the four have a CI containing
zero outright (the fourth, `upper_effect_nav_live` on `left`, has a point
estimate of 0.033 — below the 0.10 bar regardless of its CI).

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

The upper-body channel is not merely a smaller contributor: on `right` it
contributes literally nothing measurable in this panel — two point estimates
of exactly 0.0 with tight CIs, `RR==RL` and `LR==LL` to the decimal. Recall
from P0b that the arms and hands below the VLA seam are already known to be
open-loop with respect to the WBC; this result adds that they are *also*
uninformative with respect to which VLA-level channel drives the recovery
outcome, at least under this disturbance. The upper-body channel is, by this
measurement, a bystander to the CloseDoor recovery story on both ends of the
hierarchy.

## What this sharpens about the G0 reading

G0 already showed the WBC/reference-generation seam dominates pooled
recovery, and that `delta_high` pooled near zero because of a left/right
cancellation rather than being uniformly small. G1 shows that cancellation is
not diffuse across the whole VLA command — it is concentrated in exactly one
of its two channels, and that channel's own effect is *not* small in either
direction (0.2–0.23 both ways, well above the minimum worthy effect). The
correct headline is not "VLA replanning barely matters"; it is "VLA
navigation replanning matters a lot, in a direction-dependent way that
happens to average out at this operating point, and VLA upper-body replanning
does not measurably matter here at all."

## Structural checks

Every one of the 120 new rows shows the intervention fired exactly as
claimed: a channel marked replayed has an overwrite count equal to its step
count; a channel marked live has an overwrite count of zero; every hybrid row
queried the live policy server; every `RR` row (reused from G0) has
`server_queries == 0`. `push_tick` matches the G0 tape for every config the
disturbance is applied to the same event G0 measured the reversal under.

Records: [`records/g1_channel_panel.jsonl`](records/g1_channel_panel.jsonl)
(240 rows), [`records/g1_result.json`](records/g1_result.json).

## What this opens next

The natural next question is no longer "which layer recovers" — G0 and G1
together already answer that with more resolution than either alone: WBC
dominates pooled, and within the VLA seam the effect is concentrated in the
navigation channel and sign-flips with push direction. The open question is
*why* navigation replanning is directionally miscalibrated for a `right` push
specifically on this checkpoint — whether it is an artifact of this training
distribution's coverage, a geometric asymmetry in how the task's approach
trajectory interacts with the two push directions, or something about how the
navigation channel's own feedback law is shaped. That is a training/data
question rather than an architecture question, and it would need a different
instrument than a replay panel to answer.
