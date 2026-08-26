# Topic 24 -- registered follow-up hypothesis (NOT EXECUTED)

> **Status: registered, not run.** No rollouts, no configs, no code beyond
> what is described here. This document exists so the reasoning behind the
> best follow-up hypothesis Topic 24's cross-task arm produced is not lost,
> in case a future session picks Topic 24 back up. It is deliberately not a
> full pre-registration (no frozen seeds, no committed config list) because
> the decision after
> [`G3_EXPLORATORY_LIFT_AUDIT.md`](G3_EXPLORATORY_LIFT_AUDIT.md) was to
> pause the experimental line here, compare against other candidate topics,
> and only come back to execute this if Topic 24 still looks like the best
> use of further effort.

## What G3 + its exploratory audit actually showed

`G3_RESULTS.md`: confirmatory panel STOPPED at
`PREREQUISITE_FAIL_REPLAY_FIDELITY` -- the 28-config eligible panel's 0N
baseline success (82.1%) did not clear the pre-registered 90% floor, so no
`delta_VLA` on binary success was ever licensed to be read as established.

`G3_EXPLORATORY_LIFT_AUDIT.md`: a post-hoc, non-preregistered look at the
same 392 rows through the continuous `final_target_lift_m` the binary
success is thresholded from. After the required corrections:

- Plain per-cell 95% CIs made `100N/right` and `150N/left` look significant.
  This was **not** a valid inferential claim -- six cells were scanned
  without multiplicity correction.
- A studentized max-T sign-flip permutation (config-clustered, family-wise
  corrected across all six cells) found **no cell significant at 0.05**
  (omnibus p = 0.092; `100N/right` adjusted p = 0.092, `150N/left` adjusted
  p = 0.345).
- Leave-one-config-out: `150N/left`'s effect depends entirely on 3 of 28
  configs and collapses if any one is removed -- fragile, effectively
  demoted.
- `100N/right` is comparatively sturdy: the point estimate stays in a narrow
  negative band under every single-config removal, and its asymmetric tail
  (`fresh`-worse catastrophic swings, zero counterexamples in the opposite
  direction) is stable across four different swing thresholds
  (0.10/0.15/0.20/0.25m: `5/5/4/4` fresh-worse vs `0/0/0/0` replay-worse).

None of this is a result. It is a hypothesis-generator with an unusually
clean shape: **not** "live VLA feedback makes average outcomes slightly
worse everywhere," but "in a small, consistent subset of disturbed
configs, live VLA feedback appears associated with a qualitatively worse
failure -- and this association shows up nowhere near significance in the
mean, because it is sparse, but shows up with zero counterexamples in the
one-sided tail at every threshold tried."

## The candidate confirmatory question

> Can online VLA replanning increase the probability of a catastrophic
> failure transition after a disturbance, even when it does not change
> average task success?

This is a different -- and, if it replicates, more interesting -- claim
than anything G0-G3 tested. It also explains, after the fact, why binary
success and the task's own clipped reward saw nothing: `clip((z-z0)/0.1,
0, 1)` collapses "missed the grasp," "object knocked off the table," and
"object flung across the room" into the identical `reward = 0`. If feedback
changes *which* of those a disturbed episode lands in without changing
whether it lands in "failure" at all, no reward-thresholded endpoint could
ever have seen it -- which is consistent with G3's own binary result being
silent on this.

## What a confirmatory follow-up would need to look like, if run

**Primary endpoint must be a pre-defined physical event, not a post-hoc
lift threshold.** `|Delta_lift| > 0.2m` was chosen after looking at the
data; using it (or any of the four thresholds tried) as a confirmatory
cutoff would repeat exactly the mistake this audit was written to catch.
Candidate operational definition, computed directly from MuJoCo contact
state rather than any threshold on `final_target_lift_m`:

```text
C = 1 if the target object contacts the ground plane (or leaves the table's
    support region) at any point after the push, else 0.
```

This should be nailed down from the sim geometry (ground-plane geom id,
table bounding region) the same way `canonical_reconnaissance.py` nailed
down the target body id and the right-hand kinematic subtree for G3 -- not
inferred from a lift value after the fact.

**Design**: paired per config, `C_fresh` vs `C_vla_replay`, on the single
force/direction cell this hypothesis is about (`100N/right` -- not a fresh
six-cell scan; the six-cell scan was what needed correcting for here, and a
confirmatory follow-up on this specific finding should not repeat it).
Analysis: paired McNemar / paired randomization test on the 2x2 discordance
table, not a difference-in-means bootstrap -- this matches what the
underlying phenomenon actually looks like (sparse discordant switches, not
a graded shift). `final_target_lift_m` demotes to a secondary continuous
severity endpoint (how bad was the failure, given that one happened),
reported but not primary.

**Sample**: a wholly new, independent config pool -- not the 28 configs G3
already used, and specifically not the 4-5 configs that produced the
`100N/right` tail here. The system is deterministic under the frozen
virtual clock; re-running the same configs (even with different upstream
seeds, if any exist) does not manufacture new independent evidence. A fresh
draw from the task's own DR-level config generation is required.

**Eligibility ("nominally solvable"), decided before any push data exists**
-- tighter than G3's `timing_eligible` criterion, and evaluated only from
canonical (0N) rollouts:

```text
1. fresh canonical succeeds
2. vla_replay canonical is lossless / identical outcome to fresh
3. right-hand/target contact exists during the canonical approach
4. a valid push_tick exists (same rule as G3's CANONICAL_RECONNAISSANCE.md)
```

Only configs passing all four enter the confirmatory panel. This is not a
retroactive filter on the 392 rows already collected -- it is a
pre-disturbance screen applied to an entirely new draw of configs, deciding
membership before any fresh/vla_replay/push data exists for them, the same
way G3's own `timing_eligible` rule was decided from canonical
reconnaissance before any disturbed rollout ran.

**Sample size**: 60-80 eligible configs, not the ~40-55 a naive power
calculation off the exploratory `100N/right` effect
(`delta_lift ~ -0.079`, `t ~ -2.32`, `n=28` -> `d_z ~ 0.44` -> ~40-55 for
80-90% power on a single pre-registered continuous endpoint) would suggest.
That calculation is itself a winner's-curse estimate from the same
exploratory pass being used to justify the follow-up, and the actual
primary endpoint is about to change from a continuous mean to a sparse
discordant-event rate, which needs more, not fewer, independent
observations to power adequately. 60-80 is deliberately conservative
against both problems.

## Why this is not being executed now

Evidence-strength comparison, as of this registration:

```text
CloseDoor (G0-G2):    a real, preregistered, task-success-endpoint,
                       channel-level, bidirectional causal result.
XMoveBendPick (G3 +
  exploratory audit):  a structured, correction-surviving-in-shape but not
                       in-significance hypothesis (p ~ 0.09-0.35 depending
                       on cell, after correction), not a result.
```

A `p ~ 0.09` exploratory finding does not justify committing several more
days of compute to a new confirmatory panel immediately. The decision made
alongside this registration was: commit the exploratory audit, stop the
experimental line on XMoveBendPick here, and only return to execute this
follow-up if a comparative look at other candidate topics still leaves
Topic 24 as the most promising direction.

## What would change the calculus

If this follow-up is eventually run and the catastrophic-transition
association at `100N/right` (or whatever cell an independent confirmatory
panel targets) replicates, that upgrades Topic 24's cross-task claim from
"CloseDoor's left/right channel reversal is a property of one task's push
geometry" to something structurally deeper: **online replanning can change
not only whether a robot fails, but how it fails -- occasionally pushing an
otherwise-recoverable trajectory into a qualitatively worse failure basin.**
That claim, if it survives a properly powered, pre-registered,
independent-sample confirmatory test, would be a substantially stronger
cross-task result than anything currently in this project.
