# Topic 24 G3 -- POST-HOC EXPLORATORY: does binary success hide continuous
# effects of VLA feedback under floor conditions?

> **This is explicitly post-hoc and exploratory. It does not change the G3
> confirmatory verdict, which remains `PREREQUISITE_FAIL_REPLAY_FIDELITY`
> (see [`G3_RESULTS.md`](G3_RESULTS.md)).** No new rollouts were run. This
> reuses the same frozen 392-row panel and asks a different question of it,
> after the confirmatory gate had already failed and the confirmatory
> analysis was already reported.
>
> **Revision note.** An earlier draft of this file called two cells
> (`100N/right`, `150N/left`) "established" from plain per-cell 95%
> bootstrap CIs. That was wrong to call established: `raw lift` is not the
> preregistered primary endpoint, and six cells were scanned without
> correcting for multiplicity. This revision replaces that language
> throughout with `exploratory signal`, and adds the three checks the
> multiplicity/fragility/threshold-choice concerns required: a studentized
> max-T sign-flip permutation test (family-wise error control across the six
> cells), leave-one-config-out fragility, and threshold-choice sensitivity
> for the "catastrophic swing" counts. **Neither cell survives the max-T
> multiplicity correction at the conventional 0.05 level.** The findings
> below are downgraded accordingly.

## The question

`G3_RESULTS.md` reports `success` at every disturbed cell, and at 100N and
150N (all four directions) that number is at or near the task's own floor:
`fresh` and `vla_replay` both land at 0.000 or 0.036 success. Binary success
cannot distinguish "missed the grasp by a little" from "the object left the
table" -- both are `success=False`.

But `g3_runner.py` recorded `final_target_lift_m` on every row from the
start, and the task's own `compute_reward` is itself continuous before it is
thresholded:

```text
reward = clip((z - z0) / 0.1, 0, 1)     # z0 = init height, 0.1m = full-credit lift
success = reward >= 0.8
```

So there are two more informative statistics sitting in the same 392 rows,
neither of them an invented metric:

```text
Delta_lift(f, d)   = final_target_lift_m[fresh] - final_target_lift_m[vla_replay]
Delta_reward(f, d) = clip(lift_fresh/0.1, 0, 1) - clip(lift_replay/0.1, 0, 1)
```

`Delta_reward` is the exact quantity `success` is thresholded from, computed
without the threshold.

## Pre-declared kill rule

Set before this script ran, by the person requesting it: if the full six-cell
`Delta_lift`/`Delta_reward` grid shows no structure -- deltas indistinguishable
from zero with signs that do not track anything systematic -- Topic 24 stops
on XMoveBendPickTeleop. No cell, direction, or force was selected after
seeing a result; all six perturbed cells plus the 0N control are reported.

## 0N control: sanity check, not a finding

```text
Delta_lift   = 0.000000  (exact, all 28 configs)
Delta_reward = 0.000000  (exact, all 28 configs)
```

Fresh and vla_replay produce byte-identical `final_target_lift_m` under no
disturbance, at the continuous level, not just the binary one. Nothing about
the replay mechanism itself is contributing noise to what follows.

## Full six-cell grid (plain per-cell bootstrap CI -- not multiplicity-corrected)

| force | dir | n | mean lift fresh | mean lift replay | Delta_lift (95% CI, uncorrected) | mean reward fresh | mean reward replay | Delta_reward (95% CI, uncorrected) |
|---|---|---|---|---|---|---|---|---|
| 50N | left | 28 | -0.020 | -0.052 | +0.032 [-0.021, 0.096] | 0.322 | 0.281 | +0.041 [-0.151, 0.235] |
| 50N | right | 28 | +0.010 | +0.017 | -0.007 [-0.064, 0.047] | 0.479 | 0.412 | +0.067 [-0.148, 0.284] |
| 100N | left | 28 | -0.065 | -0.047 | -0.017 [-0.063, 0.016] | 0.034 | 0.030 | +0.003 [-0.091, 0.101] |
| 100N | right | 28 | -0.106 | -0.027 | -0.079 [-0.150, -0.020] | 0.001 | 0.005 | -0.004 [-0.016, 0.003] |
| 150N | left | 28 | -0.084 | -0.033 | -0.051 [-0.111, -0.005] | 0.030 | 0.033 | -0.003 [-0.009, 0.0001] |
| 150N | right | 28 | -0.271 | -0.309 | +0.038 [-0.071, 0.149] | 0.00004 | 0.00001 | +0.00003 [-0.00002, 0.0001] |

`Delta_reward` excludes zero in *no* cell, uncorrected or not: `clip(x,0,1)`
floors both conditions to (near) zero once lift goes negative, which is the
same floor problem that already erased the signal from binary `success`,
one step less severe. **The continuous quantity that carries any signal at
all is the raw, unclipped lift -- not the task's own officially clipped
reward.** The clipping at 0 for reward already does most of the damage that
thresholding at 0.8 for success finishes off.

At the plain, uncorrected level, `Delta_lift` at **100N/right** and
**150N/left** has a 95% CI that excludes zero, both in the direction of
`fresh` landing at a worse (more negative) final lift than `vla_replay`. The
other four cells cross zero. The rest of this document tests how much that
survives once multiplicity, single-config fragility, and threshold choice
are accounted for.

## Check 1: studentized max-T sign-flip permutation (multiplicity control)

Method: one random +-1 sign per config (28 configs, shared across all six
cells for that config, preserving the correlation induced by reusing the
same configs across cells), applied to that config's `Delta_lift` in every
cell simultaneously. Studentized `t = mean/(sd/sqrt(n))` per cell per
permutation; `T_max` = max `|t|` over the six cells. 100,000 permutations,
seed `20260826`. Each cell's **maxT-adjusted p-value** is the fraction of
permutations whose *joint* `T_max` (max over all six cells, not that cell's
own marginal null) meets or exceeds that cell's own observed `|t|` -- this
is what makes it family-wise-error-controlled (single-step Westfall-Young
maxT), as opposed to each cell's uncorrected marginal permutation p-value.

```text
T_obs_max = 2.317   (achieved by 100N/right)
q95(null T_max) = 2.431
omnibus p (global null: no cell differs) = 0.092
```

| cell | t_obs | maxT-adjusted p | simultaneous 95% CI | multiplicity-robust signal |
|---|---|---|---|---|
| 50N/left | +1.05 | 0.901 | [-0.042, 0.106] | no |
| 50N/right | -0.25 | 1.000 | [-0.077, 0.063] | no |
| 100N/left | -0.85 | 0.967 | [-0.067, 0.032] | no |
| **100N/right** | **-2.32** | **0.092** | [-0.161, 0.004] | **no** |
| **150N/left** | **-1.86** | **0.345** | [-0.117, 0.015] | **no** |
| 150N/right | +0.67 | 0.993 | [-0.099, 0.176] | no |

**Neither `100N/right` nor `150N/left` survives family-wise correction at
the conventional 0.05 level.** The omnibus test (is there *any* real
difference anywhere in the six-cell family) also does not reject the global
null (p=0.092). `100N/right` is the closest thing to a signal here -- it is
literally the cell that sets `T_obs_max`, and its adjusted p (0.092) equals
the omnibus p by construction -- but 0.092 is not a discovery by any
standard threshold, and the multiplicity-adjusted simultaneous CI for that
cell still touches zero (upper bound +0.004m).

This is the honest update the plain per-cell CIs in the previous revision of
this file did not carry: **once six cells are scanned and the comparison is
corrected for that, this dataset does not establish a real `Delta_lift`
effect anywhere, at conventional significance.**

## Check 2: leave-one-config-out fragility

For each of the 28 configs, drop it and recompute. Full bootstrap CI
recomputed on the remaining 27 for the two cells whose plain (uncorrected)
CI excluded zero.

```text
                    LOO point range        sign ever flips?   CI survives every single-config removal?
50N  left           [+0.012, +0.039]       no                 n/a (already crosses zero)
50N  right          [-0.021, +0.012]       yes (dr-level-0:4) n/a
100N left           [-0.024, +0.0002]      yes (dr-level-0:0) n/a
100N right          [-0.084, -0.062]       no                 YES -- no single removal re-crosses zero
150N left           [-0.056, -0.034]       no                 NO  -- removing ANY of 3 configs re-crosses zero
150N right          [+0.020, +0.059]       no                 n/a (already crosses zero)
```

**`100N/right` is not fragile to any single-config removal**: the point
estimate stays in a narrow negative band ([-0.084, -0.062]) no matter which
one of the 28 configs is dropped, and the bootstrap CI never re-crosses zero
on any single removal. That is consistent with this cell's mean being
carried by four configs pulling the same direction rather than one outlier
doing all the work (see Check 3).

**`150N/left` is fragile**: removing any one of three specific configs
(`dr-level-1:1`, `dr-level-1:3`, `dr-level-1:5`) sends the CI back across
zero. All three are exactly the three configs identified as the
"fresh-catastrophically-worse" tail for this cell in the original swing
count -- i.e. this cell's whole effect rests on 3 of 28 configs, and it does
not survive losing any one of them. Combined with Check 1 (this cell's
maxT-adjusted p was already the weaker of the two, 0.345), **`150N/left`
should be read as a fragile, non-robust observation, not a signal.**

## Check 3: catastrophic-swing threshold sensitivity

Repeating the fresh-worse vs. replay-worse tail count at four thresholds
instead of the single 0.20m cutoff used to first notice the pattern:

```text
                    |Delta|>0.10        |Delta|>0.15        |Delta|>0.20        |Delta|>0.25
                    fresh / replay      fresh / replay      fresh / replay      fresh / replay
50N  left           4 / 4               2 / 3               0 / 2               0 / 2
50N  right          5 / 3               4 / 2               1 / 1               1 / 1
100N left           3 / 1               1 / 1               1 / 0               1 / 0
100N right          5 / 0               5 / 0               4 / 0               4 / 0
150N left           3 / 0               3 / 0               3 / 0               3 / 0
150N right          4 / 7               4 / 7               4 / 7               4 / 7
```

Two things hold up across every threshold from 0.10m to 0.25m, not just the
0.20m cutoff that was picked after seeing the data:

- **`100N/right` and `150N/left` are one-sided at every threshold**: zero
  `replay`-worse catastrophic swings at any threshold in either cell. The
  asymmetry is not a threshold artifact.
- **`150N/right` is two-sided at every threshold** (roughly 4-vs-7, never
  collapsing to one side), and has *more* total catastrophic swings than any
  other cell. Whatever produces large basin-switches at 150N/right is not
  the same one-sided phenomenon as the two cells above.
- **`50N/left`, `50N/right`, `100N/left` do not show a stable pattern**:
  which side has more swings changes depending on the threshold, and the
  counts thin out fast (down to 0-2 by 0.25m). These three should not be
  read as showing directional structure at all.

So the *descriptive* claim -- "100N/right and 150N/left have a one-sided
tail of large fresh-worse swings, and this is not an artifact of the 0.2m
cutoff" -- is robust. The *inferential* claim -- that this one-sidedness
adds up to a statistically established mean difference once six cells are
properly corrected for -- is not (Check 1), and for 150N/left it is not even
robust to which 27-of-28 configs are used (Check 2).

## Reading the result against the pre-declared kill rule

The grid is not uniformly indistinguishable from zero in the descriptive
sense -- there is a real, threshold-stable, one-sided pattern of large
basin-switches at two of six cells, with literally zero counterexamples in
either cell across four different threshold choices. That clears the "no
structure at all, deltas bouncing randomly" bar the kill rule was written
against, in the weak, descriptive sense.

But after the multiplicity and fragility checks, the honest inferential
status is much lower than the first pass of this document claimed:

- No cell survives family-wise-corrected significance at conventional
  levels (Check 1). `100N/right` is the closest (p=0.092), not a discovery.
- `150N/left`'s effect depends entirely on 3 of 28 configs and does not
  survive removing any single one of them (Check 2).
- `100N/right`'s effect is comparatively sturdier under LOO (Check 2) and
  its asymmetric tail is stable under threshold choice (Check 3), but that
  sturdiness is about the *shape* of the phenomenon (a handful of configs
  reliably swinging the same direction), not about statistical
  significance, which this cell also fails to clear once corrected.

The most defensible framing of what these 392 rows show:

> **Binary task success did not identify a VLA feedback effect on this
> panel, but the underlying continuous physical outcome exposes a
> disturbance-dependent pattern -- concentrated in a small number of
> configs, most consistently at 100N/right -- that does not survive rigorous
> multiplicity/fragility testing at conventional significance, but is also
> not the "everything crosses zero, no structure" pattern that would justify
> stopping here.**

Put differently: this is not (yet) evidence of an effect. It is evidence
that the *shape* of a possible effect -- occasional, one-sided,
disturbance-specific catastrophic basin-switching, invisible to a floored
binary or clipped-reward endpoint -- is worth designing a properly powered,
pre-registered follow-up around, rather than evidence that stands on its
own. **CloseDoor's G1/G2 result is a preregistered, task-success-endpoint,
channel-level, bidirectional causal effect. This XMoveBendPick audit is a
post-hoc, raw-physical-endpoint, whole-VLA-intervention, exploratory pattern
that does not clear conventional significance after correction.** The two
are not at the same evidentiary level and should not be described as if
they were.

## What this does and does not license

- It does **not** reopen or revise the G3 confirmatory verdict. That gate
  failure is a fact about this 28-config panel's baseline competence, not
  about whether continuous effects exist under disturbance.
- It does **not** license calling anything here "established" or a
  "replication" of CloseDoor's effect. After correction, nothing here clears
  conventional significance.
- It **does** support keeping this thread open as a *design* question rather
  than a *result*: if this pattern is real, a properly powered, pre-registered
  panel targeting `Delta_lift` (or better, a task whose 0N baseline clears
  the fidelity gate this one did not) as a primary endpoint, sized to detect
  an effect this sparse (3-5 configs out of 28 driving it here), is the next
  legitimate step -- not a claim built on this exploratory pass alone.

Records: [`records/g3_merged.jsonl`](records/g3_merged.jsonl) (392 rows,
unchanged from G3). Scripts:
[`g3_exploratory_lift_audit.py`](g3_exploratory_lift_audit.py) (grid),
[`g3_exploratory_lift_robustness.py`](g3_exploratory_lift_robustness.py)
(maxT permutation, LOO, threshold sensitivity). Result JSON:
[`g3_exploratory_lift_audit_result.json`](g3_exploratory_lift_audit_result.json),
[`g3_exploratory_lift_robustness_result.json`](g3_exploratory_lift_robustness_result.json).
