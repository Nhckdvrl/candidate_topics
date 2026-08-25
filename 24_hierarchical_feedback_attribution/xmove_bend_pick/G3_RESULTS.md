# Topic 24 G3 result: does live VLA feedback have positive value on XMoveBendPickTeleop?

**Verdict: `PREREQUISITE_FAIL_REPLAY_FIDELITY`.** A stop, not a finding about
VLA feedback. The panel ran clean to completion (392/392, zero structural
violations), but the frozen instrument gate on the force=0 control column
fails, so no `delta_VLA` cell below is licensed to be read as evidence either
way. Reported in full anyway, exactly as a null result would be, because a
stop is a result too.

## What actually ran

```text
task           simple/G1WholebodyXMoveBendPickTeleop-v0
configs        28 of 30 (eligible per CANONICAL_RECONNAISSANCE.md)
conditions     fresh / vla_replay (no actuator_replay -- dropped in the
               post-hoc-avoidance correction, see G3_VLA_FEEDBACK_PREREGISTRATION.md)
force grid     0N control + {50,100,150}N x {left,right}
rows           392/392, structural_violations: []
```

## Frozen gates, checked in order

```text
                          required        observed         result
fidelity_control (0N)     >= 0.90         0.821 (23/28)    FAIL
  fresh                                   0.8214285714
  vla_replay                              0.8214285714
  drop (fresh-vla_replay) <= 0.10         0.000            (would pass alone)
push_effective (150N)     >= 0.02 m       0.091 m          PASS
```

The two numbers that matter are different failures. `vla_replay` reproduces
`fresh` on the control column **exactly** (0.8214285714... on both sides,
23/28 both conditions) -- the replay mechanism itself is not lossy on this
task, matching what P0' already established. What fails is the **absolute
floor**: 23/28 is 82.1%, and the preregistered gate requires >= 90% before a
`delta_VLA` comparison built on top of it is trusted. The replay is faithful
to a baseline that is itself worse than the bar this design required.

## Full force x direction grid (reported in full, not read selectively)

```text
force  dir     fresh    vla_replay   delta_VLA   n
0N     none    0.821    0.821        0.000       28  (control; fails fidelity floor)
50N    left    0.357    0.321        +0.036      28
50N    right   0.536    0.464        +0.071      28
100N   left    0.036    0.036        0.000       28
100N   right   0.000    0.000        0.000       28
150N   left    0.036    0.036        0.000       28
150N   right   0.000    0.000        0.000       28
```

No bootstrap CIs or `established` calls are reported per-cell because the
prerequisite gate stops the procedure before that stage runs (see
`g3_core.py::evaluate`, which returns `PREREQUISITE_FAIL_REPLAY_FIDELITY`
immediately after the fidelity check and does not compute per-cell CIs at
all). The `delta_VLA` column above is descriptive, not a licensed causal
estimate.

## Why the fidelity floor fails here and not in P0'

P0' deliberately ran only on `dr-level-0`, the one DR level where upstream's
own published number is `10/10` -- chosen precisely so a `>=0.90` gate would
measure the instrument, not the policy's own difficulty curve
(`P0_XMOVE_RESULTS.md`). G3's 28-config panel spans all three DR levels
(`dr-level-0/1/2`, minus the two timing-ineligible configs), where upstream's
own published competence is `10 | 9 | 9` -- an unweighted average of 93.3%,
but the *eligible* subset used here lands at 82.1%. This is not evidence the
instrument is broken; it is evidence that the eligible-config filter
(contact-timing eligibility, frozen before any push data existed, per
`CANONICAL_RECONNAISSANCE.md`) correlates with configs where the base policy
is already less reliable. That correlation was never checked at
pre-registration time, because pre-registration explicitly forbade choosing
or re-checking the config set after seeing outcome data. It is being reported
now, after the fact, as a design limitation discovered by the gate doing its
job -- not smoothed over.

## A floor effect the fidelity failure does not fully explain

Even setting the fidelity gate aside, look at what the grid would say if read
naively: at 100N and 150N, `fresh` and `vla_replay` are identical in both
directions and both near zero (0.036/0.000). This is not evidence VLA
feedback "does nothing" at high force -- it is a **floor effect**: task
success has already collapsed to near-zero under both conditions, leaving no
headroom for a `fresh` vs `vla_replay` difference to appear regardless of
whether one exists. The only cells with any daylight between conditions are
50N left (+0.036) and 50N right (+0.071), both small and neither screened
through a bootstrap CI because the panel never reached that stage.

## What this does and does not license

- Does not license `NO_ESTABLISHED_VLA_VALUE`, `CONSISTENTLY_HELPFUL`,
  `CONSISTENTLY_HARMFUL`, or `SIGNED_HETEROGENEITY`. All four require passing
  the fidelity gate first; none of them fired.
- Does not license "VLA feedback doesn't matter on this task" -- the 100N/150N
  near-identical numbers are as consistent with a floor effect as with a real
  null.
- Does not retroactively cast doubt on P0'/P0b': those verified the identical
  mechanism (`vla_replay` fidelity, WBC seam liveness) on a panel where the
  fidelity floor was satisfiable by construction. What changed here is the
  config panel's composition, not the mechanism.
- Does not license re-running with a different config selection, a lower
  fidelity floor, or a different DR-level mix chosen after seeing this number.
  That would be exactly the after-the-fact adjustment this project's
  discipline exists to prevent.

## What comes next is not G3b

G3b (channel factorization) was explicitly conditioned on this panel showing
an established negative cell or signed heterogeneity
(`G3_VLA_FEEDBACK_PREREGISTRATION.md`, "What is explicitly deferred"). Neither
happened -- the panel never reached the stage where either could be
established. Running G3b now would be factorizing a comparison that was never
licensed in the first place. The open question this result actually raises is
narrower and upstream of channel factorization: a fresh preregistration for
G3-retry would need a config-eligibility rule that does not correlate with
depressed base competence, decided before seeing any new push data, not
re-fit to this result.

Records: [`records/`](records/) (392 rows across 9 per-worker files:
`dr0_w1/w2/w3.jsonl`, `dr1_w1/w2/w3.jsonl`, `dr2_w1/w2/w3.jsonl`, plus the
merged [`records/g3_merged.jsonl`](records/g3_merged.jsonl)). Result:
[`g3_result.json`](g3_result.json), independently re-run against the merged
392-row file to confirm -- the watcher's own evaluator invocation hit an
unrelated `python` interpreter-path error immediately after merging
(`PYTHONPATH=. /home/xiang/venvs/ragen/bin/python`, a path that does not exist
on the machines this ran on), so its exit status could not be trusted without
a fresh, independently-invoked run. The merge itself (`g3_merged.jsonl`,
written before the evaluator call) was already complete and correct; re-running
`g3_core.py` against it with a working interpreter reproduced byte-identical
numbers to the file the watcher had already written on an earlier, successful
invocation, so the numbers above stand.
