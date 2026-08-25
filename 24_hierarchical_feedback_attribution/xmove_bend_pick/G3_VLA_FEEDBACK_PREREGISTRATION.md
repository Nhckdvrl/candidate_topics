# Topic 24 G3 pre-registration (corrected): does online VLA feedback have reliably positive causal value under disturbance on a second task?

> Written and committed **before any push-condition row exists** — supersedes
> `G3_PREREGISTRATION.md` before that document's panel was ever run.

## The correction, and why it matters before any data exists

The superseded design asked G0's question again on a new task: where does
recovery live between the WBC and the VLA. That is not where Topic 24's live
thread actually is. G1 factored G0's pooled near-zero `delta_high` into a real,
direction-dependent VLA effect (`+0.20` left, `-0.20` right at 100N); G2 showed
that effect is not established at every force. The open, cross-task question is
not "which layer gets credit" — it is:

> **Does live VLA replanning reliably help under physical disturbance, or does
> its value depend on the disturbance and task even when a capable low-level
> controller stays fully online?**

This is a target correction made because the original design was aimed at the
wrong question, not because any result forced a change — no push-condition row
had been collected under the old design, so nothing here is a post-hoc
reinterpretation of data.

## The comparison

```text
fresh        live observation -> live VLA -> live WBC -> robot
vla_replay   recorded pre-disturbance VLA plan -> live WBC (live proprio) -> robot
```

`actuator_replay` is dropped entirely. The WBC is live in **both** conditions
now — the only thing that differs is whether the VLA gets to see the
disturbed state and re-plan, or whether it is held to the plan it already
committed to before the push. This isolates exactly one causal question: with
a capable low-level controller staying online in both arms of the comparison,
does spending a VLA replan on this observation help.

```text
delta_VLA(f, d) = S_fresh(f, d) - S_vla_replay(f, d)
```

## Why `vla_replay` is already a clean intervention here, at no extra cost

Nothing new needs to be proven about the seam for this comparison to be valid
— it was already proven by `P0'`/`P0b'`, run *before* this correction and not
wasted by it:

- `P0'` (`records/p0_xmove_records.jsonl`): `vla_replay` reproduces `fresh`
  exactly under no disturbance (`9/10` both, gap `0.000`), so the tape is
  lossless.
- `P0b'` (`records/p0b_xmove_records.jsonl`): the WBC seam is genuinely
  state-dependent on this task (`repeat=0`, `restore=0`, `perturb` clearly
  above that floor on all 3 configs tested) — the same `12 leg + 3 waist`
  joint signature G0/G1 found on CloseDoor, replicated fresh on a different
  task rather than assumed.

So `vla_replay` on this panel means exactly what it is supposed to mean: *only
the high-level VLA feedback is removed; the downstream physical controller
still has full access to the live disturbed state.* A capable low-level
controller staying online in both arms is not an assumption here, it is
already a measured property of this task's own instrument.

## Frozen panel

```text
task          simple/G1WholebodyXMoveBendPickTeleop-v0
policy        Psi0 ckpt_40000
configs       28 of 30, unchanged from CANONICAL_RECONNAISSANCE.md
conditions    fresh / vla_replay   (actuator_replay dropped)
force grid    0 N control  +  {50, 100, 150} N  x  {left, right}
push body     torso_link, 0.2 s, unchanged from G0/the superseded G3 design
timing        per-config right-hand/target contact anchor, unchanged
horizon       800 control steps, unchanged
clock         virtual (50 Hz), unchanged
success       unmodified upstream env.unwrapped._success
```

Total rollouts: `28 configs x 7 force/direction cells x 2 conditions = 392` —
a third less than the superseded design's 588, and the narrower question is
the reason, not a cost-cutting motive stated after the fact.

## Primary statistic: the full grid, not a pooled number

G0's own history is the reason this is non-negotiable: a pooled `delta_high`
near zero looked like "VLA replanning barely matters" until G1 showed it was
`+0.30/+0.23/+0.07` (left) and `-0.10/-0.20/-0.21` (right) cancelling. G3 will
report and read the full `force x direction` grid of `delta_VLA` values before
any pooled summary, exactly the discipline G2 already applied.

```text
delta_VLA(50,left)    delta_VLA(50,right)
delta_VLA(100,left)   delta_VLA(100,right)
delta_VLA(150,left)   delta_VLA(150,right)
```

`0N` is a replay-fidelity control only, not a scientific cell.

## Frozen verdict, defined before any cell is read

A cell is **established** if its clustered-bootstrap 95% CI excludes zero and
`|delta_VLA| >= 0.10` (unchanged minimum worthy effect and bootstrap procedure
from G0/G1/G2 — not reinvented).

```text
CONSISTENTLY_HELPFUL          every established cell is positive, and at
                              least one cell is established
SIGNED_HETEROGENEITY          at least one established positive cell AND
                              at least one established negative cell
CONSISTENTLY_HARMFUL          at least one established negative cell, and
                              no established positive cell
NO_ESTABLISHED_VLA_VALUE      no cell is established in either direction
```

CloseDoor already showed all three of positive, negative and (at some
force/direction combinations) unestablished cells within one task. XMoveBendPick
does not need to reproduce that same left/right geometry — carrying the
"navigation channel" finding over as a starting assumption here would be
exactly the selection bias this correction exists to avoid, since that finding
was localized on CloseDoor's specific push geometry, not derived from anything
architecture-general. What XMoveBendPick needs to show is whether *the property
itself* — that live VLA feedback's value is not uniformly positive — holds on a
second, structurally different task (locomotion + bend + grasp, not a
door-approach), where the upper-body channel plausibly matters far more than it
did for CloseDoor's base-heavy interaction.

## What is explicitly deferred

**No channel factorization (`RR`/`LR`/`RL`/`LL`) in this panel.** That
question is only asked as `G3b`, and only if this panel's result motivates
it — i.e. only if some cell shows `delta_VLA < 0` or the grid shows signed
heterogeneity. Running the factorization now, before knowing whether there is
anything to factorize, would import CloseDoor's navigation-channel
localization as a prior onto a task it was never established on. Sequence
stays: phenomenon on this task first, causal localization within it second —
never the reverse.

## Structural prerequisites, checked before any delta is read

```text
PREREQUISITE_FAIL_STRUCTURAL         a vla_replay row contacted the policy
                                     server, a tape exhausted early, a
                                     control cell got a push, a force cell
                                     got none, or the push tick differed
                                     between fresh and vla_replay in a cell
PREREQUISITE_FAIL_REPLAY_FIDELITY    the force=0 control column fails to
                                     reproduce P0' fidelity under the exact
                                     G3 code path
PREREQUISITE_FAIL_PUSH_INEFFECTIVE   at the largest force, median base
                                     displacement from the canonical
                                     trajectory is below 0.02 m
INSUFFICIENT_MATCHED_CONFIGS         any grid cell has fewer than 22 of 28
                                     complete fresh/vla_replay pairs
```

## Files

- `g3_runner.py` — collects `fresh`/`vla_replay` only; no `actuator_replay`
  logic. Records the canonical tape (for `vla_replay`) exactly as G0's runner
  does, adapted for this task's target/right-hand contact tracking already
  built and verified in `canonical_reconnaissance.py`.
- `g3_core.py` — the frozen decision procedure above.
- `tests/test_g3_core.py` — pure-logic tests before any real rollout.
