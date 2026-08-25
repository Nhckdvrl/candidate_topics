# Topic 24 G3 pre-registration: does WBC-level recovery dominance replicate on XMoveBendPickTeleop?

> **SUPERSEDED BEFORE ANY PUSH DATA.** No push-condition row was ever
> collected under this design. The scientific question below re-asked G0's
> WBC-vs-VLA question on a second task, but Topic 24's actual live thread is
> the VLA-centric one G1/G2 opened (is online VLA feedback reliably
> positive-value, not where does credit sit between two hierarchy layers).
> Replaced by
> [`G3_VLA_FEEDBACK_PREREGISTRATION.md`](G3_VLA_FEEDBACK_PREREGISTRATION.md),
> written before this file's `actuator_replay` condition or any push data was
> ever run. Left in place, not deleted, as the record of that correction.
> Nothing below was invalidated by a result — this is a target correction,
> not a stop.
>
> The panel infrastructure this file froze is NOT superseded and carries
> forward unchanged: the eligible 28-config panel, the per-config contact
> timing anchor, the 800-step horizon, and the {0,50,100,150}N x {left,right}
> force grid, all in `CANONICAL_RECONNAISSANCE.md`.

---

> Written and committed **before any push-condition row exists**. The
> instrument (`P0'`/`P0b'`), the eligible panel and the timing anchor
> (`CANONICAL_RECONNAISSANCE.md`) are frozen first; this document freezes
> everything about the scientific panel itself.

## What G3 asks

G0 found `WBC_LEVEL_DOMINATES` on CloseDoor: of the recovery `fresh` shows
over `actuator_replay`, the WBC/reference-generation seam accounts for
essentially all of it. G1/G2 then found that pooled result concealed a
push-direction-dependent reversal concentrated in the navigation channel,
strongest at `100N` and not established at `50N`/`150N`.

G3 asks the cross-task question those results cannot answer on their own:
does the WBC-dominance finding replicate on a structurally different task
(locomotion + bend + manipulation, not a door approach), or was it a property
of CloseDoor specifically? This is exactly the `RUN_LOCAL_AGENT.md` /
`G2_PREREGISTRATION.md` discipline applied one level up: prove the instrument
fresh (`P0'`/`P0b'`, both passed), then ask the identical frozen question
before asking anything about *why*.

## Frozen panel

```text
task          simple/G1WholebodyXMoveBendPickTeleop-v0
policy        Psi0 ckpt_40000 (...xmovebendpickteleop-v0...gpus7.2604100422)
configs       28 of 30 (dr-level-0/1/2 x episodes 0-9, minus
              dr-level-0:2 and dr-level-1:2 -- timing_ineligible,
              see CANONICAL_RECONNAISSANCE.md; not replaced)
conditions    fresh / vla_replay / actuator_replay
force grid    0 N control  +  {50, 100, 150} N  x  {left, right}
              identical magnitudes to CloseDoor's G0 -- same robot, same
              torso_link push point; not recalibrated for this task, so no
              new "equivalent disturbance" judgment call is introduced
push body     torso_link (unchanged from G0)
push duration 0.2 s (unchanged from G0)
timing        per-config, from CANONICAL_RECONNAISSANCE.md:
              push_tick = first right-hand/target contact tick - round(1.0/control_dt)
              -- NOT a single constant; every eligible config carries its own
              frozen push_tick, already computed and recorded
horizon       800 control steps for all three conditions (task's own
              max_episode_steps, not CloseDoor's 450)
clock         virtual (nominal 50 Hz), declared in every record
success       unmodified upstream env.unwrapped._success
```

**The whole grid is reported.** No force or direction is selected after
seeing an outcome, exactly as in G0.

Total rollouts: `28 configs x 7 force/direction cells x 3 conditions = 588`.

## What changed from G0, and why each change is not a new judgment call

- **Timing is per-config, not one global tick-fraction or one global
  constant.** This is a *stricter* rule than G0's, not a looser one: G0 used
  one push-lead rule (`first_contact_step - 1.0s`) applied identically to
  every config because CloseDoor's door-contact timing was itself already a
  per-config quantity computed from the canonical rollout. G3 does exactly
  the same thing; a "single anchor" was never the actual rule, "one second
  before this config's own canonical interaction" is.
- **Horizon is 800, not 450.** Read from `task.metadata` at runtime and
  logged on every P0'/reconnaissance run (`CANONICAL_RECONNAISSANCE.md`),
  not hand-copied from CloseDoor.
- **Contact/effect tracking uses `info["target"][2]` and the
  right-hand/target-body contact probe**, not a hinge joint. This follows
  directly from the task's own `compute_reward` implementation, not an
  independent design choice.

Everything else — push magnitude, push body, push duration, the three
conditions, the virtual clock, the structural proof-of-fire checks — is
carried over unchanged from G0, deliberately, so a difference in the result
cannot be attributed to an incidental protocol change.

## Structural prerequisites, checked before any delta is read

```text
PREREQUISITE_FAIL_STRUCTURAL         a replay contacted the policy server, a
                                     tape exhausted early, a control cell got
                                     a push, a force cell got none, or the
                                     push tick differed across the three
                                     conditions of a cell
PREREQUISITE_FAIL_REPLAY_FIDELITY    the force=0 control column fails to
                                     reproduce P0' fidelity under the exact
                                     G3 code path
PREREQUISITE_FAIL_PUSH_INEFFECTIVE   at the largest force, median base
                                     displacement from the canonical
                                     trajectory is below 0.02 m
INSUFFICIENT_MATCHED_CONFIGS         any grid cell has fewer than 22 of 28
                                     complete condition triples (matches G0's
                                     proportional floor: 24/30 ~ 0.80;
                                     22/28 ~ 0.79)
```

## Stop outcomes that are results, not failures

```text
NO_ROBUSTNESS_PHENOMENON                 fresh and actuator_replay both
                                         >= 0.90 at every force
FRESH_COLLAPSE_NOTHING_TO_ATTRIBUTE      fresh <= 0.10 at every force
NO_MEANINGFUL_LEARNED_FEEDBACK_CONTRIBUTION
                                         both deltas below the pre-registered
                                         minimum worthy effect of 0.10, or
                                         their clustered-bootstrap CI includes
                                         zero
```

## What G3 is not

Not a repeat of the G1/G2 channel-factorization question. If G3 finds
`WBC_LEVEL_DOMINATES` again, or finds a different pooled result, the channel
question on *this* task is a separate, later decision — not run inside this
panel. G3 answers exactly one question: where does recovery live on a second,
structurally different task, under the identical three-condition instrument
that already worked on CloseDoor.

## Files (to be added alongside this document)

- `g3_runner.py` — canonical tape recording, the push, and the three
  conditions, adapted from `topic24_runner.py` for this task's contact/target
  tracking.
- `g3_core.py` — the frozen decision procedure, structurally identical to
  `g0_core.py` with the 22/28 floor and the 28-config panel.
- `tests/test_g3_core.py` — pure-logic tests before any real rollout.
