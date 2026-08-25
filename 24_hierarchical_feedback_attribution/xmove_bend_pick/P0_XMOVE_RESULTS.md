# Topic 24 cross-task — P0' result: replay fidelity on XMoveBendPickTeleop

**Verdict: `PASS`.** Fresh instrument re-verification on a second task, not an
assumption carried over from CloseDoor.

## Why this task, and why this checkpoint

`simple/G1WholebodyXMoveBendPickTeleop-v0`: G1 must walk to a table, bend, and
pick up a randomized `graspnet1b` object with the right hand
(`GraspObjectSpec(..., hand_uid="dex3_right")` in the task's own
`decompose()`). Same G1 humanoid, same Psi0/decoupled-WBC stack as CloseDoor.

Chosen for pre-existing, published competence, not for anything discovered
after running it: SIMPLE's own benchmark table lists Psi0 on this exact task
at `10 | 9 | 9` across the three DR levels — the released checkpoint
(`...xmovebendpickteleop-v0...gpus7.2604100422`, `ckpt_40000`, the only
checkpoint step published for this run) was competence-verified before any
local rollout.

Success is not a hinge-joint predicate like CloseDoor's door. The task's own
`compute_reward` reads `info["target"][2]` (the picked object's world-frame
z-height) and calls it a success at `>= 0.8` of a 0.1m normalized lift.
`info["target"]` is returned directly by `env.step()`/`env.reset()`, so no
MuJoCo body lookup was needed to track it descriptively here — same field the
task itself reads.

## Frozen gate

Deliberately **not** a mechanical copy of CloseDoor's threshold applied to a
mixed-DR sample. Upstream's own published competence on this task is not
10/10 everywhere (`10 | 9 | 9`), so requiring 10/10 replay fidelity on a
cross-DR sample would misread the *policy's own* natural variation as an
*instrument* failure. P0' runs on `dr-level-0` only, where upstream's own
number is `10/10` — the one level where a `>=0.90` fidelity gate is actually
measuring the instrument rather than the policy.

```text
                    required     observed
fresh               >= 0.90      0.900  (9/10)
vla_replay           >= 0.90      0.900  (9/10)
actuator_replay       >= 0.90      0.900  (9/10)
fresh - vla_replay        <= 0.10      0.000
fresh - actuator_replay   <= 0.10      0.000
```

Structural checks, all clean: `server_queries == 0` on every replay row;
`tape_exhausted_early` false on every row; every replay consumed the full
tape.

## One config fails, and it is not a fidelity problem

`dr-level-0:2` fails in all three conditions: `lift ≈ 0.00003m`, `steps=800`
(runs to the task's own horizon, never approaches the object). This is a
1-config deviation from the published `10/10` for this DR level — recorded
rather than smoothed over. It is not a replay-fidelity issue: `fresh`,
`vla_replay` and `actuator_replay` on this config produce byte-identical
trajectories (`final_target_lift_m` and `steps` match exactly across all
three), so the instrument reproduces this failure deterministically. Whatever
causes the released checkpoint to fail this particular scene draw locally
(likely stochastic-sampling or hardware-dependent variation from upstream's
own evaluation run, not investigated further here — out of scope for an
instrument check), all three conditions agree on it perfectly.

## Horizon and cadence, read from the running system rather than assumed

`max_episode_steps = 800`, confirmed printed at runtime from
`task.metadata` — not copied from CloseDoor's `450`. Control cadence
(`control_dt = 4 * SIMULATE_DT`) comes from the same `sonic_config` machinery
used everywhere else in Topic 24, task-independent by construction (it is a
whole-body-controller property, not a task property), and is logged on every
run for audit rather than assumed to be the same number as CloseDoor's.

Records: [`records/p0_xmove_records.jsonl`](records/p0_xmove_records.jsonl)
(30 rows, dr-level-0 episodes 0-9).
