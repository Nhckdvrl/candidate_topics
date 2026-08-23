# Topic 23 — G0 Results

**Status: G0 COMPLETE — BOTH FROZEN PANEL TASKS FAIL PREREQUISITES.**

The registered question was whether a released robot foundation policy preserves a
task effect after the demonstrator's canonical right-side motor route is removed.

On the frozen first target — Psi0 `ckpt_40000` on SIMPLE — that question **cannot be
identified**, because no task in the frozen panel satisfies both necessary
conditions at once:

```text
CloseDoor    policy is competent (30/30)  but the arm carries no motor program
OpenFaucet   the arm carries a motor program  but the policy is only 3-4/10
```

These are complementary failures, and both are measured in outcome space rather
than inferred from a proxy. This is a clean negative about the *measuring
instrument*, not a refutation of motor equivalence.

## Headline: the original design would have produced a false positive

The panel registered on 2026-08-24 had four conditions. Run on CloseDoor, it
yields exactly the "cleanest event" the registration was looking for:

| quantity (original 4-condition panel) | value | frozen bar | |
| --- | ---: | ---: | --- |
| `canonical` success | 30/30 | ≥ 0.70 | pass |
| `right_disabled` success | 29/30 | ≥ 0.20 | pass |
| `full_hold` success | 0/30 | ≤ 0.10 | pass |
| paired `right_disabled − full_hold` | 0.967, 95% CI [0.90, 1.00] | CI low > 0 | pass |
| substitution events | 29 | ≥ 5 | pass |

Every gate passes, overwhelmingly. With an oracle it would have reported
`PROMISING_MOTOR_SUBSTITUTION`.

It is wrong. Psi0 closes the door by **walking into it with the hand that already
hangs at its side**. The shoulder and elbow move less than 5 degrees. There is no
right-arm motor program, so nothing was removed and nothing was substituted. The
two conditions added in revision 2 are what catch this; see
[VALIDATION_AUDIT.md](VALIDATION_AUDIT.md#revision-2-2026-08-24).

## CloseDoor — `PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM`

`simple/G1WholebodyCloseDoorTeleop-v0`, 30 matched configs (dr-level-0/1/2 × 10),
all six policy conditions, mujoco rendering.

| level | canonical | right_frozen | right_disabled | left_disabled | both_arms_disabled | full_hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dr-level-0 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 |
| dr-level-1 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 | 0/10 |
| dr-level-2 | 10/10 | 9/10 | 9/10 | 9/10 | 10/10 | 0/10 |
| **all** | **30/30** | **29/30** | **29/30** | **28/30** | **30/30** | **0/30** |

| gate | value | bar | result |
| --- | ---: | ---: | --- |
| matched configs | 30 | ≥ 20 | pass |
| canonical success | 1.000 | ≥ 0.70 | pass |
| `canonical − right_frozen` | **0.033** | ≥ 0.20 | **FAIL** |
| canonical right-side route rate | 0.967 | ≥ 0.70 | (pass) |
| max clamp leak | 0.187 rad | ≤ 0.20 | (pass) |
| `both_arms_disabled` | **1.000** | ≤ 0.10 | (would also FAIL) |
| `full_hold` | 0.000 | ≤ 0.10 | (pass) |

Two independent prerequisites fail. Freezing the right arm in place costs the
policy 1 episode in 30, and removing **both** arms costs it nothing at all.

### Why: the canonical route has no motor program

Per-step MuJoCo contact attribution on a canonical rollout, under the official
`mujoco_isaac` sim mode:

```text
right_shoulder_pitch  range 0.053 rad     left arm ranges <= 0.24 rad
right_shoulder_roll   range 0.046 rad     base xy travel  0.734 m
right_elbow           range 0.077 rad     only contact part: right_hand
right_wrist_yaw       range 0.309 rad
contact ends at step 258; the door coasts from -0.062 to -0.166 unaided
official success at step 269
```

The same probe under `sim_mode=mujoco` reproduces this (wrist ranges 0.231/0.126,
base travel 0.720 m, `right_hand` the only contact), so it is not a rendering
artifact. The hand is a passive bumper transported by locomotion.

Retracting the arm does not help either: the robot's neutral at-side pose *is*
where the arm already is. There is no intervention on this task that both removes
the effector and leaves the robot recognisable.

### The instrument itself is sound

`full_hold` is 0/30 with base path 0.02 m and total arm path 0.003 rad, so the
negative control is clean and the interventions are physically effective. Mean
realised right-arm excursion is 0.278 rad unclamped versus 0.160-0.174 rad
clamped, and max clamp leak is 0.187 rad, inside the 0.20 bar. The null result is
not a broken clamp.

Canonical 30/30 also reproduces the published SIMPLE Table 7 figure of 10/10/10
exactly, which validates the whole client/server/eval path.

## OpenFaucet — `PREREQUISITE_FAIL_CANONICAL`

`simple/G1WholebodyOpenFaucetTeleop-v0`, released `ckpt_40000`, official
`mujoco_isaac` rendering.

Published SIMPLE Table 7 reports Psi0 at **3 / 3 / 4 out of 10** across DR levels,
and the paper states that "fine-grained contact tasks (OpenFaucet) reach 3-4/10,
confirming a meaningful difficulty gradient". That is far below the frozen
`canonical_min_success = 0.70`, so gate 1 is unreachable with this checkpoint.

Our own measurement (see table below) agrees.

Measured here, `canonical` only, 30 configs (dr-level-0/1/2 x 10), official
`mujoco_isaac` rendering, released `ckpt_40000`:

| level | canonical success |
| --- | ---: |
| dr-level-0 | 5/10 |
| dr-level-1 | 2/10 |
| dr-level-2 | 4/10 |
| **all** | **11/30 = 0.367** |

Published SIMPLE Table 7 for the same task and checkpoint is 3/3/4 = 10/30. Our
11/30 reproduces it. Against the frozen bar:

| gate | value | bar | result |
| --- | ---: | ---: | --- |
| matched configs | 30 | >= 20 | pass |
| canonical success | **0.367** | >= 0.70 | **FAIL** |

The failure is not an approach failure. The right palm closes to 0.165 m of the
handle on successes and 0.199 m on failures, and `right_hand` is the route
attribution on 25 of 30 rollouts. Mean right-arm excursion is 1.067 rad and is
essentially identical on successes (1.049) and failures (1.078). The policy
reaches the handle and moves its arm; it fails to turn the valve far enough, which
is exactly the "fine-grained contact" difficulty the benchmark paper describes.

The frozen stop rule forbids lowering the gate, selecting configs, or looking for
a different task after seeing this. OpenFaucet stops here.

### Diagnostic observations, explicitly not evidence

The following were collected **before** the published 3/3/4 figure was known, i.e.
before gate 1 was known to be unreachable. Under the frozen gate order they sit
downstream of a failed gate. They are recorded because they explain *why* the two
task failures are complementary, and they must not be read as support for the
hypothesis.

On one config, mujoco rendering:

| condition | success | right-arm excursion | contact with the handle |
| --- | --- | ---: | --- |
| canonical | yes, 304 steps | 0.502 rad | `right_hand` |
| right_frozen | no, 1000 steps | 0.033 rad | none at all |
| right_disabled | no, 1000 steps | 0.032 rad | none at all |

So OpenFaucet does contain a real right-arm motor program — excursion 0.50 rad
versus CloseDoor's 0.27, and freezing it removes all contact with the handle. It
is the task CloseDoor is not. The policy simply is not competent enough on it for
the removal experiment to mean anything.

### Feasibility oracle: partial, and not a passed gate

The oracle is gate 6. Gate 1 failed, so under the frozen order the oracle gate was
never reached and the oracle was not developed to the ≥ 0.70 standard. What exists
is recorded honestly:

A first implementation of `topic23_oracle.LeftArmOracle` **did** solve the task
with the left hand under exactly the `right_disabled` clamp on one config:
`success=True`, route attribution `['left_hand']`, faucet driven to `-0.806` rad
past the `|q| > 0.7` predicate, left palm closing to 0.197 m. So a left-hand
solution to OpenFaucet under the right-side sling demonstrably exists.

That run is weak evidence, because the robot was thrashing: left-arm path length
60-102 rad against roughly 1.2 for the policy, and the base drifted 1.2-2.0 m
despite a zero navigate command. Retuned for stability (base path 0.01 m) the IK
no longer reaches the handle, stalling at 0.28-0.37 m. Neither variant meets the
frozen oracle bar, and no further tuning was done, because the design says not to.

## What this does and does not establish

Established:

1. On CloseDoor, Psi0's solution contains no right-arm motor program, and the task
   survives removing both arms. Any motor-substitution claim on this task is
   unidentified.
2. The registered four-condition panel would have called that a positive result
   with overwhelming statistical support.
3. On OpenFaucet, the released checkpoint does not clear the competence bar the
   design requires before an intervention can be interpreted.
4. A left-hand solution to OpenFaucet under the right-side sling exists.

Not established, in either direction:

> whether robot foundation policies learn motor equivalence classes.

The instrument could not be brought to bear. A future attempt needs a task where
the policy is *both* competent and genuinely using the limb — that pairing does
not exist in the frozen two-task panel, and picking a third task now would be the
post-hoc task shopping the stop rules exist to prevent.

## Reproduction

```bash
pytest -q tests/test_g0_core.py                       # 15 tests
python summarize.py records/door_mj.jsonl             # every table above
python g0_core.py records/door_mj.jsonl --out g0_result.json
```

Frozen upstream: SIMPLE `b49c1aea2dd57309bb533219d0d34d6020f3d943`,
Psi0 `9ad917526394c1cacc72dba08562629936505987`, released `ckpt_40000` per task.
