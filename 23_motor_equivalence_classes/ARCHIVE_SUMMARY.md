# Topic 23 Archive Summary — Motor Equivalence Classes

**Final decision: ARCHIVED / ORIGINAL PANEL FALSE-POSITIVE + FROZEN PANEL PREREQUISITE FAILURE.**

Topic 23 asked a natural question:

> Do robot foundation policies learn the task constraint, or the particular motor realization chosen by the demonstrator?

The question remains unresolved. The project is archived because the registered Psi0 + SIMPLE identification failed twice, in two different ways.

## 1. The original four-condition design is non-identifying

On CloseDoor, the original panel would have produced an apparently decisive positive result:

| quantity | result |
| --- | ---: |
| canonical | 30/30 |
| right_disabled | 29/30 |
| full_hold | 0/30 |
| paired right_disabled − full_hold | 0.967, 95% CI [0.90, 1.00] |
| substitution events | 29 |

Every registered gate would have passed and, with an oracle, the protocol would have reported `PROMISING_MOTOR_SUBSTITUTION`.

That conclusion would be wrong.

Contact and kinematic inspection showed that Psi0 closes the door by walking into it with the right hand already hanging at the side. Shoulder and elbow motion are tiny; the hand is effectively a passive bumper transported by locomotion. Therefore the intervention did not remove a causal right-arm motor program and no substitute program was required.

This is the most important result of the project:

> **Strong task-level statistics do not identify substitution unless the intervention is first shown to remove a motor program that causally matters for the canonical behavior.**

A nominal actuator clamp is not enough.

## 2. Revision 2 catches the false positive, then fails the frozen prerequisites

The revised panel added `right_frozen` and `both_arms_disabled` before evaluating substitution.

### CloseDoor

- canonical: `30/30`
- right_frozen: `29/30`
- canonical − right_frozen: `0.033` vs frozen minimum `0.20`
- both_arms_disabled: `30/30` vs frozen maximum `0.10`
- full_hold: `0/30`
- max clamp leak: `0.187 rad` vs frozen maximum `0.20`

Verdict:

`PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM`

The intervention stack is functioning, but there is no right-arm program to remove on this task.

### OpenFaucet

Official `mujoco_isaac`, released Psi0 `ckpt_40000`, 30 configs:

- level 0: `5/10`
- level 1: `2/10`
- level 2: `4/10`
- total: `11/30 = 0.367`

This reproduces the published `3/3/4 = 10/30` regime and is far below the frozen `canonical >= 0.70` prerequisite.

Verdict:

`PREREQUISITE_FAIL_CANONICAL`

Unlike CloseDoor, OpenFaucet does contain a real right-arm motor program, but the released policy is not competent enough for a route-removal failure to be interpretable.

## 3. Oracle status

The alternative-solution oracle was gate 6 and therefore was never required after earlier gates failed.

One early left-arm oracle run did open the faucet under the right-side clamp (`q=-0.806`, left-hand route), demonstrating that a left-hand solution exists. However, that controller thrashed badly; the stabilized version could not reach the handle. The frozen `>=0.70` oracle bar was never met and was not tuned after the stop condition.

This is recorded as feasibility evidence, not a passed gate.

## 4. What was established

1. The original registered panel can produce an overwhelming statistical false positive for motor substitution.
2. CloseDoor is unsuitable because Psi0's successful behavior does not depend on an articulated arm program.
3. OpenFaucet is unsuitable because canonical policy competence is too low.
4. The intervention/client/server/evaluator stack itself is not the explanation: CloseDoor canonical reproduces 30/30, full-hold is 0/30, and the clamp leak remains within the frozen bar.
5. The broader scientific question — whether foundation policies learn motor-equivalence classes — was not answered in either direction.

## 5. Why the topic stops here

The frozen panel contained CloseDoor and OpenFaucet. Both fail prerequisites for complementary reasons. Searching a third task after observing these failures would be post-hoc task shopping.

No hidden-state analysis, new metric, threshold relaxation, checkpoint search, or task search is warranted under the registered stop rule.

## 6. Reusable lessons

### Causal-program removal before substitution claims

Before calling constrained success `substitution`, verify that the canonical behavior actually depends on the removed program.

A useful preflight is:

```text
canonical success
vs
same-effector articulation frozen
```

If performance barely changes, there is no identified program to substitute.

### Body-only route control

For whole-body robots, removing one limb does not imply another limb took over. Locomotion, torso motion, passive contact, or environmental dynamics can preserve task success. A body-only control can be necessary before interpreting effector substitution.

### Statistical confidence cannot repair construct invalidity

Here the wrong interpretation had 29 events and a bootstrap CI of `[0.90, 1.00]`. More samples would only make the wrong conclusion look stronger.

### Pair competence with causal dependence before the expensive experiment

A candidate task must satisfy both:

```text
policy is competent
AND
canonical behavior genuinely depends on the motor program to be removed
```

Checking these two properties should precede the substitution experiment itself.

## 7. Evidence

- [`G0_RESULTS.md`](./G0_RESULTS.md) — complete results and interpretation.
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md) — panel revision and implementation audit.
- `records/*.jsonl` — committed raw evidence.
- [`summarize.py`](./summarize.py) — regenerates the result tables from the records.
- [`g0_core.py`](./g0_core.py) — frozen gate logic.
- [`topic23_runner.py`](./topic23_runner.py) — final intervention runner.
- [`topic23_oracle.py`](./topic23_oracle.py) — oracle experiments.

No further experiment is planned for Topic 23.