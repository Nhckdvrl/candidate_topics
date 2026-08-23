# Do Robot Foundation Policies Learn Motor Equivalence Classes?

> **PROMOTED TO TOPIC 23, THEN ARCHIVED (2026-08-24).**
>
> Formal project: [`../../23_motor_equivalence_classes/`](../../23_motor_equivalence_classes/)
>
> Final decision: **ARCHIVED / ORIGINAL PANEL FALSE-POSITIVE + FROZEN PANEL PREREQUISITE FAILURE.**

## Original question

> **Do robot foundation policies learn the task constraint, or the particular motor realization chosen by the demonstrator?**

The idea was promoted from Round 9 because the question was natural and Psi0 + SIMPLE appeared to provide a clean same-robot / same-task / same-world counterfactual: remove a canonical motor route and test whether the policy preserves the task effect through another body solution.

## What happened after promotion

The first registered four-condition design looked spectacular on CloseDoor:

```text
canonical       30/30
right_disabled  29/30
full_hold        0/30
paired diff      0.967, 95% CI [0.90, 1.00]
substitution events 29
```

But this was a false positive. The robot closes the door mainly by locomotion, carrying a hand that already hangs at its side into the door. Freezing the right arm barely changes performance and removing both arms does not reduce success.

So the original intervention had **not removed a causal right-arm motor program**. The result therefore could not identify motor substitution, no matter how strong the CI looked.

A revised frozen panel added `right_frozen` and `both_arms_disabled` prerequisites.

### CloseDoor

```text
canonical - right_frozen = 0.033  < 0.20
both_arms_disabled       = 1.000  > 0.10
```

Verdict: `PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM`.

### OpenFaucet

OpenFaucet does contain a real arm program, but released Psi0 canonical success is only `11/30 = 0.367`, reproducing the published `10/30` regime and failing the frozen `>=0.70` competence prerequisite.

Verdict: `PREREQUISITE_FAIL_CANONICAL`.

## Final interpretation

The broad scientific question remains unresolved.

The frozen Psi0 + SIMPLE panel contained no task that was simultaneously:

```text
policy competent
AND
causally dependent on the motor program to be removed
```

Searching for a third task after seeing the failures would be post-hoc task shopping, so Topic 23 was archived.

## Most reusable lesson

> **Before interpreting constrained task success as motor substitution, first prove that the intervention removes a causal motor program used by the canonical behavior. Nominally disabling an actuator group is not enough.**

Whole-body robots make this especially important because locomotion, torso motion, passive contact, and object dynamics can preserve task success without any cross-effector substitution.

For the full results, code, raw JSONL, and archive rationale, see:

- [`../../23_motor_equivalence_classes/G0_RESULTS.md`](../../23_motor_equivalence_classes/G0_RESULTS.md)
- [`../../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md`](../../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md)
- [`../../23_motor_equivalence_classes/VALIDATION_AUDIT.md`](../../23_motor_equivalence_classes/VALIDATION_AUDIT.md)
