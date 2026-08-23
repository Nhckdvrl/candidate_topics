# 2026-08-24 — Topic 23 G0 stop: motor-equivalence panel was non-identifying

This note records the disposition of provisional candidate D after promotion to root Topic 23 and execution of its frozen Psi0 + SIMPLE G0.

## Final decision

**Archive Topic 23. Do not search for a third task.**

The broad question remains natural:

> Do robot foundation policies learn task effects independently of the demonstrator's particular motor realization?

But the frozen two-task panel did not contain a task where the released policy was simultaneously:

1. competent enough for a constrained failure to be interpretable; and
2. causally using an arm motor program that could be removed.

The two tasks failed complementary prerequisites:

```text
CloseDoor
  canonical = 30/30
  right_frozen = 29/30
  both_arms_disabled = 30/30
  -> PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM

OpenFaucet
  canonical = 11/30 measured
  published = 10/30
  frozen minimum = 0.70
  -> PREREQUISITE_FAIL_CANONICAL
```

## The important failure of the original design

The originally registered four-condition panel would have called CloseDoor a spectacular positive:

```text
canonical                  30/30
right_disabled             29/30
full_hold                   0/30
paired difference           0.967
95% CI                      [0.90, 1.00]
substitution events         29
```

Yet the interpretation is false. The robot walks into the door while the right hand hangs at its side; freezing the arm costs almost nothing, and removing both arms costs nothing. No meaningful right-arm motor program was removed, so no motor program was substituted.

This gives a reusable selection lesson stronger than “add another control”:

> **A statistically overwhelming intervention result can still be structurally non-identifying when treatment fails to remove the causal object named by the scientific claim.**

For future embodied mechanism audits, establish both prerequisites on the same task *before* interpreting the main intervention:

```text
reliable task competence
AND
causal use of the route/mechanism being removed
```

A clean negative control (`full_hold = 0`) only rules out accidental environment success; it does not prove the treatment removed the intended mechanism.

## Why no third task

After observing that CloseDoor is competent-but-route-free and OpenFaucet is route-bearing-but-incompetent, selecting a third SIMPLE task would amount to post-hoc task shopping for the desired intersection. That violates the frozen-panel stop rule.

If an independently motivated future model/task supplies the missing intersection, it should be registered as a fresh experiment rather than used to rescue Topic 23.

See root archive:

- [`../23_motor_equivalence_classes/G0_RESULTS.md`](../23_motor_equivalence_classes/G0_RESULTS.md)
- [`../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md`](../23_motor_equivalence_classes/ARCHIVE_SUMMARY.md)
