# 23 — Do Robot Foundation Policies Learn Motor Equivalence Classes?

> **ARCHIVED / ORIGINAL PANEL FALSE-POSITIVE + FROZEN PANEL PREREQUISITE FAILURE (2026-08-24).**
>
> Read [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) and [`G0_RESULTS.md`](./G0_RESULTS.md) first.
>
> The broader scientific question remains unresolved. Topic 23 stops because the original registered panel was shown to be non-identifying on CloseDoor, and the corrected frozen two-task panel then failed prerequisites on both CloseDoor and OpenFaucet.

## Natural question

> **Do robot foundation policies learn the task constraint, or the particular motor realization chosen by the demonstrator?**

If a task admits multiple goal-equivalent body solutions, does a foundation policy preserve the task effect when the canonical motor route is removed, or does it remain tied to the demonstrated realization?

This remains a natural question. Topic 23 is archived because Psi0 + SIMPLE did not provide a valid experimental object for answering it under the frozen panel.

## Final G0 outcome

The corrected panel produced complementary prerequisite failures:

```text
CloseDoor
  policy competent: 30/30
  but canonical arm program absent
  -> PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM

OpenFaucet
  real right-arm program exists
  but canonical success: 11/30 = 0.367
  -> PREREQUISITE_FAIL_CANONICAL
```

No frozen task simultaneously satisfied:

```text
policy competence
AND
causal dependence on the motor program to be removed
```

The question was therefore not tested in either direction.

## The most important result: the original design creates a strong false positive

The first registered four-condition design compared:

```text
canonical
oracle_right_disabled
right_disabled
full_hold
```

On CloseDoor, the observed policy conditions were:

| quantity | value |
| --- | ---: |
| canonical | 30/30 |
| right_disabled | 29/30 |
| full_hold | 0/30 |
| paired `right_disabled - full_hold` | 0.967, 95% CI [0.90, 1.00] |
| substitution events | 29 |

Every original statistical gate passed overwhelmingly.

But the interpretation was wrong.

Psi0 closes the door by walking into it with the right hand already hanging at its side. Shoulder and elbow motion are tiny; the hand is effectively a passive bumper carried by locomotion. There is no right-arm motor program to remove, so the 29 apparent substitution events contain no demonstrated motor substitution.

This establishes a reusable identification rule:

> **Before claiming substitution, first verify that the intervention actually removes a causal motor program used by the canonical behavior. Nominally disabling an actuator group is not enough.**

The bootstrap CI becoming tighter does not repair this construct failure.

## Revision 2 panel

The panel was revised before interpreting the final G0 to include:

1. `canonical`
2. `right_frozen` — removes right-arm articulation while preserving the limb pose
3. `right_disabled` — retracts and clamps the right side
4. `left_disabled` — laterality control
5. `both_arms_disabled` — tests body/base-only solutions
6. `full_hold` — negative-control hold
7. `oracle_right_disabled` — alternative-solution feasibility

The key new prerequisite is behavioral:

```text
canonical - right_frozen >= 0.20
```

If freezing the arm's articulation hardly affects task success, there is no identified arm program to substitute.

A second prerequisite requires:

```text
both_arms_disabled <= 0.10
```

because success after removing both arms is not evidence that one arm substituted for the other.

## CloseDoor

Thirty matched configurations across three DR levels:

| condition | success |
| --- | ---: |
| canonical | 30/30 |
| right_frozen | 29/30 |
| right_disabled | 29/30 |
| left_disabled | 28/30 |
| both_arms_disabled | 30/30 |
| full_hold | 0/30 |

Frozen gate quantities:

```text
canonical - right_frozen = 0.033   < 0.20  FAIL
both_arms_disabled        = 1.000   > 0.10  FAIL
max clamp leak            = 0.187   <=0.20  pass
```

Final verdict:

`PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM`

The null is not explained by a broken harness. `full_hold` is 0/30, the clamp remains within its bar, and canonical 30/30 reproduces the published SIMPLE result exactly.

## OpenFaucet

Released Psi0 `ckpt_40000`, official `mujoco_isaac`, 30 configs:

```text
level 0: 5/10
level 1: 2/10
level 2: 4/10
all:     11/30 = 0.367
```

The published result is 3/3/4 = 10/30, so our measurement reproduces the same low-competence regime.

Frozen prerequisite:

```text
canonical >= 0.70
```

fails immediately.

Final verdict:

`PREREQUISITE_FAIL_CANONICAL`

Diagnostic observations show that OpenFaucet does contain a real right-arm motor program: the arm moves substantially, reaches the handle, and freezing it removes contact. The problem is that the checkpoint succeeds too rarely for constrained failure to identify route binding.

## Oracle status

The oracle is downstream of the failed prerequisites and therefore was not tuned to completion.

One early left-arm oracle run did open the faucet under the right-side clamp (`q=-0.806`, left-hand route), demonstrating that a left-hand solution exists. That controller was unstable; the stabilized version could not reach the handle. The frozen `>=0.70` oracle bar was never met.

This is preserved as feasibility evidence, not a passed gate.

## Relation to Topic 19

Topic 19 failed because a joint-space projection did not identify task-space correction in a redundant arm.

Topic 23 correctly moved the dependent variable into outcome space, but uncovered a different identification failure:

```text
outcome success after nominal route removal
!=
proof that a causal motor program was removed and substituted
```

For whole-body systems, locomotion, torso motion, passive contact, or object dynamics can preserve the task without any cross-effector substitution.

The two archived topics therefore give complementary lessons:

- Topic 19: define correction in task/outcome space, not a joint-axis proxy.
- Topic 23: verify causal program removal before interpreting preserved outcome as substitution.

## Frozen upstream contracts

- SIMPLE: `b49c1aea2dd57309bb533219d0d34d6020f3d943`
- Psi0: `9ad917526394c1cacc72dba08562629936505987`
- released checkpoint: `ckpt_40000`

## Evidence and reproduction

- [`G0_RESULTS.md`](./G0_RESULTS.md) — complete results and interpretation.
- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — final archive decision and reusable lessons.
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md) — identification and implementation audit.
- `records/*.jsonl` — raw committed panel evidence.
- [`summarize.py`](./summarize.py) — regenerates the result tables.
- [`g0_core.py`](./g0_core.py) — frozen gate logic.
- [`topic23_runner.py`](./topic23_runner.py) — final intervention runner.
- [`topic23_oracle.py`](./topic23_oracle.py) — oracle experiments.
- `tests/test_g0_core.py` — 15 logic tests in the final G0 commit.

```bash
pytest -q tests/test_g0_core.py
python summarize.py records/door_mj.jsonl
python g0_core.py records/door_mj.jsonl --out g0_result.json
```

## Final stop rule

Do not rescue Topic 23 by:

- shopping for a third task after the frozen two-task panel failed;
- lowering the OpenFaucet competence threshold;
- changing the CloseDoor causal-dependence gate;
- searching checkpoints until a convenient regime appears;
- adding hidden-state analysis to manufacture a mechanism result.

A future project could revisit the broad motor-equivalence question only if a genuinely new external observation supplies a task/model pair that is independently known to be both competent and causally limb-dependent. That would be a new registered topic, not a continuation of Topic 23.