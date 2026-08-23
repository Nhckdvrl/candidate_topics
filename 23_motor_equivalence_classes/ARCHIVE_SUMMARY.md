# Topic 23 Archive Summary — Motor Equivalence Classes

> **FINAL STATUS: ARCHIVED / FROZEN PANEL PREREQUISITES FAIL.**
>
> The broader scientific question remains unresolved. The frozen Psi0 + SIMPLE panel did not contain a task on which the released policy was both (a) competent enough for an intervention to be interpretable and (b) actually using a removable arm motor program.

## Question

> **Do robot foundation policies learn the task constraint, or the particular motor realization used by the demonstrator?**

The intended clean test was to keep the robot, task, world, and language fixed while removing the canonical right-side motor solution, then ask whether the policy preserves the same environmental effect through another body solution.

## Why the topic stops

The two frozen tasks fail complementary prerequisites.

### CloseDoor: competent policy, but no canonical arm program

On 30 matched configs:

- `canonical`: **30/30**;
- `right_frozen`: **29/30**;
- `both_arms_disabled`: **30/30**;
- `full_hold`: **0/30**;
- `canonical - right_frozen = 0.033`, below the frozen `0.20` minimum cost of removing the canonical arm program.

The policy closes the door primarily by locomoting into it with the right hand hanging near its side. Freezing the right arm costs essentially nothing, and removing both arms costs nothing. Therefore there is no arm program whose removal can identify motor substitution.

Final gate:

```text
PREREQUISITE_FAIL_NO_CANONICAL_ARM_PROGRAM
```

### OpenFaucet: real arm program, but insufficient competence

On the same 30-config official `mujoco_isaac` panel:

- measured canonical success: **11/30 = 0.367**;
- published SIMPLE result for the same checkpoint: **10/30**;
- frozen canonical prerequisite: **>= 0.70**.

The policy approaches the handle and uses the right arm, but fails the fine-grained contact/control problem often enough that constrained failures cannot be interpreted as inability to substitute another motor solution.

Final gate:

```text
PREREQUISITE_FAIL_CANONICAL
```

The frozen stop rule therefore forbids lowering the competence bar, selecting easy configs, tuning the oracle, or shopping for a third task after observing the result.

## Most important discovery: the original panel would have produced a strong false positive

The registered four-condition design on CloseDoor gave:

| Original quantity | Result |
| --- | ---: |
| canonical | 30/30 |
| right_disabled | 29/30 |
| full_hold | 0/30 |
| paired `right_disabled - full_hold` | **0.967**, 95% CI **[0.90, 1.00]** |
| substitution events | **29** |

Every frozen positive gate passed overwhelmingly. With the oracle prerequisite added, the old code would have returned `PROMISING_MOTOR_SUBSTITUTION`.

That conclusion would have been wrong: no meaningful right-arm motor program had been removed. The hand was functioning as a passive bumper carried by locomotion.

This is stronger than saying the first design was merely “missing one control.” It demonstrates that a statistically decisive outcome-space contrast can still be **structurally non-identifying** when the intervention does not remove the causal object named in the scientific claim.

## Why revision 2 was legitimate

The revision was not a post-hoc attempt to turn a negative result positive. It corrected concrete mismatches between the registered intervention and the real execution stack:

- the arm clamp was moved to the actuator boundary after the GR00T WBC, which could otherwise re-solve around pre-WBC action edits;
- `full_hold` was connected to the queued base command rather than a no-op downstream location;
- clamp leakage was measured per episode;
- route attribution was moved from a fixed temporal window to the period of peak object motion because the door coasts after contact;
- `decompose()` was removed as evidence about Teleop demonstration laterality;
- `right_frozen` and `both_arms_disabled` were added as prerequisite tests for whether an arm motor program exists at all.

The revised gates then failed cleanly in opposite ways on the two frozen tasks.

## Oracle disposition

The alternative-solution oracle was gate 6 and was never reached after earlier gates failed.

A first left-arm OpenFaucet oracle solved one config under the right-side constraint and drove the faucet to `-0.806 rad`, showing that a left-hand physical solution exists. However that version exhibited unrealistic thrashing and base drift; after stabilizing it, reachability failed. Neither implementation satisfied the frozen `>= 0.70` oracle bar.

No further oracle tuning was performed because doing so cannot repair the failed earlier prerequisites.

## What was established

1. The full Psi0/SIMPLE client-server-evaluation path was reproduced: CloseDoor reached **30/30**, exactly matching the published 10/10/10 split.
2. The intervention machinery is active: `full_hold = 0/30` and clamp leakage remains within the frozen bound.
3. CloseDoor does **not** instantiate the required removable arm program.
4. OpenFaucet does instantiate meaningful right-arm use, but the released checkpoint does **not** satisfy the frozen competence prerequisite.
5. The original four-condition panel can generate an apparently overwhelming false positive for motor substitution.

## What was not established

The experiment does **not** show that robot foundation policies lack motor-equivalence abstractions.

It also does not provide positive evidence that Psi0 possesses them.

The correct scientific conclusion is:

> **On the frozen Psi0 + SIMPLE panel, the motor-equivalence question could not be identified because no task jointly satisfied policy competence and removable canonical-motor-program prerequisites.**

## Reusable lessons

### 1. Verify that the intervention actually removes the causal object in the claim

`right_disabled succeeds` does not imply “the policy substituted away from a right-arm program” unless an arm program was causally necessary in the canonical behavior in the first place.

A route label, contact attribution, or body-part name is not enough. The clean prerequisite is behavioral:

```text
canonical success - same-task success with that motor program frozen
```

must be meaningfully positive before substitution can be studied.

### 2. A perfect negative control does not rescue a non-identifying treatment

`full_hold = 0/30` proved the environment was not succeeding by itself. It did **not** prove that `right_disabled` had removed the motor solution of interest.

### 3. Statistical strength cannot compensate for construct failure

A paired effect of `0.967` with CI `[0.90, 1.00]` and 29/30 events looked stronger than most pilot results in this repository. It was still the wrong scientific conclusion.

Before trusting significance, ask:

> **If this result is maximally strong, is the named causal object definitely present in control and definitely removed in treatment?**

### 4. Require competence and causal usage jointly

For intervention-based mechanism audits, two prerequisites must coexist on the same task:

```text
policy can reliably do the task
AND
policy causally uses the mechanism/route being removed
```

CloseDoor satisfied only the first. OpenFaucet satisfied the second but not the first.

### 5. Do not task-shop after the frozen panel fails

A third task might eventually provide the desired intersection, but selecting it after seeing why the first two fail would turn a preregistered identification test into open-ended benchmark search. A future attempt should be a newly registered topic/panel motivated independently of this outcome.

## Preserved artifacts

- [`G0_RESULTS.md`](./G0_RESULTS.md) — full result tables, diagnostics, and oracle disposition.
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md) — revision rationale and upstream-contract audit.
- `topic23_runner.py` / `topic23_oracle.py` — rollout and oracle implementations.
- `summarize.py` — regenerates reported tables from committed raw records.
- `records/*.jsonl` — raw evidence for the frozen panel.
- `tests/test_g0_core.py` — 15 passing tests at the final G0 revision.

## Stop decision

**Archive Topic 23. Do not search for a third task, lower the OpenFaucet competence gate, tune the oracle, or reinterpret the CloseDoor positive contrast as motor substitution.**

A future motor-equivalence project is only justified by a newly motivated system/task where competence and canonical-limb causal usage are established *before* the substitution intervention is evaluated.
