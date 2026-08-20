# Topic 04 measurement repair record

## Decision after G-1v1

**Verdict: measurement failure; one repair allowed.**

G-1v1 correctly stopped before G0 because only 61 matched pairs survived the preregistered `<200` stop rule.

The result does **not** test the correction hypothesis. No corrective SFT was run.

## Observed v1 numbers

```text
n_scored_input                    9981
n_eligible_wrong                   716
wrong_concentration low cutoff    0.7947
wrong_concentration high cutoff   0.9358
n_low_pool                         215
n_high_pool                        215
n_pairs                             61
mean |Δ p_correct|                0.00547
mean commitment separation        0.2529
```

The fact that the v1 "low" cutoff was already ~0.795, despite nine wrong options, showed that the stability gate had removed most genuinely diffuse distributions.

## Why this is a measurement defect rather than a hypothesis result

### Defect 1 — treatment-dependent stability gate

G-1v1 required the same semantic top-wrong option in >=8/10 rotations.

But low wrong commitment means multiple wrong options are close. Their top-1 identity is therefore expected to swap under small option-order perturbations.

The gate selected on a consequence of the treatment itself.

### Defect 2 — arithmetic mean confounds semantic uncertainty and position susceptibility

Some raw items were highly confident on almost every permutation but changed *which semantic option* received the confidence when option positions moved.

Arithmetic averaging turned:

```text
sharp but position-sensitive
```

into:

```text
apparently semantically diffuse
```

These are different phenomena.

## G-1v2 repair

1. remove top-wrong stability as an inclusion gate;
2. aggregate balanced permutations in log-probability space;
3. keep position susceptibility as a separate JS-divergence diagnostic;
4. independently audit a second balanced permutation family and alternate prompt;
5. save full-vocabulary answer-label mass and greedy answer-channel compliance;
6. use the same repaired scorer in any eventual G0 checkpoint evaluation.

## Why mean log-probability

For an additive nuisance model

\[
z_{r,j} = \alpha_j+\beta_{position(r,j)}
\]

and a complete balanced permutation set, every semantic option sees every position exactly once.

Then averaging log probabilities across rotations leaves semantic score `alpha_j` plus constants shared across all choices. Renormalizing removes those constants.

A deterministic unit test verifies exact recovery in the synthetic additive-bias case.

## Hard stop after v2

This is the only allowed repair.

Kill Topic 04 if:

- `<200` matched pairs after offline v2 reaggregation;
- balanced-family reliability fails;
- prompt reliability fails;
- response-channel diagnostics show the choice probabilities are not a meaningful answer channel;
- 1.5B/one predeclared 3B replication both fail.

No additional model/metric/dataset rescue.
