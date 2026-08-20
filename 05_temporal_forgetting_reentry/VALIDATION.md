# Validation contract — Topic 05

## Primary scientific question

Does robust temporal forgetting primarily reflect loss of **self-access to a previously successful reasoning route**, or deeper loss of the route itself?

The first experiment is behavioral and likelihood-based. No hidden-state probe is part of G0.

## G-1: robust state construction

Input: one JSON object per sampled completion with `problem_id`, `checkpoint`, `checkpoint_order`, `prompt`, `response`, `correct`, `gold_answer`.

Frozen thresholds with >=8 samples/checkpoint:

```text
robust_correct >= 0.75
robust_wrong   <= 0.125
```

Groups:

- **F forgotten**: final <= .125; at least one earlier checkpoint >= .75; select the latest qualifying old checkpoint.
- **N never-correct**: every checkpoint <= .125.
- **S stable-correct**: final >= .75 and at least one earlier checkpoint >= .75.

### G-1 pass condition

For MATH-500: `F>=50`, `N>=50`, `S>=50`.

The F threshold is the key feasibility gate. If F<50, do not loosen thresholds after seeing results merely to manufacture support. Either expand to OlympiadBench under the same thresholds or stop.

## G0-A: re-entry

For each F item, use the latest robust-correct checkpoint and freeze one deterministic old correct trace before interventions. A reasonable primary rule is shortest valid correct trace.

For N, obtain an externally correct trace from a fixed teacher or verified solution source.

Primary prefix fractions: `0.00, 0.10, 0.25, 0.50`.

Reject prefixes containing boxed final answer, explicit final-answer phrase with normalized gold answer, or >60% of the full trace.

### Conditions

```text
F_oldself   forgotten + its own old correct prefix
F_other     forgotten + another verified correct prefix
F_wrong     forgotten + final model's own wrong prefix
N_correct   never-correct + verified correct prefix
S_oldself   stable-correct + its own old prefix
```

`F_other` must be matched in reasoning-step count and approximate token count.

### Sampling

Primary: final checkpoint only, 8 rollouts/problem/condition/prefix level, temperature 0.6, top_p 0.95.

Also run deterministic decoding as a robustness check, but do not select whichever looks better as primary.

### Primary estimands

`R_c(k) = mean correctness`.

Route-specific contrast: `Delta_self(k) = R_F_oldself(k) - R_F_other(k)`.

Retention-vs-novelty contrast: compare F old-self rescue against N correct-prefix rescue after matching difficulty/information.

Cluster-bootstrap by problem ID.

## G0-B: teacher-forced old-route likelihood

For F and S, evaluate final-model mean per-token NLL on frozen old traces as a function of prefix depth. For N use matched verified correct traces.

Important:

- per-token NLL, not total loss;
- identical prompt/chat template;
- exclude answer-only suffixes;
- report the curve rather than a cherry-picked point.

A useful pattern is: free generation fails but old-route suffix NLL remains substantially better than N controls.

## G1: relearning savings

Secondary triangulation, preregistered before G0 results.

Starting from identical final checkpoints, expose separate clones/adapters to equal-size F and matched N solution sets for a very small fixed number of updates.

Measure exposures needed to recover a fixed success criterion and normalized solution NLL.

Do not use G1 to rescue a failed G0 by changing item definitions or traces.

## Positive controls

1. `S_oldself` should be easy to continue.
2. Correct prefix depth should increase answer information on average; if no condition benefits at all, prefix construction is suspect.
3. Answer checker should reproduce official checkpoint accuracies approximately.
4. All prefix conditions must pass answer-leakage checks.

## Confirmation split

Recommended MATH-500 split: 60% pipeline/discovery, 40% locked confirmation; then optional OlympiadBench replication.

Primary fractions and contrasts are already fixed; discovery is for implementation debugging, not metric shopping.

## No-rescue rule

After a failed locked confirmation, do not rescue the topic by trying many prefix fractions, selecting a different old checkpoint because it works better, switching to hidden probes, or reporting only easy-to-rescue problems.
