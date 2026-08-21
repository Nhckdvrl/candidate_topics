# Validation Contract — Uncontrollability Generalization

This file freezes the first serious validation of Topic 06. The purpose is to decide quickly whether the natural phenomenon exists in LLM agents, not to optimize a benchmark score.

## 1. Scientific estimand

For each base seed we create four histories:

- `C1`: controllable + concentrated semantics;
- `C10`: controllable + distributed semantics;
- `U1`: uncontrollable + concentrated semantics;
- `U10`: uncontrollable + distributed semantics.

Let `Y=1` if the model chooses `A`, `B`, or `C` rather than `WAIT` on the **first action of the held-out controllable test**.

Primary cell means:

```text
C1  = E[Y | controllable, concentrated]
C10 = E[Y | controllable, distributed]
U1  = E[Y | uncontrollable, concentrated]
U10 = E[Y | uncontrollable, distributed]
```

Primary interaction:

```text
Delta_diversity = (U10 - U1) - (C10 - C1)
```

The proposed higher-order-generalization result is `Delta_diversity < 0`.

We also report pooled transfer:

```text
Delta_transfer = mean(U1,U10) - mean(C1,C10)
```

with the predicted sign `< 0`.

## 2. Frozen task contract

Unless a plumbing bug makes the task impossible to execute, confirmation must keep these values fixed:

```text
training episodes:       10
training steps/episode:  10
training experiences:    100
active intervention budget/episode: 6
test steps:              8
actions:                 A, B, C, WAIT
active effects (C):      per-episode permutation of -1,0,+1
WAIT effect:             exogenous sequence
uncontrollable effects:  exogenous / exact yoked replay
primary test outcome:    test step-1 active action
```

The goal in every family is to keep an integer reading near zero.

## 3. Identification controls built into generation

These are not optional post-hoc covariates. They define the experimental contrast.

### 3.1 Same amount of experience

All cells receive exactly 10 episode boundaries and 100 training steps. `concentrated` is **not** one 100-step episode; that would confound semantic diversity with reset/context-switch frequency.

### 3.2 Same latent environments across diversity

For a fixed `base_seed`, C1/U1 and C10/U10 use the same latent `EpisodePlan` sequence: identical starting states, hidden A/B/C mappings, and exogenous random schedules.

Thus the planned difference between the `1` and `10` arms is semantic family identity, not causal-structure sampling.

### 3.3 Exact C/U outcome yoking

For each `(base_seed, diversity)` pair, C is run first. The actual per-step effects generated during C training are then replayed to U independent of U's actions.

This follows the logic of classic triadic/yoked controllability experiments: exposure is matched while response-outcome contingency differs.

Important nuance: because two LLMs can choose different actions, their *uttered actions* differ by design. The yoke matches environmental effects and resulting state trajectory, not action strings.

### 3.4 Held-out semantic test

Training and test family sets are disjoint. `base_seed % 4` balances the four held-out test wrappers when the number of base seeds is divisible by four.

### 3.5 First test action before feedback

The primary response is recorded immediately after the novel test introduction. No test outcome has yet occurred, avoiding adaptation-within-test as an explanation for the primary effect.

## 4. G-1: deterministic environment audit

Run:

```bash
pip install -r requirements.txt
./run_preflight.sh
```

`src.audit_environment` simulates a random active policy and checks two structural properties:

1. **matched outcome marginals:** Jensen-Shannon divergence of effects between controllable and uncontrollable generators `< 0.005` bits;
2. **contingency separation:** mean per-episode action-effect mutual information `> 0.5` bits in C and `< 0.08` bits in U.

Unit tests additionally verify exact-yoke state replay, budget behavior, seed reproducibility, semantic family disjointness, parser behavior, interaction arithmetic, and an async end-to-end subject smoke test.

Before registration, the local audit returned approximately:

```text
effect JS divergence:            0.0000023 bits
controllable action-effect MI:   1.559 bits
uncontrollable action-effect MI: 0.052 bits
pytest:                          14 passed
```

Exact small numerical values can vary with audit seed/count; only the declared thresholds matter.

## 5. G0: cheap model pilot

### Run

```bash
export MODEL=<openai-compatible-served-model>
export OPENAI_BASE_URL=http://localhost:8000/v1
export N_SEEDS=40
./run_pilot.sh
```

This yields 160 histories: 40 base seeds x four cells.

Use one model only for the first kill test. Do not model-shop after seeing the interaction.

### G0-A — action I/O validity

Hard plumbing stop if:

```text
mean unparsable response rate > 2%
```

The runner records both the raw response and parsed action. An unparsable action executes `WAIT` but remains flagged invalid.

Budget-forced WAIT is separately recorded and is not a formatting failure.

### G0-B — local controllability sensitivity

This is a premise gate, not the paper claim.

For the last three training episodes, take only the **first action of each episode** and compare pooled C versus U.

Reason for using first actions: within-episode action rates are partly determined by current state, success, and remaining budget. The first action of a reset episode better reflects what the accumulated training history makes the learner expect before seeing new local feedback.

Continue only if:

```text
P(active | C) - P(active | U) >= 0.03
```

in the pilot directionally. Confirmation should report the CI rather than treat 3 points as a significance threshold.

If this fails strongly, the chosen model/task does not instantiate a behavioral sensitivity to controllability. Stop rather than adding probes or training machinery.

### G0-C — cross-task transfer

Primary transfer screen:

```text
Delta_transfer = pooled U - pooled C
```

on test step 1.

Predicted sign: negative.

The pilot analyzer returns `TRANSFER_KILL` when there is little evidence of even a modest negative transfer. This is a practical candidate-selection rule, not a claim that the true psychological effect is mathematically zero.

If there is no transfer, the diversity hypothesis has no useful substrate for an AI paper.

### G0-D — diversity interaction

Only after local sensitivity and transfer are alive, inspect:

```text
Delta_diversity = (U10-U1) - (C10-C1)
```

A large negative pilot interaction justifies locked confirmation. Do not tune family lists, test difficulty, budget, or outcome metric to improve it.

## 6. G1: locked confirmation

Run on a fresh model sampling run:

```bash
./run_full_confirmation.sh
```

Default:

```text
250 base seeds x 4 cells = 1,000 histories
```

Keep all environment/task parameters and the analysis unchanged.

### Primary inference

Report:

- four cell means of `test_step1_active`;
- `Delta_transfer`;
- `Delta_diversity`;
- 95% cluster bootstrap CIs resampling `base_seed`;
- cluster-robust logistic regression `Y ~ U + D + U:D` as a model-based robustness check.

The bootstrap is primary because the four arms sharing a base seed share latent plans and test family assignment.

### Candidate continuation rule

For this repository, the effect must be not only nonzero but useful enough to justify a paper-scale program.

Strong support:

```text
Delta_diversity <= -0.05
AND 95% bootstrap upper bound < 0
```

Negligible diversity effect / kill this formulation:

```text
95% bootstrap CI entirely inside [-0.02, +0.02]
```

Reliable wrong direction:

```text
Delta_diversity >= +0.05
AND 95% bootstrap lower bound > 0
```

Everything else is inconclusive. One larger frozen replication is allowed if the interval is genuinely too wide; changing the task or metric after seeing the result requires a new registration.

## 7. Secondary outcomes

These cannot rescue a failed primary interaction:

- `test_first3_active`;
- `test_active_rate`;
- `test_time_to_first_active`;
- `test_mean_abs_state`;
- `test_active_improvement_rate`;
- recovery trajectory after test feedback;
- full late-training active rate.

A particularly interesting positive-follow-up measure is recovery speed: if U10 begins more passive, does objective evidence of control take longer to reverse it? This is secondary because it contains new test evidence and therefore no longer isolates the initial prior.

## 8. Family-level robustness

The four held-out test families are assigned deterministically and balanced over base seeds. If the primary effect is positive, report it by test family as a heterogeneity check.

Do not require every family to be individually significant. Kill/rethink only if the aggregate result is driven by one obviously anomalous semantic wrapper or reverses across most families.

If semantic similarity appears to govern transfer, register a separate Experiment 2 manipulating train-test semantic distance. Do not fold that search into G0.

## 9. Model replication policy

The first model is a phenomenon screen. If G1 supports the effect:

1. freeze the full protocol;
2. replicate on at least one genuinely different model family;
3. report heterogeneity instead of selecting only positive models.

If the first model is negative at the local-sensitivity premise, a single second model can be used to check whether the task is model-specific. Repeated model shopping is not allowed as a rescue strategy.

## 10. Context-history versus durable learning

The primary experiment uses raw multi-turn interaction history. Therefore its strongest justified claim is about **history-conditioned policy generalization**.

Do not write:

```text
"the model learned a persistent worldview"
```

unless the effect is reproduced when the history is no longer in context and has been retained through a persistent memory or parameter update.

If the behavioral effect survives G1, keep the causal task identical and compare retention mechanisms rather than redesigning the phenomenon:

```text
raw context -> persistent memory -> online parameter adaptation
```

This turns the reviewer objection into an explicit second-stage question: which learning substrate can acquire and preserve an over-generalized controllability prior?

## 11. Stop rules inherited from repository failures

Topic 06 must not repeat the failure modes of Topics 01/02/04/05.

- No layer/threshold/model fishing after a weak result.
- No hidden-state probe is needed to make the construct exist.
- No matching procedure is needed; causal matching is built into generation.
- No hint/prefix intervention changes the test task.
- No secondary endpoint can rescue a failed first-action interaction.
- If the primary observation does not distinguish the scientific explanations, stop and revise the design before collecting more data.
