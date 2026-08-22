# Validation contract — v2

## Scientific object

The known phenomenon is:

```text
frozen VLA representation -> eventual success
```

The unresolved question is:

```text
is that signal mainly generic state difficulty D(s),
or does it contain policy-specific competence C(pi, s)?
```

The first experiment must identify that distinction without creating hard states, searching layers, or introducing a new confidence method.

## P0 — technical identity gate

Before any scientific number is interpreted, the released OpenPI / LIBERO stack must pass `src/preflight.py`.

The protocol follows the official OpenPI LIBERO evaluator: `set_init_state`, ten dummy settling actions, 224x224 preprocessed agent/wrist images, five executed actions per replan, and the official LIBERO success termination.

The important extra check is physical-state identity. `task_id + init_idx` is not accepted on faith. After settling, hash the flattened MuJoCo simulator state. The hash must be identical across checkpoint processes and repeats. A mismatch is a technical failure, not a scientific sample.

Policy stochasticity is also explicit. Every inference receives a deterministic Gaussian noise seed. Reusing a base `policy_seed` creates the same sequence of inference-noise seeds for every checkpoint on the same physical state (common random numbers).

Preflight requires:

1. reset + settle reproduces the same simulator-state hash;
2. same observation + same noise seed gives bit-identical actions;
3. same observation + same noise seed gives bit-identical captured feature;
4. changing the noise seed changes the sampled action;
5. layer-11 capture yields exactly ten denoising-step activations.

If these fail: `TECHNICAL_BLOCKED`.

## G0 — does natural bidirectional competence crossover exist?

### States

Use LIBERO-10 with the official fixed initial states.

Frozen split in `LOCKED_CONFIG.json`:

```text
discovery:    init_idx 0..14 for every task  -> 150 physical states
confirmation: init_idx 15..29 for every task -> 150 physical states
reserve:      init_idx 30..49
```

`env_seed=7`, `wait_steps=10`, `replan_steps=5` are fixed.

### Policies

Released same-family pi0.5 LIBERO checkpoints:

```text
2k, 3k, 9k
```

All three are evaluated on exactly the same discovery states.

### Why one rollout is invalid

pi0.5 is a stochastic flow policy. A single success/failure draw does not define competence at a state and can manufacture apparent checkpoint crossover.

Therefore every `(physical_state, checkpoint)` receives eight rollouts with the same frozen set of base policy seeds across checkpoints. At every replan the actual Gaussian-noise seed is deterministically derived from:

```text
policy_seed, suite, task_id, init_idx, replan_idx
```

This keeps each rollout on-policy while pairing stochastic draws across checkpoints.

### State-level winner

For each checkpoint estimate:

```text
p_hat(pi, s) = successes / 8
```

For pair A/B:

```text
A wins robustly if p_A - p_B >= 0.50
B wins robustly if p_B - p_A >= 0.50
otherwise the state is ambiguous
```

This is deliberately a large effect threshold, not a significance test.

### G0 stop rule

Select the checkpoint pair only by natural **bidirectional robust support**:

```text
min(# robust A-wins, # robust B-wins)
```

Require at least 15 robust states in each direction on discovery.

If no pair reaches this:

```text
STOP_NO_NATURAL_CROSSOVER
```

Do not create perturbations, train special checkpoints, lower the rate-gap after seeing data, or replace success with a hand-designed difficulty score.

If one pair passes, freeze that pair before hidden-state work.

## G1 — paired policy-specific success signal

### Feature location

Primary feature is fixed before collection:

```text
pi0.5 action-expert decoder layer 11 output
```

OpenPI's PyTorch inference calls the action expert once per denoising step. `src/openpi_instrumented_server.py` attaches an observational forward hook to layer 11; it never edits activations or model outputs.

For one policy-noise draw:

1. mean layer-11 residual output over action tokens;
2. mean those vectors over the ten denoising steps.

The representation itself is stochastic because the action expert conditions on the current noisy action. Therefore each `(state, checkpoint)` feature is the mean of four **common feature-noise seeds** shared by the two checkpoints. The feature seed sets must match exactly or analysis aborts.

### Shared decoder

Fit one decoder across both frozen checkpoints on discovery states:

```text
q = w^T standardized(h) + b
```

Implementation: one fixed ridge-linear readout (`alpha=1.0`). Target is the eight-rollout Monte-Carlo success rate `p_hat(pi,s)`.

There is only one scaler and one `(w,b)` for both checkpoints. Separate probes are forbidden in the primary test because they make cross-checkpoint scores incomparable.

### Why the paired contrast identifies the intended object

For the same physical state:

```text
relative_score(s) = q_A(s) - q_B(s)
```

A pure state-only signal contributes equally and cancels. A constant checkpoint-quality prior produces a constant offset and cannot rank **bidirectional** crossover states. To succeed, the decoded representation must change with the checkpoint-state interaction in a way that tracks which policy is competent there.

This supports the operational claim **policy-specific success signal**. It is stronger than generic scene difficulty, but it should not be described as a literal explicit self-model without further evidence.

### Independent confirmation

Use confirmation physical states only (`init_idx 15..29`), with new behavior policy seeds and new feature-noise seeds from `LOCKED_CONFIG.json`.

Confirmation must first reproduce at least 15 robust A-wins and 15 robust B-wins. If not:

```text
KILL_CROSSOVER_NOT_REPLICATED
```

Primary metric on robust crossover states:

```text
AUROC(q_A - q_B, A-is-winner)
```

Bootstrap whole physical states, keeping both checkpoint rows paired.

Continue only if:

```text
relative AUROC >= 0.70
and bootstrap 95% lower bound > 0.60
```

Then:

```text
PASS_POLICY_SPECIFIC_SUCCESS_SIGNAL
```

Otherwise:

```text
KILL_SELF_KNOWLEDGE_INTERPRETATION
```

Balanced accuracy of `sign(q_A-q_B)` is secondary only.

## Interpretation of a negative G1

A negative shared-readout result does **not** prove that no nonlinear or representation-aligned notion of self-knowledge exists. Different checkpoints can drift in representation space.

For this project that caveat does not trigger a rescue campaign. If the clean shared linear contrast fails, stop. Do not add Procrustes alignment, nonlinear probes, layer sweeps, SAE searches, failure taxonomies, or hand-selected subsets.

## Anti-complexity rule

The topic gets one natural behavioral contrast and one paired representation test. If the story requires more machinery to become true, the research question is demoted rather than the gate expanded.
