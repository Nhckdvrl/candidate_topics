# Validation contract — v3

> **Amendment record.** v2 -> v3 on 2026-08-22, after a pre-data audit and **before any
> behavior or feature data existed**. Three changes: (1) G0 must now beat an explicit
> sampling-noise null; (2) G1 reports a pre-declared power control that can only downgrade
> a negative to *inconclusive*, never manufacture a pass; (3) the feature decision point is
> stated explicitly. A fourth change was made later the same day, still pre-data: the
> shared ridge penalty is selected by state-grouped CV on discovery rather than pinned at
> `alpha=1.0`. All decision thresholds are unchanged.

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

Inference runs in **eager mode**. OpenPI wraps `sample_actions` in
`torch.compile(mode="max-autotune")` by default; Topic 09 disables it, for measurement
reasons rather than performance ones. The layer-11 capture is a module forward hook
registered immediately before inference — after the function has already been compiled and
cached — and a compiled graph need not dispatch through it, failing as a silent
zero-capture. More importantly, G0 runs without the hook and G1 runs with it; under
compilation those are two different graphs, so the policy whose competence we measured
would not be bit-for-bit the policy whose representation we read. Eager mode is applied
identically to every checkpoint, so it cannot bias the comparison.

Preflight requires:

1. reset + settle reproduces the same simulator-state hash;
2. same observation + same noise seed gives bit-identical actions;
3. same observation + same noise seed gives bit-identical captured feature;
4. changing the noise seed changes the sampled action;
5. layer-11 capture yields exactly ten denoising-step activations.

A separate check covers a failure mode that would otherwise masquerade as a result. Every
downstream conclusion assumes the servers hold *different* checkpoints; if a conversion
wrote the same weights twice, or two servers were pointed at one directory, the pipeline
would run cleanly and return no crossover and a relative score of exactly zero — a perfect
null that means nothing. `src/check_checkpoints_differ.py` gives every checkpoint the same
settled observation and the same policy-noise seed and requires the actions and layer-11
features to differ.

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

### The crossover must beat sampling noise

Requiring `p_A - p_B >= 0.5` over eight rollouts is a large-effect rule, but it is not
noise-free. If two checkpoints are equally competent at a state with true rate `p`, the
probability that noise alone produces a robust "A win" is:

```text
p=0.1 -> 0.002    p=0.3 -> 0.026    p=0.5 -> 0.038
p=0.7 -> 0.026    p=0.9 -> 0.013
```

At the worst case (`p=0.5` everywhere) a 150-state panel would be expected to yield about
**6 spurious A-wins and 6 spurious B-wins with no competence difference anywhere**. The
`min(...) >= 15` rule sits above that floor, but only by roughly four standard deviations,
and the true per-state rates are unknown.

`src/noise_null.py` therefore runs an exact **within-state relabeling test**. For each
physical state the 16 observed rollout outcomes (8 from each checkpoint) are pooled and
randomly re-split into two groups of eight. This holds each state's pooled difficulty
*exactly* fixed and destroys only the outcome/checkpoint association, so any crossover
surviving the permutation is sampling noise by construction.

Reported statistics are the observed and null values of `min(n_a_wins, n_b_wins)` and of
the direction-free `n_a_wins + n_b_wins`. Note the permutation null is symmetric in A/B,
whereas a real global quality gap would *lower* the bidirectional count; the direction-free
count is therefore the cleaner noise diagnostic and both must pass.

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

The selected pair must additionally beat the sampling-noise null at `alpha=0.05` on both
statistics, with observed bidirectional support above the null 95th percentile. Otherwise:

```text
STOP_CROSSOVER_EXPLAINED_BY_SAMPLING_NOISE
```

Do not create perturbations, train special checkpoints, lower the rate-gap after seeing data, or replace success with a hand-designed difficulty score.

If one pair passes, freeze that pair before hidden-state work.

## G1 — paired policy-specific success signal

### Feature location

Primary feature is fixed before collection:

```text
pi0.5 action-expert decoder layer 11 output
```

The **decision point is also frozen**: the feature is read at the settled initial state
(`replan_idx = 0`), the identical first decision both checkpoints face on that physical
state. Reading at a later timestep would compare different physical states, because the
two policies diverge the moment they act. Both checkpoints therefore see byte-identical
images, proprioception and prompt when the feature is captured.

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

Implementation: one ridge-linear readout. Target is the eight-rollout Monte-Carlo success
rate `p_hat(pi,s)`.

The penalty is **not** a free constant. The action expert is 1024-wide while discovery
supplies only `150 states x 2 checkpoints = 300` rows, so a fixed `alpha=1.0` would leave
the fit essentially interpolating and dominated by noise directions. It is selected from a
frozen grid by 5-fold cross-validation **within the discovery split only**, grouped by
physical `state_id`.

Grouping is essential rather than cosmetic: `h_A(s)` and `h_B(s)` are two views of the same
scene, so an ungrouped K-fold would place one view in train and the other in validation,
report an optimistic error, and select too little regularization. Confirmation states enter
neither the fit nor the selection.

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

Otherwise the power control decides which negative this is.

### Power control (pre-declared, cannot create a pass)

The paired contrast deliberately cancels generic state difficulty. That makes a null
relative AUROC ambiguous between two very different situations:

1. the representation carries success information, but only the policy-agnostic part
   (an informative negative about self-knowledge);
2. the representation carries no success information at all at `replan_idx 0`
   (a measurement-power failure that says nothing about self-knowledge).

`absolute_success_metrics` separates these by measuring, **within** each checkpoint, the
Spearman correlation between the shared readout `q` and the Monte-Carlo success rate
`p_hat`, over *all* confirmation states. The frozen threshold is a mean within-checkpoint
Spearman of `0.15`.

```text
relative gate passed                    -> PASS_POLICY_SPECIFIC_SUCCESS_SIGNAL
relative gate failed, |rho| >= 0.15     -> KILL_SELF_KNOWLEDGE_INTERPRETATION
relative gate failed, |rho| <  0.15     -> INCONCLUSIVE_NO_ABSOLUTE_SUCCESS_SIGNAL
```

This control is one-directional by construction: it can only downgrade a negative, never
turn a negative into a positive. Neither of the two lower branches is a CONTINUE.

Balanced accuracy of `sign(q_A-q_B)` is secondary only.

## Interpretation of a negative G1

A negative shared-readout result does **not** prove that no nonlinear or representation-aligned notion of self-knowledge exists. Different checkpoints can drift in representation space.

For this project that caveat does not trigger a rescue campaign. If the clean shared linear contrast fails, stop. Do not add Procrustes alignment, nonlinear probes, layer sweeps, SAE searches, failure taxonomies, or hand-selected subsets.

## Anti-complexity rule

The topic gets one natural behavioral contrast and one paired representation test. If the story requires more machinery to become true, the research question is demoted rather than the gate expanded.
