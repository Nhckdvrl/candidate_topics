# Data contract — v2

## Physical state identity

A scientific state is keyed by:

```text
suite, task_id, init_idx, env_seed
```

and verified by:

```text
sim_state_hash
```

`sim_state_hash` is computed from the settled MuJoCo simulator state after the official ten dummy actions. The hash must be identical across checkpoints and stochastic repeats. A mismatch aborts analysis.

## G0 behavior panel

One CSV row is one stochastic policy rollout:

```text
suite,task_id,init_idx,env_seed,sim_state_hash,checkpoint,policy_seed,success,status,steps,replans
```

Requirements:

- every physical state is evaluated by every checkpoint present in the panel;
- every checkpoint on a state uses exactly the same `policy_seed` set;
- `policy_seed` defines a full deterministic inference-noise stream, not one reused noise tensor;
- `success` is official LIBERO binary success;
- `status` must be `ok`; technical-error rows are never interpreted as failures;
- no state/rollout is dropped after seeing outcomes.

The analyzer aggregates the repeated rows to `p_hat(checkpoint,state)` before defining crossover.

## G1 raw feature panel

Features are stored as NPZ arrays:

```text
state_id       [N]
checkpoint     [N]
sim_state_hash [N]
feature_seed   [N]
feature        [N,D]
```

There are four feature-noise repeats per `(state,checkpoint)` in the locked protocol. The two checkpoints must use exactly the same feature-seed set. Analysis averages the four features before fitting the shared readout.

`feature` is fixed as:

```text
pi0.5 action-expert layer 11 full layer output
-> mean over action tokens
-> mean over 10 denoising steps
```

The four common-noise feature repeats are then averaged.

## Train / confirmation separation

Frozen LIBERO-10 split:

```text
discovery:    init_idx 0..14 for each task
confirmation: init_idx 15..29 for each task
reserve:      init_idx 30..49
```

Discovery and confirmation also use disjoint behavior policy seeds and disjoint feature-noise seeds.

No physical `state_id` may appear on both sides.

## What cannot define state inclusion

Do not filter on:

- probe score;
- feature geometry;
- failure subtype;
- camera/visual perturbation response;
- action entropy;
- hand-selected trajectories;
- whether the example makes the story look clean.

The only primary eligibility rule is the frozen Monte-Carlo crossover definition in `VALIDATION.md`.
