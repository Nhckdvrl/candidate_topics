# 09 — Does a VLA know its own limits?

## Status

**Candidate / identification-first. No mechanism claim yet.**

> When a VLA carries a signal that predicts eventual success, is that signal specific to **this policy's own chance of succeeding**, or is it mostly a policy-agnostic estimate that **this state looks easy / hard**?

Recent work already establishes the prerequisite phenomenon: frozen VLA representations predict eventual success, even under same-task / same-timestep matching. The same literature also finds strong success information in generic visual encoders, so the interpretation is unresolved. We do not need to invent a new phenomenon; we need to identify what the existing signal means.

## The clean contrast

Use the **same settled simulator state**, the **same VLA family**, and different checkpoints from one pi0.5 LIBERO fine-tuning trajectory.

For checkpoint pair A/B at state s:

```text
same physical state s
     |                     |
     v                     v
 checkpoint A          checkpoint B
     |                     |
  p_A(success|s)        p_B(success|s)
     |                     |
  h_A(s)                h_B(s)
```

We need naturally occurring **bidirectional competence crossover**:

```text
some states: p_A >> p_B
other states: p_B >> p_A
```

If one checkpoint simply wins everywhere, a global checkpoint-quality prior could mimic self-evaluation. We do not manufacture crossover states.

## Why one rollout per state is invalid

pi0.5 is a stochastic flow policy. One success/failure rollout can flip purely because a different action-noise sample was drawn.

G0 therefore uses **eight common policy-noise streams per checkpoint/state**. Every checkpoint receives the same base policy seeds, and each replanning decision deterministically derives its own Gaussian-noise seed. Competence is estimated as:

```text
p_hat(pi,s) = successes / 8
```

A state is a robust A-win only when `p_A - p_B >= 0.50`; reverse for B. Smaller gaps are ambiguous and do not support the claim.

That rule is still not noise-free. Two checkpoints that are *equally* competent at a state
produce a spurious robust win about 3.8% of the time at `p=0.5`, so a 150-state panel would
be expected to show roughly six false wins in each direction with no competence difference
anywhere. G0 therefore also runs an exact **within-state relabeling null**: each state's 16
observed outcomes are pooled and re-split, holding that state's difficulty exactly fixed
while destroying the checkpoint association. Crossover that does not beat this null is
sampling noise, and the topic stops.

## Same state means actually the same state

The protocol follows OpenPI's official LIBERO evaluator: fixed LIBERO init state, environment seed 7, ten dummy settling steps, 224x224 image preprocessing, five executed actions per replan, official LIBERO success.

After settling we hash the flattened MuJoCo simulator state. The hash must match across every checkpoint process and repeat. `task_id + init_idx` alone is not accepted as proof of state identity.

`src/preflight.py` additionally requires same observation + same policy-noise seed to produce bit-identical actions and bit-identical captured features.

## G0 — behavioral identifiability

Preferred family: released pi0.5 LIBERO checkpoints **2k / 3k / 9k** from the same fine-tuning trajectory.

Frozen discovery panel:

```text
LIBERO-10 tasks 0..9
init_idx 0..14 per task = 150 physical states
8 policy-noise repeats per checkpoint/state
```

`src/analyze_disagreement.py` selects the pair with the most bidirectional robust support.

Require at least **15 robust A-wins and 15 robust B-wins**, and require that support to
beat the sampling-noise null. Otherwise:

```text
STOP_NO_NATURAL_CROSSOVER
STOP_CROSSOVER_EXPLAINED_BY_SAMPLING_NOISE
```

No perturbation search, special checkpoint training, or lowered threshold after seeing data.

## G1 — does the success signal follow *whose* success?

Only after G0 passes and the pair is frozen.

Primary representation is fixed as the **pi0.5 action-expert layer-11 full decoder-layer output**, motivated by prior success/failure activation work. OpenPI's PyTorch inference invokes the action expert once for each of ten flow denoising steps, so `src/openpi_instrumented_server.py` can observe layer 11 with a normal forward hook without changing model computation.

For each action-noise draw:

```text
layer-11 residual -> mean over action tokens -> mean over 10 denoise steps
```

Because this representation itself depends on the stochastic noisy action, each `(state,checkpoint)` feature is then averaged over **four common feature-noise seeds** shared by both checkpoints.

Fit **one shared standardized ridge-linear readout** across both checkpoints on discovery states:

```text
q = w^T h + b
```

Target is the eight-rollout Monte-Carlo success rate, not a one-shot label. Separate checkpoint-specific probes are forbidden.

On independent confirmation states (`init_idx 15..29`, new behavior seeds, new feature seeds), compute:

```text
relative_score(s) = q_A(s) - q_B(s)
```

and ask whether it predicts which checkpoint has the robustly higher success probability.

Primary statistic:

```text
AUROC(relative_score, A-is-winner)
```

Continue only if confirmation independently retains >=15 crossover states each way and:

```text
AUROC >= 0.70
bootstrap 95% lower bound > 0.60
```

A failed relative gate is reported as one of two different things. If the readout does
track success *within* a checkpoint, the negative is informative — the signal is generic
difficulty. If it tracks nothing at all, the paired test simply had no signal to reverse
and the result is `INCONCLUSIVE`, not evidence about self-knowledge. That power control is
pre-declared and one-directional: it can only downgrade a negative, never create a pass.

Why this is informative:

- physical state, task, images and proprioception are fixed inside each pair;
- a state-only difficulty signal cancels in `q_A-q_B`;
- a constant “9k is globally better” offset cannot rank bidirectional crossover;
- one readout is shared across checkpoints.

A strong result establishes an **operational policy-specific success signal**. It should not yet be called an explicit self-model.

## Negative-result discipline

A weak shared-readout result does not logically prove that no nonlinear or representation-aligned self-knowledge exists. For this project, however, that is where we stop. We do **not** rescue a negative result with Procrustes alignment, nonlinear probes, layer sweeps, SAE, failure taxonomies, or hand-picked states.

If G1 passes, only then does the mechanism question become natural: **where along the VLA does generic state difficulty become policy-specific competence?**

## Files

```text
LOCKED_CONFIG.json              frozen v3 constants / state split / noise seeds
ENVIRONMENT.md                  reproducible openpi + LIBERO setup and its traps
VALIDATION.md                   exact scientific contract and kill lines
DATA_CONTRACT.md                behavior + feature schemas
SERVER_HANDOFF.md               cluster execution order
src/state_contract.py           simulator-state hash + deterministic noise streams
src/libero_common.py            frozen official LIBERO preprocessing/reset protocol
src/openpi_instrumented_server.py controlled-noise inference + observational layer-11 hook
src/preflight.py                state/RNG/feature identity checks
src/check_checkpoints_differ.py verifies the checkpoints are not the same model
src/wait_for_server.py          blocks until a policy server finishes loading
src/collect_behavior.py         repeated same-state rollouts
src/panel.py                    robust Monte-Carlo crossover statistics
src/noise_null.py               within-state relabeling null for the crossover claim
src/analyze_disagreement.py     G0 pair selection / stop decision
src/collect_features.py         repeated common-noise layer-11 extraction
src/feature_panel.py            feature validation and averaging
src/relative_probe.py           shared linear readout + paired metrics
src/run_g1.py                   independent G1 confirmation / verdict
run_g0_fleet.sh                 multi-GPU shard runner for the behavior panel
tests/                          false-positive and instrumentation tests
```
