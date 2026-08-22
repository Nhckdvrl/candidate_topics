# Topic 09 audit — 2026-08-22

Audit of the inherited v2 protocol, performed **before any behavior or feature data
existed**, so every change below is a pre-registration amendment rather than a reaction to
a result. Frozen constants moved to `topic09-v3` in `LOCKED_CONFIG.json`; no decision
threshold was changed.

The checklist below is the one the topic brief asked for, in order.

## 1. Does "same state" really mean the same physical simulator state?

**Held up, and is now empirically verified rather than assumed.**

The design was already right not to trust `task_id + init_idx`: `settle_initial_state`
hashes the flattened MuJoCo state after the ten official dummy settling steps, and
`validate_panel` aborts if one `state_id` ever shows more than one hash.

Verified on the real stack for LIBERO-10 tasks 0, 4 and 9:

```text
same init_idx, three resets in one process  -> identical settled sim_state_hash
same init_idx, freshly constructed env      -> identical settled sim_state_hash
different init_idx                          -> different hash
same init_idx, two resets                   -> bit-identical agentview image
```

The fresh-environment case is the one that matters, because each checkpoint runs in its own
process with its own env.

One deliberate deviation from OpenPI's evaluator is now documented in the code: the official
script seeds the env once at construction and then resets repeatedly, so each episode
inherits a different RNG state. Topic 09 re-seeds before every reset, because state identity
across processes is the premise of the whole design. `make_env` also mirrors the official
global numpy seeding.

## 2. Is the generative policy's randomness handled fairly?

**Held up.** Competence is `p_hat = successes/8` over eight rollouts, never a single draw.
Every checkpoint gets the *same* eight base policy seeds on a state, and each replanning
decision derives its own Gaussian-noise seed from
`(policy_seed, suite, task_id, init_idx, replan_idx)` — common random numbers without
freezing one noise tensor across the episode. `validate_panel` refuses a panel whose
checkpoints do not share a seed set.

## 3. Do the checkpoints share an RNG and rollout protocol?

**Held up**, and one latent hazard was closed. Both G0 rollouts and G1 feature extraction
go through the same PyTorch inference stack, so the two gates can never end up describing
two different policies — the JAX and PyTorch paths would not be the same policy.

## 4. Does the crossover definition mistake sampling noise for competence?

**This was the one real hole, and it was the topic brief's own question 2.**

Nothing in v2 could answer it. The `p_A - p_B >= 0.5` rule is a large-effect rule, but with
eight rollouts it is not noise-free. For two checkpoints with *identical* competence at a
state with true rate `p`, the chance that noise alone yields a robust "A win" is:

```text
p=0.1 -> 0.0023   p=0.3 -> 0.0263   p=0.5 -> 0.0384
p=0.7 -> 0.0263   p=0.9 -> 0.0133
```

At the worst case a 150-state panel would be expected to show about **six spurious A-wins
and six spurious B-wins with no competence difference anywhere**. The `min(...) >= 15` gate
sits only about four standard deviations above that floor, and the true per-state rates are
unknown, so the analytic argument alone is not enough.

Added `src/noise_null.py`: an exact **within-state relabeling test**. Each state's sixteen
observed outcomes are pooled and randomly re-split into two groups of eight. That holds the
state's pooled difficulty *exactly* fixed and destroys only the outcome/checkpoint
association, so any crossover surviving it is sampling noise by construction. G0 now reports
observed vs null for both `min(n_a_wins, n_b_wins)` and the direction-free
`n_a_wins + n_b_wins`, and stops with `STOP_CROSSOVER_EXPLAINED_BY_SAMPLING_NOISE` if the
observed support does not clear the null.

The direction-free count is reported alongside because the permutation null is symmetric in
A/B while a genuine global quality gap would *lower* the bidirectional count; relying on the
bidirectional statistic alone would be anti-conservative.

## 5. Do the hidden features come from the same decision point and semantic position?

**Correct in code, undocumented in the contract.** `collect_features.py` queries each
checkpoint once per feature seed at the settled initial state (`replan_idx 0`), before either
policy has acted, so both see byte-identical images, proprioception and prompt. Any later
timestep would silently be comparing two different physical states.

That is now stated explicitly in `VALIDATION.md`, `DATA_CONTRACT.md` and `LOCKED_CONFIG.json`
rather than being an implicit property of the collector.

The semantic position is enforced rather than trusted: the capture hook raises unless it sees
exactly ten denoising activations with a batch dimension of one, so a change in OpenPI's
inference path fails loudly instead of quietly pooling the wrong thing. Confirmed against the
pinned commit — `sample_actions` runs `num_steps=10` and the expert is entered once per step,
and for pi0.5 the suffix carries action tokens only (no state token), so "mean over action
tokens" is exactly what the pooling computes.

## 6. Is there leakage between train / discovery / confirmation?

**One real leak, in a place that did not exist yet as a question.**

The split itself was clean: disjoint `init_idx` ranges, disjoint behavior seeds, disjoint
feature seeds, and `run_g1` raises if any physical state appears on both sides.

But the readout is fit on `150 states x 2 checkpoints = 300` rows against a **1024**-dim
feature. The v2 penalty was pinned at `alpha=1.0`, which at that ratio leaves the fit
essentially interpolating and dominated by noise directions — biasing G1 toward a spurious
KILL for a purely numerical reason. The penalty is now selected from a frozen grid by 5-fold
CV **inside discovery only**.

That selection introduces its own leak if done naively: `h_A(s)` and `h_B(s)` are two views
of one scene, so an ungrouped K-fold would place one view in train and the other in
validation, report an optimistic error, and choose too little regularization. The CV is
therefore grouped by physical `state_id`.

## 7. Can the shared probe exploit a checkpoint-identity shortcut?

**No, and this is the part of the inherited design that is genuinely well built.**

Write `h_c(s) = mu_c + f_c(s)`. Then

```text
q_A(s) - q_B(s) = w.(mu_A - mu_B) + w.(f_A(s) - f_B(s))
```

The first term is a constant, and AUROC is invariant to a constant offset, so a probe that
learns nothing but "which checkpoint is this" scores exactly 0.5 on bidirectional crossover
states. A pure state-difficulty signal cancels outright. Both are covered by existing unit
tests, and both are the reason the primary statistic must be a *bidirectional* AUROC rather
than accuracy.

## Additions beyond the checklist

- **Power control for G1.** A null relative AUROC was ambiguous between "the representation
  carries only generic difficulty" (informative) and "the representation carries no success
  signal at all at `replan_idx 0`" (measurement failure). The report now carries a
  pre-declared within-checkpoint Spearman between the readout and `p_hat`. It is
  one-directional by construction — it can only downgrade a negative to
  `INCONCLUSIVE_NO_ABSOLUTE_SUCCESS_SIGNAL`, never turn one into a pass.
- **Sanity anchor.** The G0 report now includes each checkpoint's overall LIBERO success
  rate. A broken normalization or action stack should be caught there, not inferred from a
  strange crossover count.
- **Restartability.** A full panel is thousands of rollouts and many GPU-hours;
  `collect_behavior --resume` skips `(state, seed)` pairs already on disk.

## Environment traps found while building the stack

Recorded in full in `ENVIRONMENT.md`. The one that could have silently corrupted the science:

> `examples/convert_jax_model_to_pytorch.py` chooses between pi0.5 adaptive-RMSNorm
> (`Dense_0`) and plain pi0 RMSNorm (`scale`) parameters via `if "pi05" in checkpoint_dir`
> — a substring test on the path string. A checkpoint directory named `jax_2000` takes the
> wrong branch. Every checkpoint path must contain `pi05`.

Also: LIBERO cannot be installed with `pip install .` (its package directory has no
`__init__.py`, so the install silently produces nothing importable); it prompts interactively
on first import and raises `EOFError` under a non-interactive shell; and `torch>=2.6` cannot
read its pickled initial states under the new `weights_only=True` default, which surfaces as
an unpickling error rather than as a missing-data error. OSMesa is unavailable on these
nodes, so rendering uses EGL.

## Preflight outcome — 2026-08-22

P0 passed on all three checkpoints. Every number below is measured, not assumed.

```text
                                 2k       3k       9k
settled sim-state hash stable    yes      yes      yes
same noise -> action max|diff|   0.0      0.0      0.0
same noise -> feature max|diff|  0.0      0.0      0.0
diff noise -> action rms       0.0047   0.0046   0.0207
layer-11 denoise activations     10       10       10
feature dim                     1024     1024     1024
```

Bit-identical actions and features under a repeated noise seed is the strongest form of
the RNG-control claim: policy stochasticity is fully accounted for by the declared seed,
with nothing else varying between calls.

Checkpoint distinctness, same observation and same noise seed:

```text
pair      action rms   feature rms
2k / 3k     0.0224        0.766
2k / 9k     0.0960        3.324
3k / 9k     0.0780        2.728
```

The ordering is the one a single fine-tuning trajectory should produce — 2k and 3k are
adjacent and differ least, 2k and 9k are furthest apart. A conversion that had written the
same weights twice, or mixed up a branch, would not reproduce that structure.

Rollout cost, measured on 9k over eight full rollouts:

```text
mean wall clock      16.2 s / rollout
mean inference       197 ms, ~50 inferences / rollout
time split           61% inference, 22% simulator
```

Inference dominates, so the GPU is the bottleneck and the useful parallelism is
independent server/collector pairs rather than more collectors per server.

One thing to watch rather than fix: 9k succeeded on all eight pilot rollouts. If it sits
near ceiling on LIBERO-10 it will win most states outright, and a one-directional
advantage is not a crossover. That would surface as weak bidirectional support in G0 and
should be read as `STOP_NO_NATURAL_CROSSOVER`, not as something to engineer around.


## Environment reuse silently breaks state identity — found during G0, 2026-08-22

The state-identity guard fired 2074 rollouts into the discovery panel:

```text
RuntimeError: settled state changed across repeats: task=3 init=4
```

All three checkpoints failed at the *same* physical state despite having completely
different preceding trajectories, which ruled out a flaky per-trajectory effect.

What the measurements showed:

```text
fresh env, same init settled 5x back to back      max|diff| = 0.0
fresh env, 3 short random episodes in between     max|diff| = 0.0
fresh env, all 8 policy seeds at init 4           max|diff| = 0.0, hashes identical
30 successive fresh envs                          all hashes identical, RSS flat
inside the collector, init 4 after inits 0-3      settled state differs
```

So the settled state depends on **how many episodes the environment has already run**, not
on which trajectory it ran. `reset()` + `set_init_state()` does not fully restore whatever
accumulates.

This is fatal rather than untidy. The entire design rests on two checkpoints meeting the
identical physical state, and the two checkpoints have *different* episode histories by
construction. Left alone it would not have produced an error — it would have produced two
subtly different states presented as one.

Fix: build one environment per rollout, in both `collect_behavior.py` and
`collect_features.py`, so the two panels reach the settled state the same way. Cost is
about 4.7s per rollout with flat memory. A regression test pins `make_env` inside the
per-rollout loop.

The discovery data collected before the fix was archived rather than reused, because it
was produced under a different protocol.

Worth noting: the guard that caught this was added in the pre-data audit precisely because
`task_id + init_idx` was not to be trusted as proof of state identity. Without it, this
would have been an invisible confound in the primary result.
