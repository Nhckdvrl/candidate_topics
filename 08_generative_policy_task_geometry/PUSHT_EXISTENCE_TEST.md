# E1 — PushT counterfactual existence test (design, frozen before results)

## The one question

For a fixed observation, a generative policy samples many different action chunks. Does a
**scalar** action-diversity score already tell you how uncertain the **task outcome** is?

If yes, this topic is dead: there is nothing for "task geometry" to add. So E1 is written
to make the death as easy to observe as the survival.

## Why not the planar arm

`AUDIT.md` finding A1: in the planar-arm prototype the outcome ("progress toward target")
is, to first order, `J(q) * sum_h a_h` — the same `J(q)` used to define the task-sensitive
projector. Task-space action variance produces outcome variance by algebra, for *any*
distribution including isotropic noise. That design cannot fail, so it cannot inform.

E1 replaces the outcome with contact-rich pymunk dynamics that know nothing about our
projections.

## Setup

* **Policy**: `lerobot/diffusion_pusht` — the LeRobot port of Diffusion Policy (Chi et
  al., RSS 2023) trained on the original PushT demonstrations. Released config: horizon
  16, `n_action_steps` 8, `n_obs_steps` 2, ResNet18 + spatial softmax, DDPM with 100
  training timesteps. **Nothing is trained here.** Eval mode ⇒ deterministic centre crop,
  so the only stochasticity is the diffusion noise.
* **Environment**: `gym_pusht/PushT-v0` (pymunk). Action = absolute target position of the
  circular pusher in `[0, 512]^2`. Task variable = pose of the T block; the env's own
  reward is goal-zone coverage.
* **Probe states**: every point at which the policy re-plans during a closed-loop rollout
  (i.e. every 8 environment steps). Episode seeds start at 100000, disjoint from the
  released evaluation seeds 1000–1049.

## Per probe state

1. Sample `B` action chunks from the identical observation history (the observation is
   replicated `B` times into one batch, so conditioning is bit-identical and only the
   diffusion noise differs).
2. Save the **complete** pymunk dynamic state (positions, velocities, angles, angular
   velocities of both bodies). `PushTEnv._set_state` is not sufficient: it ignores
   velocities and advances physics by one `dt`.
3. Execute each chunk open-loop from that identical restored state.
4. Record the outcome of each: the eight T-block keypoints in world pixels, plus coverage.

Chunk 0 is then used to continue the rollout, so probing costs no extra policy calls.

## Measurements

**Scalar action diversity**

* `ace` — Action-Chunk Entropy, transcribed from released FIPER code
  (`utiasDSL/fiper`), not from the paper: cell width `0.03 * calibration range` per
  dimension, per-state dynamic grid limits with a 1% buffer, `np.digitize`, Shannon
  entropy in bits over cell counts, **mean** over prediction steps. Calibration ranges
  come from a separate calibration rollout set (seeds below the measurement base).
* `act_rms_dispersion`, `act_trace_cov_mean` — estimator-free dispersion, so no
  conclusion can rest on one entropy estimator's binning choices.

**True task-outcome dispersion**

* `outcome_kp_dispersion_px` (primary) — RMS deviation of the B outcome keypoint sets
  from their mean shape, in pixels. Folds block translation and rotation into one metric
  without us picking a weight between them.
* `outcome_kp_pairwise_px` — mean pairwise keypoint distance (no mean-shape reference).
* `outcome_cov_std` — std of the env's own coverage reward across the B counterfactuals.

**Covariates recorded for the obvious mundane explanation**

* `agent_block_gap_px`, `frac_samples_with_contact`, `mean_contacts`. If "the pusher isn't
  touching the block" explains everything, we need to be able to see that.

## Analysis

* Spearman(score, outcome) pooled **and** within rollout. Probe states inside one rollout
  are strongly dependent; every CI resamples **rollouts**, never states.
* `binned_spread`: inside narrow quantile bins of the score, the p90/p10 ratio of outcome
  dispersion. This is the assumption-light form of the claim — no pairing, no threshold.
* `matched_pairs`: greedy 1:1 matching at `|Δz_score| ≤ 0.10`, pairs forced to come from
  *different* rollouts.
* Everything is stratified by contact state.

## Preflight (all must pass before any number is interpreted)

`src/pusht/preflight.py`:

| | check |
|---|---|
| P1 | same restored state + same actions ⇒ bit-identical keypoints |
| P2 | restore survives an intervening 40-step unrelated rollout |
| P3 | render-free `step_physics` ≡ `env.step` on the block trajectory |
| P4 | our D-dim ACE ≡ released FIPER 3-D code path on zero-padded 2-D actions |
| P5 | policy loads, samples differ, actions inside env bounds |

## Discovery / confirmation split

The first run is **discovery**: descriptive only, `analyze_e1.py` emits no verdict. Its
job is to check the measurements behave, catch bugs, and let us choose `B`, the number of
rollouts, and the decision thresholds.

Those thresholds are then written to `gate_e1.json` and **frozen**. Confirmation runs use
disjoint episode seeds on other nodes and are scored with `--gate` only. No threshold is
adjusted after confirmation starts.

## What would kill the topic

* Scalar ACE essentially determines outcome dispersion (high Spearman, and little outcome
  spread left inside narrow ACE bins).
* Any residual spread is fully explained by contact state, i.e. the finding reduces to
  "action diversity is harmless when the pusher is in free space".
* The effect exists pooled but vanishes within rollouts.
* The effect needs a particular entropy estimator or binning constant to appear.
