# 08 — Does action diversity track functional uncertainty?

## Status

| | |
|---|---|
| planar-arm G0 (original prototype) | **killed** — see `AUDIT.md` |
| PushT existence test E1 (8-step outcome) | discovery complete |
| PushT branch test E1b (episode-level outcome) | running |
| decision-level test (does a deployed monitor misfire?) | the actual bar; pending E1b |

## The question, after demotion

The folder was originally titled *"Generative Policy Diversity Has Task Geometry"* and the
planned claim was about a task-sensitive / task-null decomposition of the sampled action
distribution. That framing is **demoted and is no longer the claim being tested.**

The reason is a selection-effect worry rather than a technical one. The old design needed
a chain: manufacture hidden posture modes → require identical observations → define a
task/null decomposition → require local linearity → match on entropy → define a risk
threshold → pass several gates. Even a full success reduces to "we deliberately injected
variability into task-null directions and then found that a scalar entropy counted it".
That is close to true by construction, and `AUDIT.md` finding A1 shows one of the gates
was literally an algebraic identity.

What survives is the part that needs none of that machinery:

> **Can a generative robot policy be highly diverse while remaining functionally certain?**

Concretely: sample many action chunks from one real policy at one real state, execute each
from that identical simulator state, and compare how spread out the *actions* are with how
spread out the *task outcomes* are. No hidden modes, no Jacobian, no linearity assumption.
One scatter plot answers it.

## The harder bar

Suppose the phenomenon is real. Is it surprising? Largely no. Robot dynamics are nonlinear
and state-dependent, so it is expected that a pusher moving in free space is diverse and
harmless while a small difference near contact matters a lot. **"Action entropy and
outcome uncertainty are imperfectly correlated" is not a result**, and on its own it
should end this topic.

The version that would matter is operational:

> A deployed uncertainty monitor that thresholds scalar action entropy fires on states
> where every sampled action leads to the same place, and stays quiet on states where the
> sampled actions genuinely diverge.

That is a semantic mismatch in a mechanism people actually run (FIPER, adaptive action
chunking), not a geometric curiosity. `src/pusht/decision_analysis.py` measures it against
episode-level branch outcomes: AUC for ranking states by true functional uncertainty, and
precision at FIPER's own released operating quantiles (0.90/0.95/0.99) against the base
rate. A monitor at chance is the finding; a monitor that works is the kill.

## Setup

No training. `lerobot/diffusion_pusht` — the LeRobot port of Diffusion Policy (Chi et al.,
RSS 2023) on the original PushT demonstrations. Our closed-loop replication on the released
eval seeds gives **68.0% ± 6.6%** against the released **65.4%**, so the inference path is
faithful (`results/pusht_preflight/replicate_eval.json`).

Environment is `gym_pusht/PushT-v0` (pymunk). Action is the pusher's absolute target
position in `[0,512]²`; the task variable is the T-block pose.

## Measurement

At every point where the policy re-plans (every 8 env steps):

1. sample B=256 chunks from the identical observation — conditioning is bit-identical
   across the batch, only the diffusion noise differs;
2. save the **complete** pymunk dynamic state;
3. execute each chunk from that identical restored state;
4. record the true outcome (T-block keypoints, coverage).

E1b additionally continues K=32 of those branches **closed-loop under the same policy**
for a further horizon, so the outcome is episode-level rather than 8 steps of contact.

Scores: FIPER ACE transcribed from released code (`utiasDSL/fiper`) — cell width
`0.03 × calibration range`, per-state dynamic grid, horizon mean, scored on the full
16-step predicted chunk — plus two estimator-free dispersion measures so nothing rests on
one binning constant.

## Two simulator bugs that would have faked a result

Both found by preflight, before any measurement.

* **59 px block teleport on restore.** `PushTEnv._set_state` assigns position then angle;
  the T's centre of gravity is offset from its body origin, so assigning `angle` moves
  `position`. In the first smoke run *every* outcome dispersion was exactly zero. Note
  that replay-determinism checks cannot catch this — they compare two replays with each
  other, so a restore that is wrong identically every time passes. That is why `P0`
  compares the restored state against the *saved* state.
* **3.67 px drift from Chipmunk's warm-start contact cache**, which survives a naive
  restore. Same order as the real signal. `restore_sim_state` now rebuilds the space.

## Analysis rules

* **Never gate on a statistic that selects on the outcome.** `matched_pairs_descriptive`
  picks pair members from the top and bottom outcome quartiles, so its ratio is large by
  construction. The gated statistic is `matched_pair_reduction`, which forms pairs using
  the score alone.
* States within a rollout are dependent. Every CI resamples **rollouts**.
* Every kill criterion in `PUSHT_EXISTENCE_TEST.md` is implemented in the gate, including
  the within-rollout, contact-stratified, estimator-robustness and finite-B noise clauses.

## Run

```bash
pip install -r requirements-pusht.txt
bash run_e1_pilot.sh          # preflight, collection, descriptive analysis
bash run_e1b_fleet.sh disc    # episode-level branching across free GPUs
```

## Layout

```text
src/pusht/sim_state.py         exact pymunk save/restore
src/pusht/env_utils.py         render-free physics, counterfactual execution
src/pusht/ace.py               FIPER ACE, transcribed from released code
src/pusht/policy_utils.py      pretrained policy + hand-restored normalisation
src/pusht/collect_e1.py        8-step counterfactual collection
src/pusht/collect_e1b.py       episode-level branch collection
src/pusht/analysis.py          non-circular matched-pair statistics
src/pusht/analyze_e1.py        E1 report + frozen gate
src/pusht/decision_analysis.py does a deployed entropy monitor misfire?
src/pusht/reliability.py       split-half + estimator-sensitivity checks
src/pusht/geometry_e2.py       empirical local sensitivity (demoted; only if E1b survives)
src/planar_arm.py, src/*_g0.py the killed prototype, kept for the record
```
