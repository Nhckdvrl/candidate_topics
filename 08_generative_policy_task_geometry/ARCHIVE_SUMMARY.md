# Archive Summary — Topic 08: Does Action Diversity Track Functional Uncertainty?

**Final status: ARCHIVED / KILLED AT THE OPERATIONAL BAR**

Archived 2026-08-22. The phenomenon is real and cleanly measured, but it is unsurprising
for nonlinear contact dynamics, and the operational claim that would have made it matter —
that a deployed entropy-based uncertainty monitor systematically makes wrong decisions
because of it — is **contradicted** by the data rather than merely unsupported.

The topic was tested twice: once as originally designed (analytic planar arm, "task
geometry"), and once in the stripped-down form that survived the audit (pretrained
Diffusion Policy on PushT, counterfactual execution from restored simulator states). The
first died of a design defect. The second produced a real but unsurprising phenomenon that
does not clear the bar we set for it.

---

## 1. What was originally claimed

> Two policy states can have similar scalar action entropy but different functional risk
> because their variability points in different task-relative directions.

Planned evidence: a 4-DoF planar arm with hidden posture preferences, a Jacobian
task/null decomposition, FIPER-style entropy matching, and a risk threshold.

## 2. Why that version was killed before it ran (`AUDIT.md`)

The prototype had never been executed. Auditing it found a blocking defect plus four
secondary ones.

**A1, blocking — the primary contrast was an algebraic identity.** `evaluate_g0.py`
defined risk from the end-effector's progress after executing `q + dt * sum_h a_h`, and
defined task-sensitive variance as `tr(P_task Sigma P_task)` using the row-space projector
of the *same* frozen `J(q)`. Since `fk(q + dq) - fk(q) = J(q) dq` to first order, progress
is an affine function of `P_task dq`. Variance along `row(J)` therefore produces variance
in progress by construction, for *any* action distribution — isotropic Gaussian noise
included. Gate `G3` could not fail, so it could not inform.

**A2** — the designed conditional multimodality existed only at `t=0`; `build_windows`
emitted a training window at every timestep, and by `t>=1` the joint configuration
identifies the hidden posture mode. ~3% of windows carried the intended structure.

**A3** — the ACE implementation did not match released FIPER (`utiasDSL/fiper`):
`cellsize_factor` 0.1 vs the released **0.03**, horizon **sum** vs **mean**. Also, and more
important for novelty: FIPER already computes ACE on the **Cartesian `position`** component
of the action chunk. "Compute entropy in end-effector coordinates" is the baseline, not a
contribution.

**A4** — the bootstrap resampled states i.i.d. though probe states within a rollout are
strongly dependent.

**A5** — gate `G1` (`median(NullPerDim/TaskPerDim) >= 0.75`) is passed most easily by an
*isotropic* distribution, for which the ratio is exactly 1.0. It cannot distinguish
learned goal-equivalent structure from a noisy sampler.

There is also a structural objection that no amount of patching fixes. The design required
a chain of constructions — inject hidden posture modes, require identical observations,
define a task/null decomposition, assume local linearity, match on entropy, threshold a
risk. Even complete success reduces to "we deliberately put variability into task-null
directions and then found a scalar entropy counted it". The conclusion is close to
determined by the setup.

## 3. The stripped-down question

Everything above was discarded in favour of the version that needs no hidden modes, no
Jacobian and no linearity assumption:

> Can a generative robot policy be highly diverse while remaining functionally certain?

Sample many action chunks from one policy at one state, execute each from that identical
simulator state, and compare action spread against true task-outcome spread.

**Setup.** `lerobot/diffusion_pusht` (LeRobot port of Diffusion Policy, Chi et al. RSS
2023), no training. `gym_pusht/PushT-v0`, pymunk. B=256 samples per state, probes at every
policy replan.

**Faithfulness.** Closed-loop replication on the released eval seeds: **68.0% ± 6.6%**
against the released **65.4%**. This check mattered — lerobot 0.4.4 silently discards this
checkpoint's normalisation buffers, and an unnormalised policy still emits plausible
actions while becoming uncertain everywhere, which would have manufactured a positive
result. Normalisation is restored by hand from the checkpoint and asserted at load.

## 4. Two simulator bugs that would have faked results

Both caught by preflight before any measurement.

* **59 px block teleport on restore.** `PushTEnv._set_state` assigns position then angle.
  The T's centre of gravity is offset `(0, 45)` from its body origin and pymunk rotates
  about the CoG, so assigning `angle` moves `position`. In the first smoke run **every**
  outcome dispersion was exactly zero. Replay-determinism checks cannot catch this: they
  compare two replays with each other, so a restore that is wrong identically every time
  passes. `P0` compares the restored state against the *saved* state.
* **3.67 px drift from Chipmunk's warm-start contact arbiter cache**, which survives a
  naive body-state restore. Same order as the real signal. `restore_sim_state` rebuilds
  the pymunk space.

## 5. Two analysis defects found and fixed mid-flight

* **A circular gate.** `matched_pairs()` selected pair members from the top and bottom
  quartiles of the *outcome*, then matched on ACE — so a large outcome ratio was
  guaranteed, bounded below by Q75/Q25 whatever the score did. It was a CONTINUE
  condition. Replaced by `matched_pair_reduction`, which forms pairs on the score alone.
* **Unimplemented kill criteria.** The design doc committed to killing on within-rollout
  disappearance, contact-state explanation, and estimator dependence; the gate checked
  only pooled quantities. All are now implemented.

## 6. Finite-B reliability (the residual is not estimator noise)

ACE and outcome dispersion come from the same 256 samples, so estimator noise could by
itself manufacture "same entropy, different outcome". Split-half over disjoint halves:

| quantity | reliability (rho) |
|---|---|
| outcome dispersion | 0.989 |
| ACE (factor 0.03) | 0.941 |
| ACE (factor 0.001) | 0.992 |

Median half-vs-half discrepancy **0.064 px** against a between-state IQR of **0.983 px**
(6.5%). `rho(B=128, B=256) = 0.977`. The measurement is saturated; the residual is real.

## 7. What the experiment actually found

### 7a. The phenomenon exists

At the 8-step horizon (187 probe states, 8 rollouts):

* 33.7% of probe states have **exactly zero** outcome dispersion — all 256 sampled chunks
  produce an identical block pose.
* Among in-contact states, outcome dispersion is **8.1%** of the block's own displacement:
  256 different sampled chunks push the T to nearly the same place.
* States *not* in contact have **higher** action dispersion (p50 2.79 px) than in-contact
  states (p50 1.99 px), with zero outcome consequence. The policy is most diverse exactly
  where diversity cannot matter.
* Matching two states on ACE reduces the median outcome difference by ~2%
  (`matched_pair_reduction = 0.978`, CI 0.73-1.19). Within rollouts: 1.15. In contact:
  0.797. Across cellsize factors 0.03/0.01/0.003/0.001 the reduction stays 0.66-0.93, so
  this is not an artefact of FIPER's constant.

So yes: **a generative policy can be highly diverse while remaining functionally certain**,
and scalar entropy does conflate the two.

### 7b. The 8-step horizon understated the outcome, so we measured episode-level

8-step in-contact dispersion is ~1 px on a 512 px workspace, which is not obviously "task
uncertainty" at all. `collect_e1b` therefore executes K=32 sampled chunks from the
identical restored state and continues each **closed-loop under the same policy** for 88
further steps. 978 branch states over 96 rollouts, 14 GPUs across 5 nodes.

Episode-level outcomes are substantial: branch dispersion p50 **7.6 px**, p90 **47.4 px**,
max **159 px**; and **53.3%** of states have branches that disagree on whether the goal is
reached. The 8-step measure understates by roughly an order of magnitude.

### 7c. But the deployed monitor is not broken — and that is what kills it

The bar we set was operational: does a monitor thresholding scalar action entropy actually
make wrong decisions? Measured at FIPER's own released operating quantiles against
episode-level ground truth (top-quartile branch dispersion, base rate 0.251):

| quantile | alarms | precision | precision / base rate | alarms on benign states |
|---|---|---|---|---|
| 0.90 | 98 | 0.459 | **1.83x** | 12.2% |
| 0.95 | 49 | 0.490 | **1.96x** | 10.2% |
| 0.99 | 10 | 0.600 | **2.40x** | 10.0% |

Pooled AUC **0.579** (CI 0.546-0.625). That is a weak but genuinely informative monitor.
Roughly one alarm in ten lands on a state where all branches end up in the same place —
nothing like "fires on states where every sampled action leads to the same place".

The only striking result is stratified:

| stratum | n | AUC | precision/base @ q=0.90 |
|---|---|---|---|
| near block (gap < 20 px) | 463 | **0.496** | 1.10 |
| far from block | 515 | 0.645 | — |

**ACE's entire skill is a proxy for proximity to the object.** Conditioned on being near
the block — the regime where actions actually matter — it is exactly at chance.

## 8. Why this is a KILL and not a CONTINUE

1. **The bare phenomenon is unsurprising.** Contact dynamics are nonlinear and
   state-dependent. A pusher in free space being diverse-and-harmless, and a millimetre
   near contact mattering, is what anyone would predict. "Action entropy and outcome
   uncertainty are imperfectly correlated" is not a result.
2. **The operational claim does not hold.** The version that would have mattered —
   existing entropy-based mechanisms systematically making wrong calls — is contradicted
   by the data. Precision is ~2x base rate at every FIPER operating point.
3. **What survives is narrow.** "ACE tracks proximity to the object rather than functional
   uncertainty" is true and cleanly measured, but it is a much smaller claim than the
   topic was built for, it is confined to one stratum defined by a threshold we chose, and
   it invites the obvious rebuttal that a proximity feature would fix it.
4. **The geometry story was never reached and should not be resurrected.** Going back to a
   task-sensitive/task-null decomposition would reintroduce exactly the chain of
   assumptions — local linearity, projector choice, matching tolerance — that made the
   original design unfalsifiable.

## 9. What would revive it

Only a result of this shape: an entropy-based runtime mechanism, run as released, causing
**materially worse task performance** than a trivial alternative — measured end-to-end, not
as a correlation. That means intervention rates, recovered failures, and spurious stops
under a real policy. Nothing short of that distinguishes this from "a scalar proxy is
imperfect".

## 10. Reproducing

```bash
pip install -r requirements-pusht.txt
bash run_e1_pilot.sh                       # preflight + 8-step existence test
bash run_e1b_fleet.sh disc                 # episode-level branching
python -m src.pusht.decision_analysis \
  --csv results/pusht_e1b_disc/shard*/branch_states.csv \
  --out results/pusht_e1b_disc/decision_report.json
```

Preflight (`src/pusht/preflight.py`) must pass before any number is interpreted; it is
what caught both simulator bugs. Committed results are the small CSV/JSON summaries; raw
per-sample arrays are not committed.
