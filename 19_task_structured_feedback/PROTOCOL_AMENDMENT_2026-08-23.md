# Protocol amendment — 2026-08-23, before G0 scoring

This amendment is frozen **before any G0a response metric is scored**. It records an upstream-evaluation-set limitation and two environment-only compatibility deviations discovered during implementation. The scientific contrast, perturbation construction, primary metric, and GO/KILL thresholds are unchanged.

## A1. Independence unit: 10 released level-0 environment configs

The released SIMPLE level-0 CloseDoor evaluation set contains only **10 distinct environment configurations**. The original G0 text requested 20 successful level-0 episodes and called the episode the independent bootstrap unit. Twenty distinct level-0 configs therefore do not exist.

The frozen implementation is:

- use all **10** released level-0 configs;
- obtain **2 independent successful policy rollouts per config**, for 20 successful rollouts total;
- retain the last three fresh-policy query states before success from each rollout, subject only to the already-frozen kinematic validity gate;
- use the four already-frozen common-random-number pair seeds per state;
- average pair seeds and states within rollout, then average the two rollouts within config;
- the **10 config-level means are the only independent units used for the primary bootstrap CI and GO/KILL verdict**.

Thus the primary hierarchy is:

`pair seeds -> states -> rollout -> environment config -> bootstrap across 10 configs`.

A bootstrap treating the 20 rollouts as independent may be reported only as a **secondary diagnostic**. It must not affect the frozen verdict.

Reason: the two rollouts from one config share the same scene/object/environment realization. Treating them as 20 independent environment samples would understate config-level dependence. Clustering at config level is the conservative and statistically cleaner interpretation of the original independence requirement.

The collector should preserve both `config_id` and `rollout_id`. For compatibility with the current analyzer, `episode_id` should equal the **config id** for the primary analysis, so all rows from both rollouts of one config are clustered together.

## A2. Git submodule transport rewrite

Some public submodule URLs were rewritten from `git@github.com:` to HTTPS so they can be fetched on the server.

This changes transport only. Repository identities and audited commit SHAs must remain unchanged. Record the rewrite in `G0_RESULTS.md`; it is not a scientific protocol deviation.

## A3. Blackwell CUDA/PyTorch compatibility

The audited Psi0 lockfile resolved `torch 2.7.0+cu126`, which does not provide a usable kernel path for the server's Blackwell `sm_120` GPU. The run therefore uses the same PyTorch version with the **cu128** wheel, consistent with SIMPLE's own CUDA 12.8 package index. `flash-attn 2.7.4.post1` must be runtime-smoke-tested on `sm_120` before G0.

This is an execution-backend compatibility change only. Model checkpoint, model code, inference algorithm, preprocessing, action semantics, pair seeds, perturbations, and metric remain frozen. Record exact package versions and the smoke-test result in `G0_RESULTS.md`.

## No other amendment

Do not change because of this amendment:

- Psi0 checkpoint (`ckpt_40000` for the audited CloseDoor setup);
- SIMPLE/Psi0 source revisions except a documented unavoidable compatibility patch;
- `epsilon=0.08`;
- task/null direction construction;
- physical re-rendering contract;
- common-random-number pair seeds;
- first absolute right-arm target metric;
- `DeltaR = R_task - R_null`;
- GO: mean `DeltaR >= 0.20` and 95% config-bootstrap CI lower bound `> 0`;
- KILL: config-bootstrap CI upper bound `<= 0.10`;
- otherwise `INCONCLUSIVE_DO_NOT_TUNE`.
