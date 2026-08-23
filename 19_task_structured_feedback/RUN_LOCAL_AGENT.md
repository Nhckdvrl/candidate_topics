# Local-agent run contract

Implement the environment-specific collector around the already-frozen logic in `g0_core.py` and `g0_simple_psi0.py`. Do not redesign the scientific test.

## Phase P0 — exact stack qualification

1. Checkout the audited SIMPLE / Psi0 revisions or document any necessary newer revision and diff the touched files.
2. Start the released Psi0 checkpoint through the official SIMPLE stack.
3. Run 10 fixed level-0 `G1WholebodyCloseDoorTeleop-v0` episodes.
4. Require >=8/10 success. If not, report `PLATFORM_PREREQUISITE_FAIL` and stop.
5. Verify action shape/order from source and runtime: right arm must be action dims `21:28` and absolute joint targets.

## Phase P1 — deterministic paired-query preflight

Expose a direct/debug model-query path that performs exactly the same image/state preprocessing as `psi0_serve_simple.py`, but permits `seed_everything(pair_seed)` immediately before inference.

For one saved physical observation:

- query twice with identical state and pair seed;
- assert first returned 36-D action max absolute difference <=1e-6 (or the smallest tolerance justified by the actual dtype/backend; report it explicitly);
- branch queries must not mutate/use each other's RTC state;
- preserve the same previous base-height/context value used by the deployed Ψ₀ state vector across base/task/null branches.

Failure: `PAIRED_INFERENCE_NOT_IDENTIFIED`; stop.

## Phase P2 — collect deployed states

Run 20 successful level-0 episodes using the official RTC policy. Record every *fresh policy query* state (not every 200-Hz WBC step), including a restorable MuJoCo snapshot and the policy-side previous-height/context values needed to rebuild exactly the same Ψ₀ state input. Retain the last three fresh-policy query states before success from each episode.

No response-based frame selection.

## Phase P3 — construct physical pairs

At each selected state:

1. restore exact snapshot;
2. call `build_pair_from_sim(..., epsilon=0.08)`;
3. reject if fixed finite-geometry gate fails or a joint limit/simulator-validity check fails;
4. base/task/null must start from the same snapshot;
5. physically change **only** the seven right-arm qpos values, preserve qvel/controller/time/other state, call `mj_forward` without integration, then re-render/rebuild proprio.

Do not spoof proprio with a frozen image and do not introduce a second velocity/history intervention.

## Phase P4 — G0a common-random-number queries

For pair seeds:

`20260823, 20260824, 20260825, 20260826`

query `base`, `task`, `null` in reset-mode ordinary `predict_action`. Before every branch model call, reset the same pair seed. Save the **first** returned 36-D action and extract right arm `21:28`.

Write one JSONL row per state × pair seed with at least:

- `episode_id`, `state_id`, `pair_seed`
- `delta_task`, `delta_null`
- `base_right_arm_target`, `task_right_arm_target`, `null_right_arm_target`
- finite geometry diagnostics
- exact Psi0/SIMPLE/checkpoint revisions

Then run:

```bash
python g0_simple_psi0.py records.jsonl --out g0_result.json
```

## Frozen verdict

Primary score: `DeltaR = R_task - R_null = A_null - A_task`.

- GO: mean >=0.20 and episode-bootstrap 95% CI low >0.
- KILL: CI high <=0.10.
- else: `INCONCLUSIVE_DO_NOT_TUNE`.

Seeds/states are nested; bootstrap episodes.

## Only if GO — G0b

Repeat the paired query with an identical frozen RTC `previous_action` / history for all three branches. This confirms deployed-history robustness. It cannot rescue G0a.

## Deliverables

Commit to this folder:

- `results/g0a/records.jsonl`
- `results/g0a/g0_result.json`
- `G0_RESULTS.md`
- any minimal collector/patch needed to reproduce the run
- exact commands and upstream revisions.

In `G0_RESULTS.md`, explicitly answer:

1. Did local Psi0 competence pass?
2. Did deterministic paired inference pass?
3. How many episodes/states passed the fixed kinematic construction?
4. What are `R_task`, `R_null`, `DeltaR`, and episode-bootstrap CI?
5. Frozen verdict?
6. Any engineering deviation from the preregistered contrast?
