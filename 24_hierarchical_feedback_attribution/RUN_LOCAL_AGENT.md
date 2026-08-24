# RUN_LOCAL_AGENT — Topic 24 G0

You are running the frozen three-level feedback attribution panel for:

> **Where Does Closed-Loop Robustness Live in Hierarchical Robot Foundation Policies?**

Do not redesign the question, the force grid, the timing rule or the thresholds
while running it. Read [`README.md`](./README.md) first.

## Frozen upstream

```text
SIMPLE b49c1aea2dd57309bb533219d0d34d6020f3d943
Psi0   9ad917526394c1cacc72dba08562629936505987
```

`*Teleop` tasks run through the decoupled-WBC path. Do not use `eval.py` / `psi0`.

## Serve the policy

One client per server: `psi0_decoupled_wbc` sends no session id, so two
concurrent SIMPLE clients would corrupt each other's RTC state. Use a separate
port and GPU per worker.

```bash
cd $PSI0 && ./.venv-psi/bin/python serve_psi0_g0.py --host 0.0.0.0 --port 22085 \
  --policy psi0 --run-dir=$RUN_DIR --ckpt-step=40000 --action-exec-horizon=24 --rtc
```

## Run one DR level

The runner is resumable and orders work so the canonical `force=0 / fresh` pass
for a config always writes the tape before any cell that needs it.

```bash
cd $SIMPLE && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python \
  /path/to/24_hierarchical_feedback_attribution/topic24_runner.py \
  --env-id simple/G1WholebodyCloseDoorTeleop-v0 \
  --data-dir $EVAL/G1WholebodyCloseDoorTeleop-v0/dr-level-0 \
  --out records/g0_closedoor.jsonl \
  --tape-dir tapes/dr-level-0 \
  --port 22085 --sim-mode mujoco_isaac --num-episodes 10 --resume
```

Repeat for `dr-level-1` and `dr-level-2` into the same JSONL. Config ids are
namespaced by DR level, so the three levels pool into one 30-config panel.

Parallelise across DR levels, one GPU and one policy server per level. Never
point two workers at the same `--tape-dir` and the same config.

## Analyse

```bash
PYTHONPATH=. python -m pytest -q tests
PYTHONPATH=. python g0_core.py records/g0_closedoor.jsonl --out g0_result.json
```

## What a valid record looks like

```json
{"config_id": "dr-level-0:3", "condition": "vla_replay", "force_n": 100.0,
 "direction": "left", "success": false, "steps": 450, "server_queries": 0,
 "push_tick": 214, "push_applied": true, "push_applied_ticks": 10,
 "push_displacement_m": 0.081, "first_contact_tick": null,
 "door_q_min": 0.31, "clock": "virtual"}
```

Sanity conditions to eyeball while it runs, before any analysis:

- `server_queries == 0` on every `vla_replay` and `actuator_replay` row;
- `server_queries > 0` on every `fresh` row;
- `push_applied_ticks == 10` on every nonzero-force row, `0` on the control column;
- `push_tick` identical across the three conditions of a cell;
- `push_displacement_m` growing with force.

## Stop rules

Report and stop without tuning on any of:

```text
PREREQUISITE_FAIL_STRUCTURAL
PREREQUISITE_FAIL_REPLAY_FIDELITY
PREREQUISITE_FAIL_PUSH_INEFFECTIVE
INSUFFICIENT_MATCHED_CONFIGS
```

These are results about this task and this force grid, not about the hypothesis:

```text
NO_ROBUSTNESS_PHENOMENON
FRESH_COLLAPSE_NOTHING_TO_ATTRIBUTE
NO_MEANINGFUL_LEARNED_FEEDBACK_CONTRIBUTION
```

If the verdict is one of `BOTH_LEVELS_CONTRIBUTE`, `VLA_LEVEL_DOMINATES` or
`WBC_LEVEL_DOMINATES`, report the full grid, not the best cell. Do not add a
fourth condition, a second task, or another force after seeing the numbers.
