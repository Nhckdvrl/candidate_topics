# Feedback-Source Attribution Prototype

This directory is deliberately a **pre-registration instrument test**, not Topic 24 yet.

Candidate question:

> When a hierarchical robot foundation policy recovers from a physical disturbance, which feedback layer actually caused the recovery?

## Upstream target

- Psi0 + SIMPLE
- `simple/G1WholebodyCloseDoorTeleop-v0`
- same released checkpoint/config panel already validated by archived Topic 23

The deployed stack exposes two distinct seams:

```text
VLA -> WBC       : vla_cmd
WBC -> actuator  : target_q / left_hand_q / right_hand_q
```

## P0 only: no external perturbation

For 10 fixed CloseDoor configs:

1. run canonical live policy and record both command tapes;
2. reset to the same realised settled simulator state and replay the recorded VLA-level tape through the live WBC;
3. reset again and replay the recorded post-WBC actuator-reference tape;
4. write one result row per condition with `force_n=0`.

Frozen technical pass:

```text
fresh >= 0.90
vla_replay >= 0.90
actuator_replay >= 0.90
fresh - vla_replay <= 0.10
fresh - actuator_replay <= 0.10
```

Do not apply pushes, inspect attribution gaps, change tasks, or tune any scientific threshold before this passes.

## What to record

The canonical recorder must capture every executed control tick:

```text
VLA seam
  target_upper_body_pose
  navigate_cmd
  base_height_command

post-WBC seam
  target_q
  left_hand_q
  right_hand_q
```

Also persist:

- config id / DR level;
- exact settled MuJoCo-state hash before policy start;
- command count and control timestamps;
- official SIMPLE success;
- raw door joint trajectory.

A replay condition must consume the same number of tape rows at the same cadence. Exhausting early or leaving unused rows is a P0 failure.

## Files

- `p0_replay_contract.py` — lossless command-tape schema and serialization checks.
- `g0_core.py` — P0 fidelity scoring plus the later three-level feedback attribution statistics.
- `tests/test_core.py` — pure-logic tests.

## Local logic tests

```bash
cd embodied_topic_search/prototypes/feedback_source_attribution
PYTHONPATH=. pytest -q
```

Current pre-push result: `6 passed`.

## Only after P0 passes

Register the question as a root Topic, then freeze the physical-disturbance G0. The provisional candidate document currently specifies a non-selected force grid `{50,100,150} N x {left,right} x 0.2 s`, with perturbation timing derived only from each config's unperturbed canonical trajectory.
