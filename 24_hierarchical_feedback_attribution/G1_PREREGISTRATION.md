# Topic 24 — G1 pre-registration: which VLA command channel causes the reversal?

> Written and committed **before any G1 result was read**. The operating point,
> the four conditions, the stop rules and the interpretation table below are
> frozen. Only `LR` and `RL` are new data; `RR` and `LL` are reused G0 rows.

## What G0 left open

G0's pooled `delta_high` was 0.017, CI `[-0.050, 0.083]` — indistinguishable
from zero. But the six per-cell values are not individually small, and they are
not noise around zero: they are **sign-split by push direction**.

```text
delta_high      left        right
50N            +0.300      -0.100
100N           +0.233      -0.200
150N           +0.067      -0.207
```

All three `left` cells positive, all three `right` cells negative. The pooled
near-zero is these two patterns cancelling. Under a `right` push, replaying the
pre-disturbance VLA plan *beats* letting the live VLA re-observe and re-plan.

The question G1 asks is not "is the VLA good or bad" — it is which part of the
VLA's command carries that sign flip.

## The seam is already factorable

A `vla_cmd` is not atomic. It has two independently addressable parts:

```text
navigate_cmd, base_height_command      navigation / base channel
target_upper_body_pose                 upper-body channel
```

So the same intervention philosophy that made G0 work — cut a seam that exists
in the released source, do not probe latents — applies one level down.

## Frozen conditions

| condition | navigation/base | upper body | source |
| --- | --- | --- | --- |
| `RR` | replay | replay | **reused**: G0 `vla_replay` @100N |
| `LR` | **live** | replay | new |
| `RL` | replay | **live** | new |
| `LL` | **live** | **live** | **reused**: G0 `fresh` @100N |

`RR` and `LL` are not re-collected. G0's `vla_replay` replays the entire
recorded tape, i.e. both channels — that is exactly `RR`. G0's `fresh` runs both
live — that is `LL`. The reuse is physically exact, not approximate: in both
cases the command reaching the whole-body controller is byte-identical, the
observation is untouched, and the virtual clock advances once per control tick.
Verified: reusing them reproduces G0's `delta_high` exactly (`LL - RR` =
`+0.233` left, `-0.200` right).

This halves the cost: 30 configs x 2 directions x 2 new conditions = **120
rollouts**, not 240.

## Frozen operating point: 100N, both directions

Chosen **because it is the cleanest diagnostic point, not because it looks
best**:

- it avoids the ceiling at 50N (`fresh` 0.833 left) and the floor at 150N
  (`fresh` 0.067 left, 0.103 right), so both channels have room to move the
  outcome in either direction;
- it is the only force where both directions carry a large effect of *opposite
  sign* (`+0.233` / `-0.200`), which is the phenomenon being decomposed.

The force is not changed after seeing G1 data. If 100N turns out uninformative,
that is the reported result.

Push tick, configs and seeds are inherited unchanged from G0 — G1 reads each
config's `push_tick` straight out of the G0 tape, so the disturbance is the
identical event G0 already measured the reversal under (verified: `push@82` on
`dr-level-0:0` in both).

## Interpretation, fixed in advance

| observed | reading |
| --- | --- |
| `RR ≈ RL` and `LR ≈ LL` | the sign flip lives in the **navigation/base** channel |
| `RR ≈ LR` and `RL ≈ LL` | it lives in the **upper-body** channel |
| both single-channel effects real | both channels contribute independently |
| neither single channel real, but `LL` differs | **cross-channel interaction**: independently refreshed base and arm references become mutually inconsistent after the disturbance |

The last outcome is the most interesting one and is written down now precisely
so it cannot later look like a post-hoc story: it would say hierarchical VLA
recovery can fail not because the planner or the controller is individually
wrong, but because two independently refreshed command channels stop agreeing.

## Stop rules

```text
PREREQUISITE_FAIL_STRUCTURAL     a hybrid row did not query the live VLA; a
                                 channel claimed replayed was not overwritten on
                                 every control tick, or one claimed live was
                                 overwritten at all; RR contacted the server;
                                 any row not at 100N
INSUFFICIENT_MATCHED_CONFIGS     either direction has <24 of 30 complete
                                 RR/LR/RL/LL quadruples
```

Minimum worthy effect: `0.10`, same as G0. An effect counts only if its
clustered-bootstrap 95% CI excludes zero **and** its point estimate clears that
bar.

## The intervention proves itself, per P0b discipline

Every new row records `nav_overwrites`, `upper_overwrites` and
`server_queries`. A condition claiming to replay a channel must show it
overwrote that channel on **every** control tick, and a channel claiming to be
live must show zero overwrites. This is checked structurally before any success
rate is read — the same rule that made P0b a separate gate rather than an
assumption. Confirmed on the first real rollouts:

```text
LR   q=19  nav_overwrites=0    upper_overwrites=450
RL   q=19  nav_overwrites=450  upper_overwrites=0
```

## Files

- `g1_channel_factorization_runner.py` — collects `LR` and `RL`.
- `g1_merge_records.py` — folds in G0's `RR`/`LL` rows.
- `g1_core.py` — the frozen decision procedure.
- `tests/test_g1_core.py` — pure-logic tests (`15 passed`).
