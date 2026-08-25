# Topic 24 cross-task — P0b' result: WBC seam liveness on XMoveBendPickTeleop

**Verdict: `SEAM_LIVE` on all 3 configs tested.** Re-run fresh rather than
assumed to transfer from CloseDoor's architectural claim.

## Why this had to be re-run, not inherited

CloseDoor's P0b already established that the WBC seam is command-level
state-dependent, confined to the 12 leg + 3 waist joints (arms/hands
open-loop below the VLA seam). `Psi0DecoupledWbcAgent` is the same class
running here, so it would be tempting to assume the same result — but Topic
24 exists specifically to distinguish "the architecture implies this" from
"this was measured on this task's actual runtime path." So the identical
command-level, no-simulation-step measurement was re-run on
XMoveBendPickTeleop.

## Result

```text
config          repeat     restore    perturb       hands (L/R)         joints changed
dr-level-0:0    0.000e+00  0.000e+00  3.119e-02     0.000e+00/0.000e+00     15
dr-level-0:1    0.000e+00  0.000e+00  3.913e-02     0.000e+00/0.000e+00     15
dr-level-0:3    0.000e+00  0.000e+00  2.620e-02     0.000e+00/0.000e+00     15
```

The repeatability floor is exactly `0.0` on every config (same observation,
same command, restored state -> bit-identical output), and the restore probe
is also exactly `0.0`, so state restoration was complete and the perturbed
divergence cannot be residue. `d_perturb` is unambiguously above that floor
on all three (`separation_ratio = inf` in every row, since the floor is
exactly zero). Both hand channels are exactly `0.0` on every config — the
same open-loop signature CloseDoor found.

**The same 15 joints, every time**: 12 leg + 3 waist, matching CloseDoor's
finding exactly. This is a clean cross-task replication of a structural
claim, not a re-derivation of a new one: it is consistent with
`G1DecoupledWholeBodyPolicy.set_observation`'s comment that the upper-body
policy is open-loop and only `lower_body_policy` receives the observation —
architecture-level code that does not change between tasks, now confirmed to
behave identically at the command level on two different tasks.

## Deviation reported, not resolved by retuning after the result

`TICK_FRACTION = 0.4`, frozen and reused verbatim from CloseDoor's protocol,
was meant to land in an "approach phase" before task-relevant interaction —
checked via `approach_untouched_at_tick`. On CloseDoor this held for the
large majority of configs. Here it is **`False` on all 3 configs tested**:
by 40% into each canonical tape, `info["target"][2]` has already moved from
its initial value, meaning the grasp/lift has already begun.

This is left exactly as measured. The `TICK_FRACTION` was not adjusted after
seeing `SEAM_LIVE` — doing so, after already getting the desired verdict,
would be exactly the kind of after-the-fact tuning this project's stop rules
exist to prevent. The measurement itself does not depend on approach-phase
status: it is a single frozen-tick functional comparison with no simulation
step afterward, so mid-interaction proprioception is a perfectly valid input
to test the seam against. But it does mean XMoveBendPick's task-relevant
interaction begins earlier in its (shorter, ~220-280 step) episode than
CloseDoor's does in its (~250-450 step) one, and any push-timing rule for G3
on this task needs its own timing anchor rather than inheriting `0.4`
directly — which is exactly the right-hand/target contact anchor to be
frozen next, not a retuned tick fraction.

Records: [`records/p0b_xmove_records.jsonl`](records/p0b_xmove_records.jsonl).
