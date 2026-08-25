# Topic 24 cross-task — canonical reconnaissance (XMoveBendPickTeleop)

No perturbation anywhere in this file. 30 canonical rollouts (`dr-level-0/1/2`
x episodes 0-9), observing only: success, episode length, and the first tick
of right-hand <-> target physical contact. This freezes the eligible panel and
the push-timing anchor for G3 **before any push-condition data exists**.

## Instrument, fixed before this ran

```text
target_body_id   env.unwrapped.mujoco.mj_objects["target"].id
                  (the accessor the engine itself built the body with;
                   cross-checked against mj_name2id(model, BODY, target.asset.label))
right_hand        full kinematic subtree rooted at right_wrist_roll_link
                  (elbow -> wrist_roll -> wrist_pitch -> wrist_yaw -> fingers,
                   confirmed against data/robots/g1/g1_29dof_with_dex3.xml)
contact           any MuJoCo contact pair with one geom in that subtree and
                  the other belonging to target_body_id, exactly
anchor            push_tick = first_contact_tick - round(1.0 / control_dt)
control_dt        read live from sonic_config: 0.02 s (50 Hz), not assumed
```

## Eligible panel: 28/30

```text
dr-level-0: 10 configs, 9 eligible
dr-level-1: 10 configs, 9 eligible
dr-level-2: 10 configs, 10 eligible
```

Two configs are `timing_ineligible` — no right-hand/target contact occurred
in the unperturbed canonical rollout at all (both ran to the full 800-step
horizon):

```text
dr-level-0:2   success=False  steps=800  first_contact=None
dr-level-1:2   success=False  steps=800  first_contact=None
```

**These are excluded from every G3 force/direction/condition cell and are not
replaced.** `dr-level-0:2` is the same config that failed in all three P0'
conditions with byte-identical trajectories — the instrument already showed
this is a deterministic, reproducible canonical failure, not noise. No other
configs were substituted in to keep the panel at 30; the frozen support floor
(`>=24/30`) is checked against the 28 that exist, not against a padded 30.

28/30 clears the floor with margin, at both the pooled level and independently
within every DR level (9, 9, 10).

## Timing distribution across the 28 eligible configs

```text
first_contact_tick   min 164   median 204.0   max 253
push_tick             min 114   median 154.0   max 203
episode steps         min 218   median 233.5   max 800  (800 only for the two ineligible configs)
```

Every `push_tick` is positive; no config needed the `push_tick < 0` exclusion
rule in practice, though the rule stays frozen in `G3_PREREGISTRATION.md` for
any future re-run on a different task or checkpoint where it might trigger.

## Canonical success, reported but never used to filter eligibility

```text
dr-level-0: 9/10 canonical success (9 eligible, 9 succeed)
dr-level-1: 6/10 canonical success (9 eligible, 6 succeed)
dr-level-2: 7/10 canonical success (10 eligible, 7 succeed)
```

Six eligible configs make right-hand/target contact but do not complete the
task — `dr-level-1:{1,3,8}` and `dr-level-2:{1,6,9}`. These stay in the panel.
Excluding them would silently redefine the evaluation population as "scenes
where the released checkpoint already succeeds," which is exactly the
selection this reconnaissance was designed to avoid: G3's own force=0 control
column already measures baseline competence directly, so there is no need to
pre-filter for it here.

The cross-DR canonical success rate here (9\|6\|7 out of 10, or 9\|6\|7 out of
the eligible 9\|9\|10) does not match the published `10\|9\|9` benchmark
number. This is recorded rather than investigated further — consistent with
P0's finding that `dr-level-0:2` already deviated from the published number
under this local setup, for reasons out of scope for this reconnaissance.

Records: [`records/canonical_reconnaissance.jsonl`](records/canonical_reconnaissance.jsonl)
(30 rows).
