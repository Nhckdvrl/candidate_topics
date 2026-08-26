# Counterfactual Value of Feedback — execution plan

> **Status (2026-08-26): implementation plan, NOT a preregistration, NOT executed.**
> This document turns `PIVOT_DECISION.md` into a concrete engineering and
> experimental sequence. Scientific thresholds that are explicitly marked
> **TO FREEZE AFTER P0** are not yet frozen; no treatment outcome may be read
> before the structural instrument passes.
>
> The old Topic 24 experiments remain historical evidence. This plan does not
> rewrite G0–G3 and does not execute the registered XMoveBendPick G4.

## 1. Paper-level target

The new project is not "when should a VLA replan?" and not another adaptive
horizon method. The measurement target is the *local counterfactual value of
one replanning intervention*:

```text
                        same branch state s_t
                              /      \
                             /        \
                 CONTINUE old suffix   REPLAN now from fresh observation
                             \        /
                              \      /
                       resync at the ORIGINAL
                       scheduled query boundary
                              |
                       normal policy thereafter
                              |
                       terminal outcome Y
```

For a branch state `b=(h_t,c_t)`:

```text
tau(b) = E[Y_replan - Y_continue | b]
```

The first paper question is therefore:

> Are the signals currently used to decide that a plan is stale / uncertain /
> worth interrupting actually calibrated to the *causal value* of interrupting
> it?

CloseDoor G1 is the motivating discovery (`+0.20/-0.20` navigation-channel
value under matched 100N disturbances), not the benchmark on which the new
claim will stand.

## 2. Frozen upstream stack for the first gate

Use a mainstream public stack with a strong baseline and a real action-chunk
seam:

- **Policy**: LeRobot `pi0.5`, checkpoint `lerobot/pi05_libero_finetuned`.
- **Benchmark**: LIBERO, first scientific gate on the complete
  `libero_object` suite (all 10 tasks, no task shopping).
- **Execution horizon**: `n_action_steps=10`, matching the published LeRobot /
  OpenPI LIBERO evaluation.
- **Simulator**: LIBERO/robosuite MuJoCo.
- **Headless rendering**: EGL.

Pin exact LeRobot and LIBERO revisions in the implementation manifest before
P0. The current source audit (2026-08-26) verified:

1. LeRobot `PI05Policy.select_action()` fills an internal action queue by
   calling `predict_action_chunk()` when the queue is empty and executes only
   `n_action_steps` from that chunk.
2. LIBERO exposes `get_sim_state()`, `set_state()`, and
   `regenerate_obs_from_state()`.

Important: `get_sim_state()/set_state()` serializes MuJoCo state, **not a proof
that every Python-side robosuite controller state, policy queue, and RNG state
has also been restored**. The first implementation must therefore use
**deterministic prefix replay from the same initial state** as the ground-truth
branch reconstruction mechanism. Raw `set_state()` is an audit and optional
optimization only after it proves equivalent.

## 3. Core intervention semantics

### 3.1 Own the chunk schedule outside `select_action`

Do not mutate LeRobot's private `_action_queue` in the experiment. Implement an
external executor that calls `policy.predict_action_chunk(obs)` only at logical
query boundaries and stores exactly the first 10 actions itself.

Why:

- branch semantics become explicit;
- `continue` can provably use the old suffix without querying the model;
- `replan` can query once without shifting all future query boundaries;
- query counts and RNG keys can be audited directly;
- the treatment is independent of private queue implementation details.

P0 must prove that, on ordinary unbranched rollouts, this external executor is
trajectory-equivalent to `policy.select_action()` under matched query seeds.

### 3.2 The local treatment

Suppose a normal chunk is queried at logical boundary `q`, and 5 of its 10
actions have already been executed. The branch state is immediately before
executing action offset 5.

```text
old chunk = [a0 a1 a2 a3 a4 | a5 a6 a7 a8 a9]
                               ^ branch after first 5
```

Two arms:

- `continue`: execute recorded `a5..a9`, **zero policy queries** inside this
  remainder window;
- `replan`: query from the fresh branch observation once, execute exactly the
  first 5 actions of the new chunk;
- both arms then arrive at the **same original logical query time** `q+1`;
- both discard any remaining local-treatment actions and return to the normal
  10-step schedule from `q+1` onward.

The replan arm is therefore not allowed to permanently phase-shift the query
schedule. The treatment is "replace the remaining suffix of this committed
window," not "switch to a different policy schedule for the rest of the
trajectory."

### 3.3 Stateless logical-query RNG

Pi0.5 action generation is stochastic. An early treatment query must not
silently consume RNG and thereby change every later nominal query.

Implement policy queries under deterministic *logical query keys*:

```text
seed_normal(episode_id, logical_query_id)
seed_treatment(episode_id, branch_id, "early_replan")
```

Every policy call runs inside a `temporary_rng(seed)` context manager that:

1. snapshots Python `random`, NumPy, Torch CPU, and all CUDA RNG states;
2. sets the deterministic query seed;
3. performs exactly one `predict_action_chunk()`;
4. restores the caller's RNG states on exit.

Thus the same future logical query in the two counterfactual arms receives the
same sampling seed even if one arm made an extra early query. This is common
random-number coupling, not a call-count confound.

The implementation must log `logical_query_id`, `query_seed`, query count, and
whether the query is `normal` or `treatment` for every call.

## 4. Branch-state reconstruction: prefix replay, not naive state cloning

Each canonical episode first records:

- initial simulator state after normal environment initialization;
- language/task identity;
- all policy query boundaries;
- each 10-step committed chunk;
- every actually executed environment action;
- simulator state and observation hash at every candidate branch tick;
- terminal success and terminal tick.

For each counterfactual arm:

1. reset the environment normally;
2. restore the *canonical initial simulator state*;
3. replay the exact recorded environment-action prefix up to the branch;
4. compare the reconstructed MuJoCo state and observation against the canonical
   branch record;
5. only if they match, apply the treatment arm.

Because the robosuite controller is driven through the same action prefix from
its normal reset state, its Python-side internal evolution is reconstructed by
execution rather than assumed serializable.

Any branch whose prefix reconstruction does not meet the frozen P0 tolerance is
invalid; it must never silently fall back to `set_state()`-only restoration.

## 5. Perturbation family for Gate-0

The first gate needs one model-agnostic off-trajectory family that does not
require task-specific object teleport rules.

Use a short **mid-chunk actuation stall** immediately before the branch:

```text
for d control ticks before branch:
    executed_action[:6] = 0      # no delta-pose motion
    executed_action[6]  = canonical_action[6]  # preserve gripper command
```

At LIBERO's 20 Hz control rate, candidate engineering durations are
`d in {1,3,5}` ticks (50/150/250 ms).

### Engineering-only calibration (no treatment outcomes)

Before scientific Gate-0, on a predeclared tiny calibration set, run only
canonical-vs-stalled prefix reconstruction. **Do not run continue/replan
outcomes.** Select the smallest `d` whose branch state produces a physically
nontrivial robot-state deviation from the unperturbed canonical branch, using a
pre-frozen kinematic criterion (recommended: median end-effector positional
shift >= 1 cm across the calibration set, with no simulator invalidity).

The selected duration is then frozen before any `Y_replan/Y_continue` is read.
If no candidate duration is physically effective, Gate-0 stops as an
instrument failure; do not invent a new perturbation after looking at treatment
outcomes.

The scientific panel includes both:

- `nominal`: no stall;
- `stall`: the single frozen effective duration.

This separates harmful replanning caused merely by overwriting a coherent
commitment from harmful replanning specifically exposed by an off-trajectory
state.

## 6. Proposed code layout

Keep this under Topic24 until Gate-0 establishes that the pivot deserves its own
full candidate directory:

```text
24_hierarchical_feedback_attribution/
  counterfactual_feedback/
    README.md
    MANIFEST.json
    gate0/
      config.py
      rng.py
      policy_adapter.py
      env_adapter.py
      chunk_executor.py
      canonical_recorder.py
      prefix_replay.py
      perturbations.py
      branch_runner.py
      records.py
      gate0_core.py
      run_canonical.py
      run_branches.py
      evaluate_gate0.py
      tests/
        test_rng.py
        test_chunk_executor.py
        test_prefix_replay.py
        test_local_treatment.py
        test_records.py
        test_gate0_core.py
    records/
      .gitkeep
```

### `rng.py`

```python
@dataclass(frozen=True)
class QueryKey:
    episode_id: str
    logical_query_id: int
    kind: Literal["normal", "treatment"]
    branch_id: str | None = None

@contextmanager
def temporary_rng(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
```

Use a stable hash (not Python's randomized `hash()`) to derive 64-bit seeds
from `QueryKey`.

### `policy_adapter.py`

Responsibilities:

- preprocess exactly as LeRobot eval does;
- expose `query_chunk(obs, query_key) -> ChunkResult`;
- call `predict_action_chunk()` exactly once per query;
- truncate to the first 10 executable actions;
- record query seed, inference metadata, and optional proxy features later;
- assert Pi0.5 memory options are disabled for Gate-0, or explicitly serialize
  them before expanding to a memory-enabled policy.

### `chunk_executor.py`

Minimal state machine:

```python
@dataclass
class ExecutionCursor:
    logical_query_id: int
    offset: int              # 0..9 inside committed chunk
    chunk: np.ndarray        # [10, action_dim]

class ChunkExecutor:
    def normal_step(obs): ...
    def committed_suffix(): ...
    def advance(action): ...
```

The normal executor must be behaviorally identical to LeRobot's 10-action
queue semantics before any branch code is used.

### `canonical_recorder.py`

Record one normal episode into an immutable tape. Candidate scientific branch
points are selected **only from this tape, before branch outcomes exist**.

Recommended first gate: three within-episode progress locations per successful
canonical episode, nearest valid `offset=5` branch to 25%, 50%, and 75% of the
canonical terminal time. This yields early/middle/late states without looking
at treatment outcomes or task-specific semantic phases.

Store both the desired fraction and the resulting logical `(query_id, offset)`.
If two fractions map to the same branch, deduplicate deterministically and log
it.

### `prefix_replay.py`

`reconstruct_branch(tape, branch_spec)` must return a branch only if:

- reconstructed simulator state matches canonical state under P0 tolerance;
- observation/state vectors match;
- image hashes match (or exact pixel arrays if deterministic rendering proves
  stable);
- elapsed control step equals the canonical tick;
- no policy query was made during replay.

### `perturbations.py`

Implement only `none` and `actuation_stall` for Gate-0. No menu of ad hoc
perturbations in the scientific runner.

### `branch_runner.py`

Pseudo-code:

```python
def run_pair(tape, branch, perturbation):
    pre = reconstruct_branch(tape, branch)
    pre = apply_shared_perturbation_prefix_if_needed(pre, perturbation)

    continue_env = reconstruct_identical_perturbed_branch(...)
    replan_env   = reconstruct_identical_perturbed_branch(...)
    assert_same_branch_state(continue_env, replan_env)

    y_c = run_local_arm(
        env=continue_env,
        local_actions=tape.old_chunk[5:10],
        treatment_query=False,
        resync_query_id=branch.query_id + 1,
    )

    fresh_chunk = policy.query_chunk(
        obs=replan_env.obs,
        query_key=treatment_key(branch),
    )
    y_r = run_local_arm(
        env=replan_env,
        local_actions=fresh_chunk[:5],
        treatment_query=True,
        resync_query_id=branch.query_id + 1,
    )

    # From the original boundary onward both arms call the normal executor
    # with identical logical query seeds.
    return paired_record(y_continue=y_c, y_replan=y_r, ...)
```

Do not execute the two arms concurrently in one vector env unless P0 later
proves vectorization preserves exact per-arm RNG and reset semantics. Start
serial and correct; optimize only after equivalence tests pass.

## 7. P0: structural instrument gates (must pass before scientific outcomes)

P0 is the next actual work item.

### P0-A — external chunk executor equivalence

On at least 3 episodes from 2 LIBERO-Object tasks:

- LeRobot `select_action()` baseline vs external `ChunkExecutor`;
- same query seeds;
- same environment initialization;
- require identical 10-step action sequences at every normal query;
- require identical simulator trajectory (tight numeric tolerance; exact if
  observed to be bit-identical) and terminal success.

Fail => fix executor, no scientific run.

### P0-B — stateless RNG / call-count invariance

Unit test with a fixed observation:

1. normal query at logical id `q+1` -> chunk A;
2. reset RNG caller state;
3. make an extra treatment query under its own key;
4. normal query again at logical id `q+1` -> chunk B;
5. require A == B.

Also assert the caller's Python/NumPy/Torch/CUDA RNG states are restored after
`query_chunk()`.

### P0-C — prefix replay equivalence

Reconstruct the same branch twice from reset + canonical initial state + action
prefix. Require matching:

- MuJoCo flattened state;
- low-dimensional observation/state;
- rendered image hashes/pixels under EGL;
- success flag at branch;
- control tick;
- no model query during replay.

Then compare reconstructed branch to the original canonical branch record.

Fail => do not use raw state-cloning as a shortcut; fix replay/reset semantics.

### P0-D — continue arm is the identity treatment

With `perturbation=none`, branch at offset 5 and execute the old suffix. At the
original next query boundary, the continue arm must match the corresponding
canonical rollout state. With the same future logical query seeds, the rest of
the continue rollout must reproduce canonical terminal outcome.

This is the single most important replay-fidelity proof.

### P0-E — local replan really is local

Instrument a replan arm and assert:

- exactly one extra policy query occurs inside the treatment window;
- it occurs at the branch tick;
- only 5 actions from that treatment chunk are executed;
- next normal query is still at original boundary `q+1`;
- all later logical query ids/schedules match the continue arm;
- query seed at each later logical boundary is identical between arms.

### P0-F — perturbation liveness

Engineering calibration only. Verify selected stall creates the pre-frozen
physical state-deviation criterion. Do not inspect branch terminal outcomes.

P0 result should be a standalone `P0_RESULTS.md`. Only after all structural
checks pass should a scientific Gate-0 preregistration be frozen.

## 8. Gate-0 scientific panel (draft to freeze AFTER P0, before outcomes)

### Population

- complete `libero_object` suite: 10 tasks;
- first 10 official evaluation initializations per task, matching the standard
  published evaluation count;
- canonical baseline run first for every task/init;
- canonical failures are reported and are not replaced by a different init;
- branch states are constructed only for canonical-success episodes because
  the estimand is whether replanning helps or hurts a *currently competent
  policy trajectory*, not whether an already-failed nominal run can be
  rescued.

Eligibility uses only the canonical run, never branch outcomes.

### Branch locations

For each eligible canonical episode:

- progress fractions `{0.25, 0.50, 0.75}` of canonical terminal tick;
- map each to the nearest valid offset-5 point in the fixed 10-step schedule;
- deterministic deduplication if necessary.

### State families

- `nominal`;
- `stall` using the single P0-calibrated duration.

Maximum intended scale if all 100 canonicals succeed:

```text
100 episode configs x 3 branch positions x 2 state families
= 600 paired branch states
= 1200 treatment-arm rollouts
```

The independence unit is the original task/init episode config, **not each
branch state**. All uncertainty calculations cluster over that unit.

### Primary outcome and paired label

Use unmodified LIBERO terminal success:

```text
Y_C, Y_R in {0,1}
delta = Y_R - Y_C

continue replan   label
   1       1      neutral-success
   0       0      neutral-failure
   0       1      HELPED_BY_REPLAN       (+1)
   1       0      HARMED_BY_REPLAN       (-1)
```

Do **not** reduce the panel to `mean(delta)` alone. Topic24 already showed why
signed effects can cancel.

Primary descriptive quantities:

```text
p_help  = P(delta = +1)
p_harm  = P(delta = -1)
p_disc  = P(|delta| = 1)
net     = E[delta]
regret(always_replan)   = p_harm
regret(always_continue) = p_help
oracle_headroom         = p_disc
```

Report these overall and for the pre-specified factors `nominal/stall` and
`25/50/75%`. Task-level results are fully reported but are not individually
searched for a rescue claim.

### Inference

- cluster bootstrap over the 100 task/init episode configs, keeping all six
  potential branch states from a config together;
- paired randomization/sign tests for global help-vs-harm imbalance where
  appropriate;
- if cellwise intervals are reported for the 2x3 pre-specified grid, use a
  simultaneous/max-T correction rather than six uncorrected 95% CIs;
- seed and number of bootstrap/permutation draws frozen in the preregistration.

### Hard scientific stop rule (TO FREEZE AFTER P0)

The preregistration must contain a quantitative rule before any branch outcome
is read. Recommended minimum to justify the full project:

```text
KEEP only if harmful replanning is nontrivial and not a one-task curiosity:

1. at least 5% of valid perturbed branch states are HARMED_BY_REPLAN,
2. at least 10 independent task/init episode configs contain >=1 harmful state,
3. harmful states occur in at least 2 distinct LIBERO-Object tasks,
4. structural / canonical-competence prerequisites pass.

Otherwise: KILL the counterfactual-feedback project.
Do not shop another LIBERO suite, perturbation, task, or model to rescue Gate-0.
```

This 5%/10-config rule is a *recommended draft*, not yet frozen. It may be
adjusted for power only before P0 is converted to the scientific
preregistration; once Gate-0 branch outcomes exist, it is immutable.

A positive average net benefit does **not** defeat the hypothesis if a real
harmful subpopulation exists. Conversely, one or two spectacular harmful
videos do not pass the gate.

## 9. If Gate-0 passes: build the Counterfactual Feedback Atlas

Only then create a dedicated new candidate/project directory.

### Axis A — benchmark breadth

Expand Pi0.5 from LIBERO-Object to all four standard suites:

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-10

Admission is based on baseline competence and instrument fidelity, never on
whether a suite produces negative tau.

### Axis B — second VLA

Add a second public VLA only if:

1. public checkpoint exists;
2. baseline competence is sufficient;
3. the same committed-suffix treatment is semantically valid;
4. a model-specific P0 passes.

SmolVLA/LIBERO is the first practical candidate. Do not add a second model
merely because it produces the desired sign.

### Axis C — within-chunk staleness

After the one-offset Gate-0, expand treatment offset to a pre-specified small
set such as `{2,5,8}` of a 10-action execution window. This directly tests
whether counterfactual value depends on how much committed plan remains.

### Axis D — disturbance families

After the universal actuation stall, add at most one qualitatively different
physical family (e.g. task-object displacement) defined by simulator geometry
and calibrated without treatment outcomes. Do not build a perturbation zoo.

### Atlas outputs

The central figure should be a **feedback-value landscape**, not a leaderboard:

```text
where is tau > 0 ?   replanning genuinely helps
where is tau ~ 0 ?   replanning is wasted compute / irrelevant
where is tau < 0 ?   replanning causally hurts
```

Report help/harm separately, by model, suite, stale offset, and perturbation.

## 10. Proxy-vs-value audit — the main novelty layer

Do not start this until real paired counterfactual labels exist.

For every already-collected branch state, compute trigger/proxy scores **without
changing the environment outcome used to define tau**.

Priority order:

1. **AutoHorizon** attention-derived horizon signal — public Pi0.5/LIBERO code,
   easiest first external baseline.
2. simple action-delta trigger — AutoHorizon already includes this baseline.
3. sampling uncertainty — AutoHorizon includes an uncertainty baseline; use
   matched sample count and log its extra policy cost.
4. AAC-style action entropy if its exact estimator is reproducible on Pi0.5.
5. VLA-Corrector latent-dynamics deviation only after training its released
   corrector infrastructure; its repo does not release the trained corrector
   checkpoint, so it must not block the main audit.
6. BCP continuation score only if code/checkpoint is publicly reproducible by
   the time this stage starts.

Do not define success of a proxy as "it improves average task success." Define
it against measured counterfactual value:

- `AUROC(delta > 0)` — predicts when replanning helps;
- `AUROC(delta < 0)` — predicts harmful replanning;
- precision/recall for harmful states;
- calibration of predicted benefit;
- harmful-replan rate among states the proxy would trigger;
- counterfactual decision regret:

```text
regret(method) = max(Y_R, Y_C) - Y_method_choice
```

For neutral states both choices have zero regret. This makes the audit directly
about whether a trigger chooses the better intervention, not whether its proxy
correlates with generic deviation.

The desired paper-level empirical statement is something like:

> Existing signals are good at detecting staleness/deviation, but are poorly
> calibrated to the causal value of handing control back to the VLA.

Only claim this if the audit actually supports it across pre-specified models
and tasks.

## 11. Mechanism analysis — after the atlas, not before

Do not inspect individual harmful states to invent a mechanism before the
population-level phenomenon is established.

Once established, pre-specify candidate explanations with distinct predictions.
Examples:

1. **mode switching / commitment break**: fresh query jumps to another plausible
   action mode inconsistent with physical progress;
2. **observation aliasing after disturbance**: new state is visually plausible
   but underdetermines the correct recovery action;
3. **remaining-plan recoverability**: the old suffix would still succeed because
   downstream dynamics absorb the deviation, while a new plan over-corrects;
4. **gripper/contact phase discontinuity**: replanning during contact produces a
   chunk whose first action is incompatible with the current contact state.

Use direct action/trajectory interventions to distinguish these if possible;
do not turn this into a latent-probe paper.

## 12. Method — only if the audit creates a real opening

Train a small **Counterfactual Feedback Critic (CFC)** on branch-derived labels.
The architecture is not the novelty.

Candidate target:

```text
p_help(b) = P(Y_R > Y_C | b)
p_harm(b) = P(Y_R < Y_C | b)
```

or an advantage estimate `tau_hat(b)`.

Input should include:

- fresh observation / frozen VLA visual features;
- representation of the committed suffix `c_t`;
- current offset inside the execution horizon;
- optionally policy uncertainty features.

The critical supervision distinction from BCP/AAC/AutoHorizon-style methods is:

> labels come from directly observed paired intervention outcomes in simulation,
> not from trajectory-level reward, entropy, attention, or deviation proxies.

Evaluation must use held-out episode configs and preferably held-out tasks.
Compare under a policy-call budget:

- fixed horizon;
- always replan at candidate branch points;
- AutoHorizon / available adaptive baselines;
- oracle counterfactual chooser (upper bound);
- CFC.

Report success, calls, harmful-replan rate, and counterfactual regret together.

## 13. Where the old humanoid G4 fits

The registered XMoveBendPick `100N/right` catastrophic-failure follow-up stays
**last**.

If the Pi0.5/second-VLA project survives and establishes a general feedback-value
phenomenon, then G4 becomes useful as a qualitatively different hierarchy:

- tabletop action-chunk policy: continue committed chunk vs fresh replan;
- humanoid hierarchy: replay committed VLA command vs live VLA with live WBC.

A successful G4 would then support cross-stack generality. Before that, it is
just another expensive SIMPLE result and must remain unexecuted.

## 14. Exact next work order

```text
NOW
  1. create `counterfactual_feedback/gate0/` scaffold
  2. pin LeRobot/LIBERO/checkpoint manifest
  3. implement `temporary_rng` + tests
  4. implement external Pi0.5 chunk executor + select_action equivalence test
  5. implement canonical tape recorder
  6. implement prefix replay + branch-state equality checks
  7. implement local continue/replan treatment
  8. pass P0-A..E
  9. implement/calibrate actuation stall WITHOUT treatment outcomes
 10. write P0_RESULTS.md

THEN, AND ONLY THEN
 11. freeze Gate-0 preregistration, including exact task/init list,
     perturbation duration, branch mapping, bootstrap seed, and hard stop rule
 12. run canonical 10x10 LIBERO-Object population
 13. generate frozen branch manifest from canonical tapes
 14. run all paired branch arms
 15. evaluate once with frozen `gate0_core.py`

IF GATE-0 FAILS
 16. archive result and KILL pivot; no task/model/perturbation shopping

IF GATE-0 PASSES
 16. expand atlas
 17. add AutoHorizon/action-delta/uncertainty proxy audit
 18. add second VLA
 19. only then decide whether a counterfactual critic is justified
 20. G4 humanoid corroboration last
```

## 15. What not to do

- Do not run XMoveBendPick G4 first.
- Do not train a replanning head before measuring paired potential outcomes.
- Do not call `set_state()` alone "exact cloning" without P0 evidence.
- Do not let the early replan shift every future query boundary.
- Do not let the early query consume RNG and alter all future sampling.
- Do not select LIBERO tasks after seeing which tasks contain harmful replans.
- Do not tune perturbation magnitude/duration on `Y_R-Y_C`.
- Do not average positive and negative discordant states into one net score and
  conclude "replanning has little effect."
- Do not turn individual catastrophic videos into a mechanism claim before the
  population-level gate passes.
- Do not sell the eventual method as "a new continue/replan head"; that space is
  already crowded. The novelty is direct counterfactual measurement and
  supervision/audit against that measurement.
