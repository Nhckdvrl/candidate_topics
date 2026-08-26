# Topic 24 pivot decision (2026-08-26)

> **Status: decision registered, nothing executed yet.** No new topic
> directory has been created, no `π0.5`/LIBERO code has been written, no
> Gate-0 has run. This document exists so the reasoning survives a session
> boundary. Topic 24's own history (`README.md`, `G0_RESULTS.md` through
> `G3_EXPLORATORY_LIFT_AUDIT.md`, the registered-not-executed G4) is not
> rewritten or deleted — it is the motivating discovery for what follows,
> not superseded work.

## The judgment

**PIVOT, not KILL, not continue-as-is.**

```text
Continue current Topic 24 tree (run G4, keep dissecting Psi0 mechanism)
    -> KILL for ICLR/ICML/NeurIPS scale. CloseDoor's +-0.20 reversal at
       100N is real and clean, but it is one humanoid stack, one task, one
       hierarchy split. XMoveBendPick's confirmatory G3 did not establish a
       cross-task result, and its exploratory follow-on is a hypothesis
       (p ~ 0.09-0.35 after correction), not a second result. "Which layer
       absorbs a disturbance" is also a crowded 2026 question (see below).

Treat it as a CoRL/RSS-scale robotics mechanism paper
    -> viable, but needs more cross-task evidence than currently exists.

Pivot to the counterfactual-value-of-feedback framing
    -> KEEP. Worth a Gate-0.
```

## Why the original framing does not clear top-venue scale on its own

The field has moved fast on "when should a VLA replan" in 2026:

- **Bernoulli-Continuation Policy (BCP)**, arXiv:2608.03483 — literally
  titled "Continue or Replan?"; learns a continuation head via
  trajectory-level RL with a replanning-efficiency reward, evaluated on
  RoboTwin (50 tasks), `pi0.5`, LIBERO/PRO, and real hardware.
- **Adaptive Action Chunking (AAC)**, arXiv:2604.04161 — action entropy sets
  chunk size dynamically.
- **AutoHorizon** (ECCV 2026, public code) — reads a VLA's own attention as
  a proxy for its predictive limit, on the `pi0.5`/OpenPI stack.
- **Adaptive Action Chunking via Multi-Chunk Q (ACH)** — estimates Q for
  different chunk lengths directly.
- **VLA-Corrector**, arXiv:2602.21445 ("VLA Knows Its Limits") — triggers
  truncation/replan from latent-dynamics deviation; public eval
  infrastructure (MetaWorld/LIBERO, action queue) but no released
  fine-tuned VLA/corrector checkpoints, so it is a baseline to compare
  against, not something to depend on for Gate-0.
- **Hi-VLA**, arXiv:2606.10267 — already systematically studies
  planner/controller/interface/switching design space in hierarchical VLA
  agents, directly crowding "which layer contributes robustness."
- **B2FF**, arXiv:2606.09258 — already documents that off-trajectory
  replanning *frequently destabilizes* action sequences, so "replanning can
  hurt" is not by itself a novel headline either.

So the *phenomenon* Topic 24 found (a hierarchy seam whose causal
contribution changes sign with disturbance direction) is real and clean, but
the *question* it currently sits inside — which layer absorbs a disturbance,
when should the VLA wake up — is being actively and competently worked by
several concurrent 2026 papers with public code and multi-model,
multi-benchmark evidence already in hand. Continuing to add cross-task
confirmatory panels inside that question (i.e. running the registered G4)
would, even if it succeeds, only produce "100N/right live VLA more often
knocks the object off the table" — a real but narrow addition, not a
top-venue-scale claim.

## The reframed question

> **Do replanning signals actually estimate the causal value of
> replanning?**
>
> (broader form) **The Counterfactual Value of Feedback in Vision-Language-Action Policies**

The distinction the reframing rests on:

> **Need for correction is not the same quantity as value of replanning.**

Every proxy above (entropy, attention, latent deviation, learned
continuation/Q score) estimates some version of "does this state look
abnormal / does the plan look wrong," then assumes: *state looks off ⇒
replanning helps*. That assumption is exactly what has never been directly
measured. What is actually wanted is the treatment effect:

```text
tau(h_t, c_t) = E[Y_replan - Y_continue | h_t, c_t]
```

where `h_t` is the current observation/history, `c_t` is an already-committed,
not-yet-executed action plan/chunk, `continue` executes the committed suffix,
`replan` takes a fresh observation now and regenerates the suffix, and `Y` is
final task outcome.

CloseDoor's headline number is already one instance of exactly this
quantity, not merely a suggestive analogy:

```text
tau_nav(100N, left)  = +0.20
tau_nav(100N, right) = -0.20
```

Same model, task, force magnitude, WBC — only disturbance direction differs,
and the causal value of fresh feedback flips sign. That is direct evidence
that "how abnormal does this state look" and "does handing control back to
the VLA actually help" are not the same axis — which is the reframed
project's central claim, not an assumption.

### Why this is not just "another adaptive-chunking paper"

BCP, AAC, AutoHorizon, ACH, VLA-Corrector all *infer* `tau` from a proxy,
because in the real world `Y_replan` and `Y_continue` cannot both be
observed from the same instant — only one branch is ever actually executed.
**In simulation, the same physical state can be cloned and both branches
actually run.** That gives real paired potential outcomes `(Y_R, Y_C)`
instead of a proxy for them, and licenses a question none of the above
papers can currently answer directly: **are the field's existing replanning
triggers actually calibrated to the thing they are implicitly trying to
estimate?**

## Proposed paper structure (five layers, not required to all land for the project to be worth attempting)

| layer | question |
|---|---|
| Measurement | How to directly measure `replan vs continue` counterfactual value from the same cloned state |
| Phenomenon | Is `tau` state-dependent, and often negative? |
| Audit | Do entropy / attention / deviation / learned continuation scores actually predict `tau`? |
| Mechanism | What kind of state/plan transition produces negative `tau`? |
| Method | Does direct counterfactual supervision beat proxy triggers at avoiding harmful replans? |

If only layers 1-2 land, and only on Psi0, this is not a three-venue paper.
If layers 1-4 hold on `pi0.5` + a second VLA across multiple tasks, that is
NeurIPS/ICLR-shaped even with a lightweight method. Layer 5 with a solid
method would round it out toward ICML-shaped as well.

A clean theoretical framing worth keeping in the intro: for a truly optimal
decision-maker, additional information can never have negative value,
because it can always be ignored in favor of the existing plan. Real VLA
replan operators do not have that "ignore if worse" oracle — a fresh
observation triggers plan generation that *overwrites* the existing
commitment unconditionally. So a negative `tau` is not a contradiction; it
is direct evidence that **learned replanning, as currently implemented, is
not policy improvement.**

## Proposed execution order (not started)

```text
pi0.5/LIBERO Counterfactual Gate-0
          |
          |-- FAIL --> KILL the whole reframed project, do not task-shop
          |
          `-- PASS
                |
        broad feedback-value atlas (multiple tasks, multiple models)
                |
        proxy-vs-value audit (entropy / attention / deviation / learned score)
                |
        second VLA / lightweight counterfactual-supervision method
                |
        G4 (the registered-not-executed XMoveBendPick follow-up) folded in
        LAST, as humanoid/cross-stack corroborating evidence -- not first
```

### Gate-0 (smallest possible falsification test, not yet run)

**Stack**: LeRobot `pi0.5` + LIBERO. `pi0.5` on LIBERO reports ~97.5% average
success in LeRobot's own published numbers, and `n_action_steps=10` already
creates a natural committed-action seam (a chunk boundary) to branch at.
LIBERO's `env_wrapper.py` exposes `get_sim_state()`/`set_state()`, so exact
MuJoCo state cloning is a supported capability, not something to build from
scratch.

**Branch design** (mid-chunk, e.g. after 5 of 10 committed actions):

```text
identical simulator state S
            |
       clone state
      /            \
  CONTINUE         REPLAN
  old chunk's      fresh observation -> new chunk,
  remaining 5      execute only its first 5 steps
  steps                |
      \               /
   both arms resynchronize at the ORIGINAL query boundary
   (the replan arm must not shift the entire future query
   schedule just because it queried early -- this is a local
   treatment on one committed window, not a different policy
   from that point on)
            |
     both arms continue under normal policy control
            |
        terminal outcome
```

**Gate-0 P0** (structural proof before any outcome is read, same discipline
as this project's own P0/P0b): clone-to-clone state equivalence is
bit/state-identical; the CONTINUE arm reproduces the original unbranched
rollout exactly; action-queue length, policy query count, and RNG state are
all auditable; no confound exists in the future query schedule outside the
treatment window. **No treatment outcome is inspected during P0.**

**Discovery panel** (frozen before running, not selected after seeing
results): all 10 LIBERO-Object tasks (not "whichever task shows an
effect"), `pi0.5`'s own published baseline there is a clean ~99%. Two
pre-defined branch-state families: nominal trajectory positions, and one
frozen mid-chunk physical-perturbation family (reusing an existing public
robustness-benchmark perturbation definition rather than hand-picking a
magnitude after looking at outcomes). Per branch state: `continue` success,
`replan` success, and specifically the **discordant** cells:

```text
1/1  neutral            0/0  neutral (at the success level)
0/1  replanning helped  1/0  replanning hurt
```

**Hard stop rule, decided now, not after seeing the panel**: if there are
almost no discordant states, or harmful replanning only shows up on
Psi0/CloseDoor and not here, or every meaningful perturbation on `pi0.5` is
positive/neutral for replanning — **kill the whole reframed project. Do not
go looking for a third task or model to rescue it.**

### If Gate-0 survives

- **Atlas**: expand to a second VLA (SmolVLA has a public LIBERO
  checkpoint) admitted purely on baseline competence and clean
  clone/replay-instrument pass, never on whether it shows a negative
  result. Cover multiple tasks, multiple within-chunk execution positions,
  nominal/disturbed, perturbation magnitude, both models. Report as a
  landscape (`tau > 0`, `tau ~ 0`, `tau < 0` regimes), with CloseDoor's
  `+.20/-.20` as the first motivating example rather than the only
  phenomenon in the paper.
- **Proxy audit** (the layer with the most novelty): compare action entropy
  (AAC), attention/predicted-horizon (AutoHorizon, public code, closest to
  adopt first), simple action-change/uncertainty baselines, latent-deviation
  (VLA-Corrector infrastructure, no checkpoints), and a learned continuation
  score (BCP, if code/checkpoints are reproducible) against the *actual*
  `tau`, not against final success. Metrics: `AUROC(tau>0)`,
  `AUROC(tau<0)`, harmful-replan rate among triggered states, calibration,
  and counterfactual regret `R = max(Y_R, Y_C) - Y_method_choice`. Target
  headline: *current replanning signals detect deviation, but deviation is
  poorly calibrated to whether replanning causally helps* — not "our method
  is 3% better."
- **Method** (only after the audit motivates it): a lightweight
  counterfactual-feedback critic, `tau_hat(s, c)` or `P(Y_R > Y_C)`, trained
  on the branch-derived paired labels directly rather than a trajectory-level
  sparse RL reward or a proxy target. The architecture itself should not be
  sold as novel (BCP/ACH/steering-with-an-improvement-head work already
  covers continuation/value heads) — the novelty claim is the supervision
  target: **actual measured intervention advantage, not a proxy or
  trajectory-level surrogate.**
- **G4 last**: only after the above, fold the registered (not yet run)
  XMoveBendPick `100N/right` catastrophic-transition follow-up back in as
  cross-stack (humanoid, not tabletop-arm) corroborating evidence for the
  same phenomenon — not as the project's primary evidence.

## What this decision does not do

- Does not delete or reinterpret any Topic 24 result. `G0_RESULTS.md`
  through `G3_EXPLORATORY_LIFT_AUDIT.md` stand as written.
- Does not execute anything toward the new direction yet — no repo, no
  code, no LIBERO/`pi0.5` download.
- Does not commit to running G4 at all; it remains registered and
  deliberately deferred, to be folded in last if the reframed project
  survives Gate-0.

## Candidate titles (not finalized)

- "Feedback Is Not Free: Measuring the Counterfactual Value of Replanning
  in Vision-Language-Action Policies"
- "The Counterfactual Value of Feedback in Vision-Language-Action Policies"

Either is a higher-level question than "Where Does Closed-Loop Robustness
Live in Hierarchical Robot Foundation Policies?" — that original title
names a mechanism-attribution question; these name a measurement +
phenomenon + audit + method question that spans sequential decision-making,
causal treatment effects, calibration, and test-time control, not just
humanoid robot control.
