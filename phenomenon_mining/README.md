# Phenomenon Mining

This directory is a **phenomenon-first research search workspace**. It is deliberately separated from `master_topic_search` and from numbered candidate topics.

The purpose is to stop generating elegant hypotheses first and then asking whether nature/model behavior happens to cooperate. We instead search for **already-visible, reproducible, high-signal behavioral or training phenomena**, replicate them in an accessible system, and only then decide whether a mechanism or method paper exists.

## Why this workspace exists

The candidate repository has accumulated repeated failures of the following form:

```text
real observation A
+ plausible explanatory relation B
+ clean-looking experiment
-> B is weak, absent, opposite, or not identifiable
```

This happened even when the seed phenomenon was strong. The current process therefore treats the repository's archived topics as evidence that **hypothesis-first topic generation has a poor base rate for us**.

A result may enter a numbered topic only after this workspace has already established a phenomenon worth explaining.

---

# 1. Research-room fit

The lab-neighborhood map in `master_topic_search` strongly favors concrete scientific objects and dynamics rather than generic benchmark engineering. The highest-fit lanes for this mining workspace are therefore:

1. **LLM training / post-training dynamics**
   - open checkpoints;
   - non-monotonic acquisition or collapse;
   - abrupt behavioral shifts;
   - sparse training signals;
   - capability redistribution across training.

2. **Reasoning behavior and mechanisms**
   - branching, backtracking, verification, correction;
   - behavior visible in traces before hidden-state explanation;
   - checkpoint-to-checkpoint changes with direct behavioral observables.

3. **Agentic RL / interaction learning**
   - action vs reasoning vs observation learning signals;
   - tool-feedback use;
   - recovery after failed actions;
   - behavior changes across RL checkpoints;
   - what interaction experience changes even when environment tokens are not ordinary policy-loss targets.

4. **Continual / self-improving learning**
   - acquisition, forgetting, interference, adaptation;
   - especially strong non-monotonic or asymmetric phenomena observable directly in behavior.

5. **Embodied / VLA behavior**, only when the phenomenon is direct and mechanism-like
   - correction/recovery latency;
   - action-chunk inertia;
   - perturbation responses;
   - behavior under task-equivalent vs task-changing deviations;
   - no generic benchmark-chasing.

The workspace explicitly does **not** prioritize linguistic annotation topics, generic RAG, generic agent harnesses, or "model X on dataset Y" projects.

---

# 2. Admission rule: no hypothesis-first entries

A mining lead must start from at least one of:

- a published curve, trace, table, checkpoint sequence, or failure example showing a non-trivial phenomenon;
- the same unusual behavior reported independently by multiple papers or practitioners;
- a released run / checkpoint family where the phenomenon can be inspected directly;
- a documented industrial/practitioner observation with enough technical detail to reproduce;
- a direct contradiction between results on the **same object**.

The following are not enough:

```text
paper A says X
paper B says Y
maybe X causes Y
```

```text
representation X exists
maybe it controls behavior Y
```

```text
we can define a clever counterfactual
maybe a difference appears
```

Such ideas can be noted as questions, but they cannot be promoted until a phenomenon is already visible.

---

# 3. What counts as a strong phenomenon

Preferred shapes:

- abrupt spike / collapse / recovery during training;
- a stable non-monotonic trajectory across checkpoints;
- a qualitative strategy switch that can be read directly from traces;
- a repeated local failure hidden by rising aggregate accuracy;
- a behavioral capability that improves while another predictably degrades;
- a model becoming more reward-sensitive / shortcut-seeking / rigid as optimization proceeds;
- a large disagreement between nominally equivalent training signals;
- action / observation / reasoning tokens carrying sharply different learning relevance;
- a robot policy failing to react to fresh observations because its action horizon commits it to stale decisions;
- a foundation policy correcting some perturbations but systematically ignoring other equally large ones;
- a phenomenon that appears only after entering a meaningful capability regime and then persists.

Weak shapes:

- tiny correlation;
- one best layer / step / threshold found by sweep;
- toy-only effect with no competent realistic regime;
- an effect whose label requires a large LLM-judge ontology;
- a phenomenon that only appears after filtering to a bespoke subset.

---

# 4. Three-stage workflow

## Stage P0 — Observation inventory

Collect sources and record only:

```text
What was directly observed?
Where?
How large / stable was it?
Is code/checkpoint/data available?
Why is it surprising?
```

No mechanism claim is required.

## Stage P1 — Cheap replication

Reproduce the observation using the closest released setup or an accessible equivalent.

Promotion requires:

- direct observable;
- no large hyperparameter search;
- meaningful regime;
- effect large enough to matter;
- ideally two models / seeds / environments when cheap.

## Stage P2 — Paper-shape audit

Only after P1:

```text
What exactly changes?
What explanatory alternatives remain?
Can one clean intervention separate them?
What method lever is exposed?
Would the positive result excite a strong audience?
```

Only then can a lead become `topicXX_*`.

---

# 5. Hard anti-failure rules learned from Topics 01–16

1. **No cross-paper empty-cell projects.**
2. **No mechanism before behavior.** The exact event to be explained must first occur at useful density.
3. **No latent noun without a direct observable.** "strategy", "route", "awareness", "internalized knowledge", etc. require exceptional care.
4. **No LLM judge as the core y-axis** unless measurement itself is the contribution and known cases can be reproduced robustly.
5. **No toy-only promotion.** The target system must already be competent in the scientifically relevant regime.
6. **No profile-correlation stories without local evidence.** Shared depth/time geometry is not a mechanism.
7. **No rescue sweeps.** Model/layer/threshold/data fishing after a miss is a kill, not a tuning plan.
8. **No tiny effects.** If the phenomenon needs statistical magnification to feel interesting, it is usually too weak for this workspace.
9. **No forced mechanism opening.** First preserve the phenomenon; explanation comes second.
10. **Prefer traces and trajectories.** If a strong reader can inspect representative examples and immediately see the effect, that is a major positive signal.

---

# 6. Current priority lanes

| Lane | Priority | Why |
|---|---:|---|
| RL/post-training behavioral phase changes | S | Strong lab fit, checkpoint-rich, trace-rich, trainable locally, multiple recent reports of non-monotonic dynamics |
| Agentic RL action/observation learning | S | Real structural tension in current training objectives; released environments/frameworks; direct behavior and token-level diagnostics |
| Reasoning trace strategy redistribution | S | Mechanistic but behavior-first; direct traces; strong connection to RL dynamics |
| VLA closed-loop correction / action-chunk inertia | A | Direct control phenomenon and clear method lever; open policies/simulators exist |
| Continual adaptation / forgetting dynamics | A | Lab fit and rich literature, but generic forgetting is crowded; only unusual dynamics survive |
| Generic representation probing | D | Repeated archive failure mode unless anchored to a replicated behavioral event |

---

# 7. Current status vocabulary

- `OBSERVED` — strong published/released evidence exists.
- `REPLICATE_NOW` — direct, accessible, high-value replication target.
- `WATCH` — interesting but source/object is not yet strong enough.
- `COLLIDED` — phenomenon or obvious method is already substantially done.
- `REJECT` — repeats a known failure mode or lacks a meaningful regime.
- `PROMOTE_CANDIDATE` — P1 survived and paper-shape audit is strong; only this status may spawn a numbered topic.

See `ROUND_01_2026-08-23.md` for the first mining pass and `SOURCE_LEDGER.md` for the evidence inventory.
