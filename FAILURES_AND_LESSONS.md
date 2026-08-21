# Failure Log and Lessons for Topic Selection

This document records **why candidate topics failed, at what layer they failed, and what lesson should transfer to future topic selection**.

It is not a list of negative results. Different topics failed for very different reasons:

- a substantive hypothesis can be wrong;
- an exploratory result can fail confirmation;
- a clean comparison may not exist at sufficient scale;
- the proposed experiment may not identify the intended concept at all.

Those failure types must not be conflated.

The goal of this file is to prevent the repository from repeatedly rediscovering the same weak research pattern under new terminology.

---

# 1. Failure taxonomy

Before running a large experiment, classify the candidate along the following stack.

## Layer A — Naturalness

Can the scientific question be stated clearly **without mentioning the model, probe, hidden state, checkpoint, SAE, or metric**?

A good candidate should first look like a real question about learning, memory, reasoning, information, behavior, or computation. AI is the experimental system, not the reason the question exists.

If the question only becomes interesting after introducing a particular representation metric or model component, the topic is likely method-driven.

## Layer B — Conceptual identifiability

Suppose the planned main result appears exactly as hoped. **Would that observation actually distinguish the claimed explanations?**

If the answer is no, more data and more controls will not fix the core problem.

This layer should be checked before the ordinary pilot.

## Layer C — Measurement / common support

Can the variables required by the question actually be measured and compared cleanly at sufficient scale?

Examples:

- can two treatment groups be matched on the confound that defines the scientific comparison?
- is the measurement itself contaminated by position, prompt, or formatting artifacts?
- does the supposedly low/high variable really have sufficient dynamic range?

A failure here means the hypothesis was **not tested**.

## Layer D — Substantive hypothesis

Once the construct is identifiable and measurable, does the predicted phenomenon actually occur?

A failure here is a genuine scientific negative.

## Layer E — Confirmation / generalization

Does the discovery survive a locked holdout or independent dataset after all measurements are frozen?

A failure here usually indicates winner's curse, over-selection, or a localized/non-robust effect.

---

# 2. The complexity-smell rule

A central lesson from Topic 05 is:

> **When the gate and kill line become more and more complicated, pause and reconsider whether the question itself is still natural and well identified.**

The dangerous pattern is:

```text
we want to show A
→ first prove it is not B
→ then match C
→ then control D
→ then rule out E
→ then add another baseline for F
→ only then can the observed effect be called A
```

This is not automatically wrong — difficult causal questions can require many controls. The warning sign is more specific:

> **the construct itself only becomes interpretable after accumulating many exclusions.**

In that case, one of two things is often happening.

### 2.1 The target phenomenon is not a stable natural object

For Topic 05, `old route` looked intuitive in prose but was not a stable observable. A continuation could begin old-like, switch strategies, and still finish correctly. The more precisely we tried to define re-entry, the less clear the object became.

### 2.2 The observable is too far from the scientific question

Topic 05 wanted to know whether an uncued skill was retained, but the experiment observed performance after supplying part of a correct solution:

\[
P(\mathrm{solve}\mid x)
\neq
P(\mathrm{solve}\mid x+\mathrm{correct\ prefix}).
\]

The distance between the target concept and the observable created an expanding list of alternative explanations: task simplification, search-space reduction, intermediate-variable provision, wrong-path exclusion, token compatibility, and generic guidability.

Adding one control for every alternative made the protocol increasingly elaborate, but the central identification problem remained.

### Practical heuristic

A strong early-stage topic should ideally admit a **one-clean-contrast** experiment:

```text
A vs B
with one primary measurement that is almost forced by the question.
```

If interpretation instead requires something like:

```text
A vs B | C,D,E,F,G
```

before the phenomenon can even be named, downgrade the topic and reconsider the question itself rather than automatically adding more controls.

Another useful heuristic:

> **Good questions often make the first experiment simpler as they are clarified. Weakly identified questions often make the gate longer as they are clarified.**

---

# 3. Topic-by-topic failure record

## Topic 01 — Behavior Stabilization vs. Representation Stabilization

**Final status:** substantive hypothesis failed at G0.

[Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md)

### Original idea

Behavior/output distributions appear to stabilize during pretraining while weights continue moving. Representation-dynamics work shows features evolve over checkpoints. The proposed adjacent question was whether meaningful internal representations continue reorganizing after behavior has largely stabilized.

### What happened

The behavior-side premise replicated, but representation movement did **not** remain elevated. Cosine drift, standardized residual drift, and CKA all stabilized at least as fast as behavior, with the same direction replicated across deterministic half-sample robustness checks.

### Failure type

**Layer D — substantive hypothesis failure.**

### Main lessons

1. **A clean cross-paper empty cell is a way to generate a question, not evidence that the phenomenon exists.**
2. `parameter drift != meaningful representation drift`.
3. Complex feature methods should explain a phenomenon already visible in a cheap screen; they should not be used to manufacture a phenomenon after the screen points the other way.
4. Predeclared kill criteria worked correctly here: the topic stopped before crosscoder/SAE escalation.

### Reusable warning sign

If the only way to preserve the story after a simple negative is to move to a much more flexible representation method, the topic is drifting from phenomenon-driven to method-driven research.

---

## Topic 02 — DLM Trajectory Fate

**Final status:** exploratory claim failed locked independent confirmation.

[Archive summary](./02_dlm_trajectory_fate/ARCHIVE_SUMMARY.md)

### Original idea

DLM trajectories can transiently recover or overwrite answers, while hidden states encode final correctness. The proposed adjacent question was whether a current hidden state predicts the **future transient fate** of the current surface state after controlling current and final correctness.

### What happened

Exploration on GSM8K found attractive cells around specific combinations of denoising step, hidden layer, lead threshold, and task. After those cells were frozen, the effects weakened on an untouched GSM8K tail and collapsed on independent GSM1K:

- recovery AUC: roughly `0.676 -> 0.498`;
- overwrite AUC: roughly `0.705 -> 0.434`.

The final-correctness positive control remained strong, so the pipeline itself was not simply broken.

### Failure type

**Layer E — confirmation failure / winner's curse.**

### Main lessons

1. Penalize topics whose first convincing result requires a large search over `step × layer × threshold × task`.
2. Bootstrap confidence intervals on a **selected best cell** do not correct for the selection process.
3. Reserve locked confirmation data before inspecting discovery results.
4. Run the cheapest locked holdout immediately after a positive exploratory result.
5. Every negative mechanistic result needs a positive control so that failure cannot be blamed on the measurement pipeline.
6. Early-stop gates must themselves be power-calibrated; a cheap gate that frequently false-stops a real effect is not a valid kill line.

### Reusable warning sign

If a phenomenon is convincing only at one specially discovered layer/step/threshold with no independent reason that those coordinates should matter, assume winner's curse until proven otherwise.

---

## Topic 04 — Confidence and Error Correction

**Final status:** stopped before hypothesis testing because the intended comparison could not be identified at sufficient scale.

[Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md)

### Original idea

When two learners are equally far from the correct answer, does being strongly committed to one specific wrong answer make corrective learning easier or harder?

The experiment attempted to separate:

\[
\text{target accessibility}=p(y^*\mid x)
\]

from concentration of probability over wrong hypotheses.

### What happened

The first measurement was structurally contaminated:

- top-wrong stability was mechanically correlated with the treatment variable;
- arithmetic averaging across option rotations could turn a sharp but position-sensitive model into an apparently diffuse semantic belief.

A single locked measurement repair fixed much of this by using log-space aggregation and removing the treatment-dependent inclusion rule. However, after retaining the original identification requirements, only 130 clean high/low matched pairs remained, below the preregistered `<200` hard stop.

Corrective SFT was never run.

### Failure type

**Layer C — measurement/common-support identification failure.**

### Main lessons

1. Do not let an inclusion/reliability gate depend mechanically on the treatment variable being studied.
2. Construct validity comes before training.
3. A measurement repair can be legitimate when the defect is mathematically explicit and discovered before outcome data, but allow at most a tightly defined repair rather than an open-ended sequence of rescues.
4. Large marginal pools do not imply the scientific comparison exists; what matters is **common support under the required controls**.
5. Do not loosen the exact confound control that gives the question meaning just to create a larger sample.
6. If the comparison only exists after extrapolation or heavy regression adjustment with little overlap, the natural experimental contrast may not actually be present in the chosen system.

### Reusable warning sign

If constructing the treatment groups requires increasingly elaborate debiasing, matching, reliability filtering, and support repair before any substantive experiment can begin, ask whether the chosen system genuinely instantiates the natural distinction.

---

## Topic 05 — Temporal Forgetting: Lost Skill or Lost Entry Point?

**Final status:** stopped at conceptual identification gate; no empirical hypothesis conclusion.

[Archive summary](./05_temporal_forgetting_reentry/ARCHIVE_SUMMARY.md)

### Original idea

If a learner solved a problem reliably at an earlier checkpoint and later fails, was the skill erased or is the former solution merely inaccessible?

The proposed validation supplied prefixes from the model's own earlier correct trajectory and compared old-self, other-correct, final-wrong, never-correct, and teacher-forced NLL conditions.

### What happened

During implementation, the experiment became increasingly elaborate because every apparent rescue result admitted another explanation. The deeper issue was not missing controls; it was that the intervention changed the task:

\[
P(\mathrm{solve}\mid x)
\neq
P(\mathrm{solve}\mid x+\mathrm{correct\ prefix}).
\]

Even a perfect old-self rescue could reflect reduced search, supplied intermediate variables, lexical/continuation compatibility, or generic guidability. In addition, `old route` was not a stable observable object, and teacher-forced NLL remained conditional on the same cue.

The run was stopped during partial checkpoint sampling, before scoring or any claim-level gate. There is therefore **no empirical result** about storage loss vs retrieval failure.

### Failure type

**Layer B — conceptual identification failure.**

### Main lessons

1. **Many controls do not rescue a non-identifying intervention.**
2. Before asking whether a test is statistically powerful, ask whether a positive result would actually imply the claimed mechanism.
3. If every refinement adds another alternative explanation and another control, treat protocol complexity as evidence about the weakness of the question/observable mapping, not merely as an engineering burden.
4. A natural verbal distinction (`forgotten` vs `inaccessible`) is not automatically an experimentally identifiable distinction.
5. Conditional likelihood after supplying a cue does not establish uncued retention.
6. A latent object such as a `route`, `strategy`, or `skill` must have a stable operational definition before it can anchor a mechanistic claim.

### Reusable warning sign

If the main interpretation is repeatedly phrased as:

> "the result would indicate A, **provided that it is not B, C, D, E...**"

then stop adding controls and reconsider whether A is directly measurable at all.

---

# 4. Cross-topic lessons

The four archived projects reveal four different ways a research candidate can fail:

| Topic | Failure layer | What failed |
|---|---|---|
| 01 | Substantive hypothesis | the expected behavior/representation temporal decoupling did not occur |
| 02 | Confirmation | the exploratory hidden-state signal did not survive a locked independent test |
| 04 | Measurement/common support | the intended high/low commitment comparison could not be constructed cleanly at sufficient scale |
| 05 | Conceptual identification | the proposed observable could not distinguish retained competence from task simplification/conditional continuation |

The ordering matters. Future projects should try to fail **as early as possible**:

```text
Natural question
    ↓
Conceptual identifiability
    ↓
Measurement validity / common support
    ↓
Cheap substantive G0
    ↓
Locked confirmation
    ↓
Only then scale up mechanisms / models / training
```

Do not spend GPU to answer a question that has already failed one of the earlier layers.

---

# 5. Mandatory preflight for future candidates

Before a new topic enters active validation, write answers to the following.

## 5.1 Natural question

State the question in one sentence **without AI-specific terminology**.

If the sentence is not interesting by itself, reconsider the topic.

## 5.2 Why is the phenomenon already real?

Identify the empirical observation or established tension that motivates the question.

Do not infer a new phenomenon merely because two papers leave an empty combinatorial cell.

## 5.3 One-clean-contrast

What is the simplest observation that separates the main explanations?

Prefer:

```text
A vs B → one primary contrast
```

over a chain requiring many conditional exclusions.

## 5.4 Identifiability counterfactual

Assume the experiment produces the strongest hoped-for result.

Write at least the two strongest alternative explanations. Then ask:

> **Would the primary observation still be compatible with them?**

If yes, and distinguishing them requires an expanding family of controls that all modify the original condition, the topic is not ready.

## 5.5 Complexity smell

Count how much scaffolding is required before the result is interpretable:

- matching dimensions;
- exclusion rules;
- auxiliary baselines;
- nested gates;
- alternative probes;
- special-case thresholds;
- post-hoc subgroups.

There is no fixed numeric cutoff, but complexity should trigger a conceptual review rather than automatic protocol growth.

Ask:

> **Are these controls making a clear causal question rigorous, or are they trying to make an unclear construct exist?**

That distinction is crucial.

## 5.6 Measurement validity

Check whether nuisance variation is mechanically entangled with the treatment/target variable.

Do this before training.

## 5.7 Common support

If the question requires matched comparisons, confirm that the comparison actually exists in the chosen model/data system at useful scale.

Do not loosen the defining confound control to rescue sample size.

## 5.8 Discovery budget

List every dimension that will be searched:

```text
model × layer × step × threshold × prompt × metric × dataset
```

If this grid is large, pre-split discovery and confirmation before looking at results.

## 5.9 Kill line

Write what observation would make the topic **not worth continuing**.

The kill line should ideally be short and tied directly to the natural question. If the kill contract itself needs a page of branching logic before the first experiment, invoke the complexity-smell rule and reconsider the topic.

## 5.10 No-rescue rule

After a locked gate fails, do not reopen layer/model/metric/threshold search to preserve the same claim.

A genuinely new observation may motivate a **new separately registered topic**, but it does not retroactively rescue the old one.

---

# 6. Current working principles

The repository should increasingly prefer questions with these properties:

1. **Natural before technical.** The question survives deletion of AI-specific vocabulary.
2. **Phenomenon before mechanism.** There is a real observation to explain, not merely an unfilled measurement cell.
3. **Short inferential distance.** The primary observable is close to the scientific concept.
4. **One clean contrast.** A main result has a direct interpretation without a long chain of exclusions.
5. **Complex methods explain; they do not create.** SAE/probes/hidden-state analyses come after a clear phenomenon.
6. **Cheap falsification first.** Use small models/data when they can genuinely kill the claim.
7. **Locked confirmation immediately after discovery.** Do not invest in a large story around an exploratory cell.
8. **Failure labels stay precise.** `hypothesis false`, `measurement failed`, and `question not identifiable` are different outcomes.
9. **Protocol complexity is evidence.** If clarification makes the gate continually longer rather than the experiment cleaner, reconsider the question.
10. **Stop means stop.** Preserve code and lessons, then move on.

This file should be updated whenever a candidate is archived.