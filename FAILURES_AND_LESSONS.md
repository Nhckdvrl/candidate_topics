# Failure Log and Lessons for Topic Selection

This document records **why candidate topics failed or were stopped, at what layer they failed, and what lesson should transfer to future topic selection**.

It is deliberately not a flat list of negative results. Different candidates can stop for very different reasons:

- the natural question may be weak or method-driven;
- the proposed experiment may not identify the intended concept;
- the clean comparison may not exist at sufficient scale;
- the selected AI system may fail to instantiate a prerequisite phenomenon;
- the substantive hypothesis may be wrong;
- the motivating phenomenon may replicate while the proposed explanatory axis is too weak or unstable to justify further work;
- an exploratory result may fail locked confirmation.

Those outcomes must not be conflated. In particular, **“not falsified” does not mean “worth continuing.”** This repository is a candidate-selection system, so a preregistered gray-zone result can legitimately be an archive decision even when a broader scientific question remains unresolved.

---

# 1. Failure taxonomy

Before running a large experiment, classify the candidate along the following stack.

## Layer A — Naturalness

Can the scientific question be stated clearly **without mentioning the model, probe, hidden state, checkpoint, SAE, metric, or implementation trick**?

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
- is the measurement contaminated by position, prompt, formatting, tokenizer, or selection artifacts?
- does the supposedly low/high variable really have sufficient dynamic range?
- do all compared systems share enough common support for a paired test?

A failure here means the scientific hypothesis was **not cleanly tested**.

## Layer D — Prerequisite phenomenon / substantive or explanatory strength

Once the construct is identifiable and measurable, two distinct questions remain.

First, does the selected AI system robustly instantiate the prerequisite phenomenon that the higher-order question depends on? Before asking when learned uncontrollability generalizes, for example, the learner must first acquire a strong controllability-dependent behavioral state.

Second, if the motivating phenomenon is present, does the proposed explanatory axis produce a **large, clean, scientifically worthwhile separation**?

This second point matters because:

> **phenomenon replicated != explanatory variable matters.**

A discovery pilot can therefore produce three useful outcomes:

```text
GO          effect large/clean enough to justify locked confirmation
KILL        effect clearly too small or opposite
GRAY ZONE   neither condition is met -> INCONCLUSIVE_DO_NOT_TUNE
```

The gray zone is not an invitation to add samples, swap models, or search a better contrast. For candidate screening, it can itself be a stop condition when the frozen result is not compelling enough to justify a larger research program.

## Layer E — Confirmation / generalization

Does the discovery survive a locked holdout or independent dataset after all measurements are frozen?

A failure here usually indicates winner's curse, over-selection, or a localized/non-robust effect.

---

# 2. The complexity-smell rule

A central lesson from Topic 05 is:

> **When the gate and kill line become more and more complicated, reconsider whether the question itself is still natural and well identified.**

The dangerous pattern is:

```text
we want to show A
-> first prove it is not B
-> then match C
-> then control D
-> then rule out E
-> then add another baseline for F
-> only then can the observed effect be called A
```

This is not automatically wrong; difficult causal questions can require many controls. The warning sign is more specific:

> **the construct itself only becomes interpretable after accumulating many exclusions.**

Two common causes are below.

### 2.1 The target phenomenon is not a stable natural object

For Topic 05, `old route` looked intuitive in prose but was not a stable observable. A continuation could begin old-like, switch strategies, and still finish correctly. The more precisely the route was defined, the less clear the object became.

### 2.2 The observable is too far from the scientific question

Topic 05 wanted to know whether an uncued skill was retained, but the experiment observed performance after supplying part of a correct solution:

```text
P(solve | x) != P(solve | x + correct prefix)
```

The distance between the target concept and the observable created an expanding list of alternative explanations: task simplification, search-space reduction, intermediate-variable provision, wrong-path exclusion, token compatibility, and generic guidability.

Adding one control for every alternative made the protocol increasingly elaborate, while the central identification problem remained.

### Practical heuristic

A strong early-stage topic should ideally admit a **one-clean-contrast** experiment:

```text
A vs B -> one primary measurement whose interpretation is nearly forced by the question
```

If interpretation instead requires something like:

```text
A vs B | C,D,E,F,G
```

before the phenomenon can even be named, downgrade the topic rather than automatically adding controls.

A related lesson from Topics 06 and 07 is that a protocol can be conceptually clean and still not be worth continuing:

- Topic 06: the selected system did not robustly instantiate the prerequisite state.
- Topic 07: the prerequisite phenomenon did appear, but the proposed explanatory architecture axis did not create a strong enough frozen separation.

---

# 3. Topic-by-topic failure record

## Topic 01 — Behavior Stabilization vs. Representation Stabilization

**Final status:** substantive hypothesis failed at G0.

[Archive summary](./01_behavior_vs_representation_stabilization/ARCHIVE_SUMMARY.md)

### Original idea

Behavior/output distributions appear to stabilize during pretraining while weights continue moving. Representation-dynamics work shows features evolve over checkpoints. The adjacent question was whether meaningful internal representations continue reorganizing after behavior has largely stabilized.

### What happened

The behavior-side premise replicated, but representation movement did **not** remain elevated. Cosine drift, standardized residual drift, and CKA all stabilized at least as fast as behavior, with the same direction replicated across deterministic half-sample robustness checks.

### Failure type

**Layer D — substantive hypothesis failure.**

### Main lessons

1. **A clean cross-paper empty cell is a way to generate a question, not evidence that the phenomenon exists.**
2. `parameter drift != meaningful representation drift`.
3. Complex feature methods should explain a phenomenon already visible in a cheap screen; they should not manufacture a phenomenon after the screen points the other way.
4. Predeclared kill criteria worked correctly: the topic stopped before crosscoder/SAE escalation.

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
2. Bootstrap confidence intervals on a selected best cell do not correct for the selection process.
3. Reserve locked confirmation data before inspecting discovery results.
4. Run the cheapest locked holdout immediately after a positive exploratory result.
5. Every negative mechanistic result needs a positive control so failure cannot be blamed on a broken measurement pipeline.
6. Early-stop gates must themselves be power-calibrated.

### Reusable warning sign

If a phenomenon is convincing only at one specially discovered layer/step/threshold with no independent reason those coordinates should matter, assume winner's curse until proven otherwise.

---

## Topic 04 — Confidence and Error Correction

**Final status:** stopped before hypothesis testing because the intended comparison could not be identified at sufficient scale.

[Archive summary](./04_confidence_error_correction/ARCHIVE_SUMMARY.md)

### Original idea

When two learners are equally far from the correct answer, does being strongly committed to one specific wrong answer make corrective learning easier or harder?

The experiment attempted to separate target accessibility from concentration of probability over wrong hypotheses.

### What happened

The first measurement was structurally contaminated:

- top-wrong stability was mechanically correlated with the treatment variable;
- arithmetic averaging across option rotations could turn a sharp but position-sensitive model into an apparently diffuse semantic belief.

A single locked measurement repair used log-space aggregation and removed the treatment-dependent inclusion rule. After retaining the original identification requirements, only 130 clean high/low matched pairs remained, below the preregistered `<200` hard stop.

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

```text
P(solve | x) != P(solve | x + correct prefix)
```

Even a perfect old-self rescue could reflect reduced search, supplied intermediate variables, lexical/continuation compatibility, or generic guidability. In addition, `old route` was not a stable observable object, and teacher-forced NLL remained conditional on the same cue.

The run stopped during partial checkpoint sampling, before scoring or any claim-level gate. There is therefore **no empirical result** about storage loss vs retrieval failure.

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

> "the result would indicate A, provided that it is not B, C, D, E..."

then stop adding controls and reconsider whether A is directly measurable at all.

---

## Topic 06 — When Does Helplessness Become a Worldview?

**Final status:** archived after the prerequisite controllability-acquisition premise failed across v1 and the one permitted v2.

[Archive summary](./06_helplessness_worldview/ARCHIVE_SUMMARY.md)

### Original idea

If an agent repeatedly experiences that its actions do not affect outcomes, does it learn only that one situation is uncontrollable, or does the experience form a broader cross-situation expectation that actions generally do not matter?

The specific hypothesis was that equal uncontrollable experience distributed across semantically different task families would transfer more strongly to a novel controllable task than the same experience concentrated within one family.

### What happened

The 2×2 master–yoked design itself was technically clean. Controllable and yoked-uncontrollable sessions saw exactly matched external success/failure histories; episode counts, test task, latent randomization, and reward exposure were controlled.

In the Qwen3-8B v1 pilot, late-training controllable-vs-uncontrollable intervention behavior differed by only about `2.4pp` in the concentrated condition and `0.3pp` in the distributed condition. The locked novel-test pooled transfer was only `1pp`; diversity amplification was `D=-2pp`, with bootstrap interval `[-8pp,+4pp]`.

Two independently motivated v1 concerns were then handled in one preregistered final v2:

1. move from Qwen3-8B to Qwen3-32B non-thinking inference;
2. increase active-intervention cost from `1` to `2` to remove the obvious binary-action ceiling.

Everything else stayed frozen. The v2 technical gates all passed, but late controllability acquisition still separated only weakly:

```text
late active:
C1  60.94% vs U1  58.85%  (+2.08pp)
C10 60.42% vs U10 55.73%  (+4.69pp)
```

The frozen novel-test quantities were:

```text
H1  = +4.17pp
H10 =  0pp
D   = -4.17pp
bootstrap interval for D = [-12.5pp, 0pp]
```

The project therefore stopped before any larger v2 pilot or confirmation.

### Failure type

**Layer D — prerequisite phenomenon / acquisition failure in the chosen AI system.**

The higher-order psychological claim was not cleanly falsified because the LLM agent never robustly acquired the local controllability distinction needed to make transfer breadth meaningful.

### Main lessons

1. **Validate prerequisite acquisition before studying abstraction/generalization.** If the question is "when does learned X generalize?", the first hard gate should establish that the selected learner robustly acquires `X`.
2. A strong human/cognitive literature does not prove that a vanilla LLM interaction-history agent instantiates the analogous latent state.
3. Before locking a behavioral endpoint, analytically check whether plausible latent-belief changes can move it away from floor/ceiling. The v1 test action was too cheap and therefore nearly always active.
4. One independently motivated preregistered repair can be legitimate. Repeatedly changing model size, reward schedule, memory, prompt, probe, or environment until the prerequisite appears is post-hoc optimization.
5. **A natural question can still be a bad AI topic.** Naturalness is necessary but not sufficient; the chosen system must instantiate the phenomenon cleanly enough to study.
6. Preserve the distinction between "the natural question is unresolved" and "this candidate is not worth continuing." Topic 06 is the latter.

### Reusable warning sign

If a higher-order claim depends on a base phenomenon that is only weakly visible, do not immediately add transfer conditions, abstraction levels, mechanisms, or probes. First establish that the selected learner robustly instantiates the base phenomenon. If it remains weak after one principled locked repair, archive rather than searching for a model/environment combination that makes it true.

---

## Topic 07 — Old Blocks New, or New Erases Old?

**Final status:** archived as `INCONCLUSIVE_DO_NOT_TUNE` at the frozen discovery gate.

[Archive summary](./07_memory_interference_architecture/ARCHIVE_SUMMARY.md)

### Original idea

Classical memory distinguishes proactive interference (old information blocks new learning/retrieval) from retroactive interference (new information damages old memory). A 2026 LLM study reported a robust Transformer tendency toward `PI > RI`, creating a natural question:

> When old and new memories conflict, what determines which side survives?

The candidate hypothesis was that the sequence model's memory-update rule is an important determinant of interference direction. A matched M-A-P family allowed a relatively clean first comparison among Transformer, GLA, DeltaNet, and Gated DeltaNet under the same shared-stream PI/RI measurement.

### What happened

The frozen pilot was technically clean:

```text
4 architectures × 192 rows = 768 rows
skip rate = 0%
duplicate rows = 0
paired cells/model = 192
tokenizer boundary shift = 0
```

The motivating Transformer phenomenon reproduced:

```text
Transformer mean I = +0.1563
I = Accuracy_RI - Accuracy_PI
```

So this was not a prerequisite or measurement failure.

However, the preregistered primary comparison was Transformer vs Gated DeltaNet:

```text
mean I Transformer       = 0.1563
mean I Gated DeltaNet    = 0.0833
Delta_I                  = 0.0729
paired bootstrap 95% CI  = [-0.0313, 0.1771]
sign-transition levels   = 0 / 4
```

The frozen rules required `Delta_I >= 0.10` and a positive CI lower bound to enter locked confirmation. They required `abs(Delta_I) < 0.05` for a clean kill. The observed result lay between them, so the only permitted decision was:

```text
INCONCLUSIVE_DO_NOT_TUNE
```

The broader architecture pattern was also not a clean monotonic memory-editability story:

```text
Transformer       0.1563
DeltaNet          0.1250
Gated DeltaNet    0.0833
GLA               0.0208
```

In particular, DeltaNet remained close to Transformer, and there were no frozen levels showing the hoped-for qualitative Transformer-positive/GatedDeltaNet-negative sign transition.

### Failure / stop type

**Layer D — explanatory-axis strength unresolved at frozen discovery gate.**

This is deliberately not called falsification. The phenomenon was real in the Transformer, but the proposed architecture axis did not produce a large, stable, qualitative enough separation to justify further investment under the preregistered candidate-selection contract.

### Main lessons

1. **Phenomenon existence and explanatory importance are separate gates.** Reproducing PI>RI established that the measurement was active; it did not establish that memory-update architecture is the main determinant.
2. **Do not replace a minimum-worthy-effect criterion with a significance criterion after seeing the result.** The point estimate `0.0729` was already below the frozen `0.10` GO threshold. More samples could narrow the CI without making the effect scientifically large enough.
3. **A gray zone can be a legitimate archive outcome.** Research screening does not require every candidate to be classified as true or false. `INCONCLUSIVE_DO_NOT_TUNE` means the evidence is not compelling enough to spend the next unit of research effort.
4. **Do not shop the architecture pair after seeing results.** Transformer–GLA looked numerically larger than the frozen Transformer–GatedDeltaNet contrast, but switching the primary comparison post hoc would invalidate the falsification-first logic.
5. **Qualitative predictions are valuable.** The hoped-for sign transition occurred at `0/4` levels. Its absence made the “different memory medium -> different interference regime” story substantially less compelling even though mean values differed.
6. **Not falsified != worth continuing.** A broad scientific question may remain open while the current candidate is correctly archived.

### Reusable warning sign

If the motivating phenomenon replicates but the proposed explanatory variable only changes its magnitude modestly, gives heterogeneous ordering across variants, and fails the frozen minimum-worthy-effect gate, do not automatically answer with `n↑`, more models, or mechanistic probes.

First ask:

> **Is the explanatory axis itself strong enough to deserve a paper?**

If the answer is not clearly yes under the frozen discovery contract, archive and move on.

---

# 4. Cross-topic lessons

The six archived projects now cover distinct ways a research candidate can stop:

| Topic | Failure / stop layer | What failed or remained unresolved |
|---|---|---|
| 01 | Substantive hypothesis | the expected behavior/representation temporal decoupling did not occur |
| 02 | Confirmation | the exploratory hidden-state signal did not survive a locked independent test |
| 04 | Measurement/common support | the intended high/low commitment comparison could not be constructed cleanly at sufficient scale |
| 05 | Conceptual identification | the proposed observable could not distinguish retained competence from task simplification/conditional continuation |
| 06 | Prerequisite phenomenon / acquisition | the chosen LLM agent did not robustly acquire the controllability-dependent state required for the higher-order question |
| 07 | Frozen discovery / explanatory-axis strength | the seed PI>RI phenomenon replicated, but the preregistered memory-architecture contrast was not large, robust, or qualitative enough to justify confirmation |

The ordering matters. Future projects should try to fail **as early as possible**:

```text
Natural question
    ↓
Does the selected AI system clearly instantiate the prerequisite phenomenon?
    ↓
Conceptual identifiability
    ↓
Measurement validity / common support
    ↓
Cheap substantive phenomenon test
    ↓
Does the proposed explanatory axis create a minimum-worthwhile clean effect?
    ↓
Locked confirmation
    ↓
Only then scale up mechanisms / models / training
```

For some topics, prerequisite-instantiation and substantive G0 are the same experiment. For higher-order questions about transfer, abstraction, forgetting, interference, or meta-learning, they may need to be separated explicitly.

A second separation is equally important:

```text
phenomenon exists
!=
our favorite explanation is important
```

Topic 07 is the clean example. The seed phenomenon replicated; the proposed explanatory axis did not earn continuation.

Do not spend GPU to answer a question that has already failed one of the earlier layers or landed in a frozen no-tune gray zone.

---

# 5. Mandatory preflight for future candidates

Before a new topic enters active validation, write answers to the following.

## 5.1 Natural question

State the question in one sentence **without AI-specific terminology**.

If the sentence is not interesting by itself, reconsider the topic.

## 5.2 Why is the phenomenon already real?

Identify the empirical observation or established tension that motivates the question.

Do not infer a new phenomenon merely because two papers leave an empty combinatorial cell.

Then separate two claims:

1. the phenomenon exists in the source domain;
2. the **selected AI system** robustly instantiates the prerequisite phenomenon needed for this candidate.

Do not treat evidence for (1) as evidence for (2).

## 5.3 Prerequisite-instantiation gate

For any higher-order question of the form:

```text
when does X transfer / generalize / abstract / disappear / interfere?
```

state the cheapest experiment showing that the selected learner first exhibits a strong, directly measurable `X`.

If this prerequisite is weak, stop before adding the higher-order manipulation.

## 5.4 One-clean-contrast

What is the simplest observation that separates the main explanations?

Prefer:

```text
A vs B -> one primary contrast
```

over a chain requiring many conditional exclusions.

## 5.5 Identifiability counterfactual

Assume the experiment produces the strongest hoped-for result.

Write at least the two strongest alternative explanations. Then ask:

> **Would the primary observation still be compatible with them?**

If yes, and distinguishing them requires an expanding family of controls that all modify the original condition, the topic is not ready.

## 5.6 Complexity smell

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

## 5.7 Measurement validity and decision calibration

Check whether nuisance variation is mechanically entangled with the treatment/target variable.

For behavioral decisions, also inspect whether the reward/cost structure forces the observable toward floor or ceiling over the plausible range of latent beliefs. A theoretically clean endpoint is useless if realistic belief changes cannot move the action.

Do this before training.

## 5.8 Common support

If the question requires matched comparisons, confirm that the comparison actually exists in the chosen model/data system at useful scale.

Do not loosen the defining confound control to rescue sample size.

## 5.9 Discovery budget

List every dimension that will be searched:

```text
model × layer × step × threshold × prompt × metric × dataset
```

If this grid is large, pre-split discovery and confirmation before looking at results.

## 5.10 Minimum-worthwhile effect, GO/KILL line, and gray zone

Do not define only statistical significance. State in advance:

- what effect would be large/clean enough to justify confirmation (`GO`);
- what effect would make the topic not worth continuing (`KILL`);
- what intermediate region should be labeled `INCONCLUSIVE_DO_NOT_TUNE`.

The minimum worthwhile effect should reflect the strength needed for the **scientific story**, not merely the sample size needed for `p<0.05`.

If the discovery lands in the gray zone, do not reinterpret it as “almost GO” and increase `n` solely to chase significance. For candidate screening, archive unless an independently motivated new observation changes the question.

## 5.11 Explanatory-axis gate

After the motivating phenomenon is established, explicitly ask:

> **Does the variable we actually want to explain the phenomenon with create a large, clean separation?**

Do not proceed to mechanisms merely because the base phenomenon replicated.

This is especially important for architecture, representation, memory, optimizer, or training-dynamics explanations where many technical axes can be correlated with the same behavior.

## 5.12 No-rescue rule

After a locked gate fails or lands in a frozen no-tune region, do not reopen layer/model/metric/threshold search to preserve the same claim.

One narrowly defined repair is defensible only when the defect is independently identifiable before the new outcome is inspected and the repair itself is frozen. If that repaired gate fails again, stop.

A genuinely new observation may motivate a **new separately registered topic**, but it does not retroactively rescue the old one.

---

# 6. Current working principles

The repository should increasingly prefer questions with these properties:

1. **Natural before technical.** The question survives deletion of AI-specific vocabulary.
2. **System instantiation before higher-order claims.** A source-domain phenomenon is not enough; the chosen AI learner must robustly exhibit the prerequisite phenomenon before studying its transfer, abstraction, or mechanism.
3. **Phenomenon before mechanism.** There is a real observation to explain, not merely an unfilled measurement cell.
4. **Explanatory axis after phenomenon.** Once the phenomenon is real, the proposed explanation must itself produce a large, clean separation before mechanisms are studied.
5. **Short inferential distance.** The primary observable is close to the scientific concept.
6. **One clean contrast.** A main result has a direct interpretation without a long chain of exclusions.
7. **Behavioral readouts must be calibrated.** Plausible latent-state changes should be capable of moving the measured decision away from floor/ceiling.
8. **Minimum-worthy effect before significance.** Decide what effect is worth a paper before looking at the result; do not substitute sample-size-driven significance later.
9. **Complex methods explain; they do not create.** SAE/probes/hidden-state analyses come after a clear phenomenon and a compelling explanatory contrast.
10. **Cheap falsification first.** Use small models/data when they can genuinely kill the claim.
11. **Locked confirmation immediately after a real GO.** Do not invest in a large story around an exploratory or gray-zone cell.
12. **Failure labels stay precise.** `hypothesis false`, `measurement failed`, `prerequisite absent`, `inconclusive explanatory axis`, and `confirmation failed` are different outcomes.
13. **Protocol complexity is evidence.** If clarification makes the gate continually longer rather than the experiment cleaner, reconsider the question.
14. **Stop means stop.** Preserve code, outputs, and lessons, then move on.

This file should be updated whenever a candidate is archived.
