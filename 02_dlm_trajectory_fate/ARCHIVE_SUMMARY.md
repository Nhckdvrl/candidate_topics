# Topic 02 Archive Summary — DLM Trajectory Fate

**Status: ARCHIVED / FALSIFIED AS A BROAD CLAIM**

This project asked a narrow question about diffusion language model (DLM) denoising trajectories:

> Before a visible correctness transition happens, does the current hidden representation already contain information about whether the current answer will transiently recover or be overwritten later?

The project produced a strong exploratory G0 signal on LLaDA-8B / GSM8K, but both preregistered effects failed on independent GSM1K confirmation despite adequate event support and a strong positive control. The final conclusion is therefore to **stop the topic rather than search for new steps, layers, lead thresholds, parsers, or probe families**.

The most important scientific takeaway is not that transient DLM trajectory events are absent. They are present on independent data. What failed is the proposed new claim that a single current hidden state robustly predicts the future transient fate of the current answer after controlling current correctness and final outcome.

---

## 1. Where the question came from

Two adjacent results motivated the project.

1. **Time Is a Feature / dLLM-MidTruth** showed that intermediate complete `x0` predictions during DLM denoising can be non-monotonic: an answer can become correct and later be overwritten, or move through other transient states.
2. **Probing Functional Correctness in Diffusion Language Models** showed that DLM hidden states increasingly encode information about **final functional correctness**.

The adjacent question was therefore:

> Replace the already-studied target “final correctness” with “future fate of the current surface state.”

A naive formulation would be confounded by final correctness. For example, a probe could predict whether a currently wrong state will ever become correct simply by reading the already-known final-correctness signal. The project therefore made the novelty test **final-outcome controlled**.

### Primary labels

**Transient recovery**

Among trajectories that are **wrong now and wrong at the end**:

- positive: later becomes observably correct at least once, then ends wrong;
- negative: never becomes observably correct later.

**Transient overwrite**

Among trajectories that are **correct now and correct at the end**:

- positive: later becomes observably wrong at least once, then ends correct;
- negative: remains observably correct.

Because current correctness and final correctness are fixed within each comparison, success cannot be explained by the known final-correctness probe alone.

---

## 2. Implementation audit and scientific safeguards

Before running the main experiment, the implementation was audited against the public `dLLM-MidTruth` and `dlm-probing` codebases. Several details were important enough to change the original validation design.

### Surface state

The surface trajectory is the **complete current `x0` prediction before token transfer**, matching the temporal measurement used by `dLLM-MidTruth`. Decoding only the partially committed token state would measure a different process.

### No-answer-yet is not wrong

Strict parsing keeps an `observed` mask. A denoising state without a valid requested answer marker is unavailable, not silently labeled wrong.

### Deterministic primary geometry

The primary experiment uses `temperature=0`, so future fate is not partly determined by future Gumbel randomness invisible to the current hidden state.

### Same-step comparison

Positive and negative examples are compared at the same absolute denoising step. This prevents the probe from succeeding by reading diffusion time.

### Final-outcome control

The primary tasks are `transient_recovery` and `transient_overwrite`, not easier generic recover/overwrite labels.

### Baselines and controls

Every hidden-state result is compared against:

1. a surface-state baseline containing uncertainty/progress features;
2. the same hidden layer at step 0, controlling static problem difficulty;
3. a positive-control final-correctness probe, verifying that the hidden-state geometry can reproduce an established signal.

### Probe family

The hidden probe follows the reference work closely:

```text
mean-pooled hidden state
-> StandardScaler
-> PCA(max 64)
-> LogisticRegression(C=1, lbfgs)
-> out-of-fold AUC
```

Uncertainty is estimated using paired bootstrap resampling over out-of-fold predictions.

---

## 3. G0 — exploratory discovery on GSM8K

### Geometry

```text
model             GSAI-ML/LLaDA-8B-Instruct
dataset           openai/gsm8k test, ids 0..999
steps             64
generation          128 tokens
block length        32
temperature         0
prompt/parser       MidTruth-style strict boxed numeric answer
hidden indices      24, 25, 28
capture steps       0,1,2,4,8,16,24,32,40,48,56,60,62,63
lead thresholds     4, 8, 16
```

### Positive control

The known final-correctness signal replicated strongly:

```text
best later final-correctness AUC = 0.8723
step-0 AUC                       = 0.7938
delta                            = +0.0785
```

This established that the fast deterministic MidTruth geometry could recover a known emergent hidden-state signal.

### Best exploratory recovery cell

```text
task                  transient_recovery
step                  16
hidden tuple index    25
minimum lead          4
n                     150
positive / negative   41 / 109
hidden AUC            0.6762
95% bootstrap CI      [0.5844, 0.7581]
surface AUC           0.4594
delta vs surface      +0.2168
step-0 hidden AUC     0.5249
delta vs step 0       +0.1513
```

The paired bootstrap intervals for the hidden-vs-surface and hidden-vs-step0 differences were also positive.

### Best exploratory overwrite cell

```text
task                  transient_overwrite
step                  4
hidden tuple index    28
minimum lead          16
n                     166
positive / negative   46 / 120
hidden AUC            0.7046
95% bootstrap CI      [0.6131, 0.7900]
surface AUC           0.4190
delta vs surface      +0.2856
step-0 hidden AUC     0.6239
delta vs step 0       +0.0807
```

The automated G0 decision was `CONTINUE`.

### Warning already visible in G0

The signal was highly localized.

- Recovery was strongest specifically around step 16 / layer 25 / lead >= 4.
- Overwrite was strongest specifically around step 4 / layer 28 / lead >= 16.
- Nearby layers and lead thresholds frequently weakened or lost the positive differential confidence interval.

The discovery procedure had searched over **denoising step × hidden layer × lead threshold × task**. The bootstrap intervals described uncertainty for a chosen cell but did not adjust for the fact that the cell itself was selected after inspecting a large grid.

At the end of G0, the dominant unresolved threat was therefore already clear: **winner's curse / selection-driven discovery**.

---

## 4. G1-A — untouched GSM8K holdout

To test the selected cells without changing the data distribution, G0 was frozen and the remaining untouched GSM8K test examples were used:

```text
ids     1000..1318
n       319
model   same LLaDA-8B
geometry unchanged
```

No new step, layer, lead threshold, parser, or probe was searched.

Result:

```text
AUDIT_ONE_DIRECTIONAL
```

- the locked recovery cell preserved the preregistered direction;
- the locked overwrite cell did not.

Because the holdout was small and the transient classes were sparse, this was defined in advance as a directional audit rather than the decisive confirmation. However, the overwrite failure was already an early warning that the G0 result was unstable.

---

## 5. G1-B first attempt — the underpowered 200-example support gate

The first Stage-2 runner used the first 200 GSM1K examples as a cheap surface-only support preflight.

Observed locked-event counts were:

| task | positive | negative |
|---|---:|---:|
| transient recovery, step 16 / lead >= 4 | 4 | 26 |
| transient overwrite, step 4 / lead >= 16 | 5 | 13 |

The old heuristic required at least 6 positives and 20 negatives, so it returned:

```text
STOP_LOW_LOCKED_SUPPORT
```

### Why that stop was not accepted as a scientific negative

The 200-example heuristic had not been power-calibrated. Using the G0 locked-cell frequencies as the expectation under a perfectly stable phenomenon, the old gate would pass with only approximately:

```text
recovery   56.9%
overwrite  75.1%
```

Therefore a real effect with unchanged event rates still had a substantial probability of being stopped early.

Crucially, this protocol defect was identified **before any GSM1K hidden-state confirmation had been run or inspected**. The scientific hypotheses remained frozen; only the support-decision sample size was corrected.

The revised protocol used all 1,205 GSM1K examples for support, with the same `min_class_count=25` already required by the confirmatory probe.

This episode was a workflow-design failure, not evidence for or against the hidden-state hypothesis.

---

## 6. G1-B final — full 1,205-example GSM1K confirmation

This was the decisive experiment.

### Full-data support

Both locked phenomena occurred often enough:

| task | positive | negative |
|---|---:|---:|
| transient recovery, step 16 / layer 25 / lead >= 4 | 33 | 163 |
| transient overwrite, step 4 / layer 28 / lead >= 16 | 34 | 100 |

Therefore the project was not failing because transient trajectory events disappeared on new questions.

### Positive control

The final-correctness probe remained strong:

```text
AUC = 0.896 at both locked layers
step-0 deltas > 0.04
```

Thus the model geometry, hidden extraction, probing pipeline, and independent dataset still supported the established correctness representation signal.

### Locked recovery confirmation

```text
hidden AUC          0.498
delta vs surface   -0.135
delta vs step 0    -0.017
```

### Locked overwrite confirmation

```text
hidden AUC          0.434
delta vs surface   -0.119
delta vs step 0    -0.011
```

Both 97.5% bootstrap lower bounds for the preregistered confirmation margin were below zero.

Final status:

```text
FAIL_BOTH
```

The retry runner correctly stopped before Dream. No rescue search was performed.

---

## 7. Final scientific conclusion

The broad claim is falsified.

A concise statement is:

> DLM transient recovery and overwrite events generalize to independent GSM1K examples, but the exploratory hidden-state predictability does not. Under frozen step/layer/lead cells, both trajectory-fate probes collapse to chance or worse despite adequate event support and a successfully replicated final-correctness positive control. The original G0 signal was therefore most likely selection-driven.

This distinction matters:

- **surface trajectory non-monotonicity exists**;
- **final correctness remains decodable from hidden states**;
- but **the specific proposed new claim — robust pre-transition hidden prediction of final-outcome-controlled transient fate — did not survive confirmation**.

The project should not retreat to “trajectory oscillations are interesting,” because that returns to the seed phenomenon rather than preserving the proposed novelty.

---

## 8. Why the project failed

### 8.1 The primary failure: exploratory multiplicity / winner's curse

G0 searched over several axes simultaneously:

```text
denoising step
× hidden layer
× future lead threshold
× trajectory task
```

A sufficiently large search space makes it easy to find a cell with an attractive AUC even when the underlying effect is weak or unstable.

The G0 confidence intervals were useful for estimating uncertainty **conditional on a selected cell**, but they were not selection-adjusted inference over the full search process.

The strongest evidence for this explanation is the confirmation pattern:

```text
G0 discovery:        recovery 0.676 / overwrite 0.705
GSM8K holdout:       only one direction preserved
GSM1K confirmation: recovery 0.498 / overwrite 0.434
```

This is the characteristic shape of a discovery effect collapsing after the selected measurement is frozen.

### 8.2 The effect was visibly localized even before confirmation

Nearby layers or lead thresholds were often much weaker. A natural robust phenomenon should ideally not require a very specific combination of measurement coordinates to become visible unless there is an independent mechanistic reason for those coordinates.

Here, the layer/step/lead specificity emerged from the search rather than from a strong prior prediction.

### 8.3 Event existence was mistaken for evidence that the new representation claim might generalize

The surface phenomenon itself is real enough to reproduce: GSM1K still contains transient recovery and overwrite examples.

But “the event occurs” and “a current hidden state predicts the event's future fate” are separate hypotheses. The latter was the actual novelty claim, and it failed.

### 8.4 The first Stage-2 compute-saving gate was statistically underpowered

The original 200-example preflight was a bad stopping rule. It could have falsely stopped a stable effect with high probability.

This did not cause the final scientific failure — the protocol was corrected before hidden confirmation — but it exposed an important experimental-design problem: **cheap gates must themselves be power-calibrated**.

### 8.5 The final negative cannot reasonably be blamed on a broken geometry

Several facts rule out the easiest technical explanations:

- full GSM1K class support was adequate;
- the same frozen parser and deterministic geometry were used;
- the known final-correctness positive control reached AUC 0.896;
- the hidden probe failed against both surface and step-0 baselines;
- both locked tasks failed, not just one sparse edge case.

Therefore the correct response is to stop, not to tune the experiment until a new positive appears.

---

## 9. Lessons for future topic selection

### Lesson 1 — penalize topics whose first convincing result requires a large measurement search

If a candidate only looks strong after scanning many combinations of step, layer, threshold, prompt, or metric, its apparent novelty should be heavily discounted.

A stronger topic has a primary measurement that is nearly forced by the scientific question.

### Lesson 2 — reserve confirmation data before looking at discovery results

For future G0s, allocate a holdout at the start rather than after a promising cell appears.

A useful pattern is:

```text
discovery split   -> choose/freeze measurement
confirmation split -> one-shot test
external dataset   -> generalization only after confirmation
```

The holdout should not be touched while selecting layers, steps, thresholds, or statistics.

### Lesson 3 — “significant after search” is not the same as “confirmed”

Ordinary bootstrap confidence intervals do not automatically account for selecting the best result from a grid.

If broad discovery is unavoidable, use one of:

- a strict discovery/confirmation split;
- nested selection and evaluation;
- selection-aware/multiplicity-aware inference;
- a preregistered single primary cell derived from an external mechanistic reason.

### Lesson 4 — run the cheapest locked holdout immediately after a positive G0

The untouched GSM8K tail already weakened the overwrite hypothesis. In future, such a holdout should happen immediately before investing in broader implementation or cross-model replication.

### Lesson 5 — separate “the phenomenon exists” from “our proposed explanation/predictor works”

A robust surface phenomenon can coexist with a failed mechanistic or representational hypothesis.

The question to preserve is the novel one, not the seed phenomenon after the novel claim fails.

### Lesson 6 — every negative must have a positive control

The final GSM1K result is interpretable precisely because the final-correctness probe still worked strongly. Without that control, a negative AUC could always be blamed on the model, hidden extraction, pooling, prompt, or probe geometry.

### Lesson 7 — power-calibrate early-stop gates

A preflight should only have authority to kill an experiment if its false-stop probability is acceptable under the minimum effect/event rate worth pursuing.

Do not use an arbitrary small prefix such as “first 200 examples” as a scientific stop rule unless its power has been quantified. If subsampling is necessary, randomize or stratify it rather than relying on dataset storage order.

### Lesson 8 — do not rescue a failed preregistered hypothesis by reopening the search space

After `FAIL_BOTH`, the following would turn confirmation back into exploration:

- changing the layer;
- moving the denoising step;
- reducing the lead threshold;
- switching parser or prompt;
- adding nonlinear probes;
- moving to another dataset to hunt for a positive;
- running Dream and reporting only a favorable cell.

Those could generate new exploratory observations, but they would not rescue this topic.

### Lesson 9 — strong candidate questions should survive a “one-shot measurement” test

Before committing to a topic, ask:

> If I had to specify one primary measurement now and could never change it after seeing the result, would the question still be compelling and testable?

If the answer is no, the topic is probably too dependent on researcher degrees of freedom.

---

## 10. Reusable infrastructure that remains valuable

Although the scientific claim failed, several pieces of the validation infrastructure are worth reusing:

- complete-`x0` trajectory extraction before token transfer;
- tri-state answer handling (`observed` vs correct/wrong);
- deterministic DLM trajectory generation for fate questions;
- final-outcome-controlled trajectory labels;
- same-step comparisons;
- surface uncertainty/progress baselines;
- step-0 hidden controls for static problem difficulty;
- out-of-fold predictions with paired bootstrap comparison;
- explicit positive-control replication;
- locked confirmation runners that stop automatically after a decisive negative.

The reusable lesson is the **validation discipline**, not the failed trajectory-fate claim.

---

## 11. Final archive decision

```text
Topic:       DLM Trajectory Fate
G0:          positive exploratory signal
G1-A:        partial directional holdout only
G1-B:        FAIL_BOTH on full independent GSM1K
Positive ctl: passed strongly
Dream:       not run by design
Decision:    ARCHIVE / KILL BROAD CLAIM
```

No further GPU time should be spent trying to recover the original claim.

If a future project revisits DLM trajectories, it should start from a **new natural question with an independently motivated primary measurement**, not from another search for a layer/step/lead combination that restores this result.

## Related project records

- [`README.md`](README.md) — original question and G0 pipeline
- [`G0_RESULTS.md`](G0_RESULTS.md) — exploratory G0 results
- [`SECOND_STAGE_PLAN.md`](SECOND_STAGE_PLAN.md) — locked G1 design
- [`STAGE2_PROTOCOL_REVISION.md`](STAGE2_PROTOCOL_REVISION.md) — correction of the underpowered preflight gate
- [`SECOND_STAGE_RESULTS.md`](SECOND_STAGE_RESULTS.md) — executed Stage-2 results and final `FAIL_BOTH`
