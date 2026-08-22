# 10 — What Does Diffusion Confidence Actually Know?

**Status:** REGISTERED CANDIDATE — CHEAP G-0 FIRST

## Natural question

> Does diffusion-language-model confidence primarily track whether a reasoning trajectory is internally coherent, or whether its conclusion is externally correct?

This is intentionally narrower than the generic claim that DLM confidence is "not calibrated." The scientific target is whether confidence contains a signal for **trajectory-internal consistency that is partially separable from final-answer correctness**.

## Seed evidence

### Seed A — the confidence paradox

Recent work reports that diffusion-LM confidence can be badly calibrated as a probability of correctness while still separating correct from incorrect mathematical solutions surprisingly well. The same work interprets this as evidence that DLM confidence may reflect structural consistency of the reasoning path rather than ordinary answer probability.

It also reports asymmetric sensitivity to different perturbations: arithmetic contradictions reduce confidence much more than simple factual answer swaps.

### Adjacent pressure

Very recent DLM work has already shown that local token confidence can diverge from global reasoning correctness and that exposed confidence can diverge from internal representations. Therefore this project must not ask the broad question "is confidence equal to correctness?" That is already crowded.

The unresolved identification question is more specific:

> **Holding external correctness fixed, does breaking internal consistency reduce confidence? Holding internal consistency fixed, does external correctness matter less?**

## Fixing the naive 2×2 identification bug

A naive construction such as "correct reasoning + wrong final answer token" is **not** internally consistent: if the reasoning concludes 42 but the final answer says 37, the trajectory itself is contradictory.

The experiment therefore orthogonalizes:

- **internal consistency**: whether later steps correctly follow from the trajectory's own earlier premises/values;
- **external correctness**: whether those premises/values and final answer match the real problem.

The four cells should be constructed programmatically:

### A — consistent + correct

A normal correct derivation and correct final answer.

### B — inconsistent + correct

Introduce an intermediate arithmetic contradiction or unsupported value, but later force the trajectory back to the externally correct final answer. This is a lucky-correct but internally broken trajectory.

### C — consistent + wrong

Introduce one early arithmetic error, then **propagate that wrong value consistently through every downstream step** so that the final answer is wrong but the trajectory is internally coherent relative to its mistaken premise.

### D — inconsistent + wrong

Introduce an internal contradiction and end with an externally wrong answer that is not coherently implied by the preceding trajectory.

The key comparison is therefore not "reasoning correct vs answer correct"; it is **self-consistency vs world correctness**.

## G-0: one locked factorial contrast

Use arithmetic word problems whose dependency chain can be parsed or generated so that perturbations and downstream propagation are deterministic rather than judged by another LLM.

For each base problem, construct matched A/B/C/D trajectories of the same overall format and approximately the same token budget.

Score them with the exact frozen confidence protocol used by the target DLM.

Define two paired effects:

`Delta_consistency` = confidence change from consistent to inconsistent while external correctness is held fixed.

`Delta_correctness` = confidence change from externally correct to externally wrong while internal consistency is held fixed.

Primary question:

> Is `Delta_consistency` substantially larger and more stable than `Delta_correctness`?

The strongest qualitative signature would be:

- **coherent-but-wrong** trajectories retain high confidence;
- **lucky-correct-but-incoherent** trajectories receive low confidence.

## Why this is a clean question

No hidden states, no learned verifier, no post-hoc error taxonomy, no threshold search, and no need to infer a latent "reasoning quality" label.

Internal consistency is changed by a known intervention and downstream arithmetic is deterministically propagated.

## Kill line

Kill the topic if the A/B/C/D construction cannot be made programmatic and unambiguous without an LLM judge or extensive manual labeling.

Also kill it if the first locked paired experiment shows that confidence is dominated by final-answer correctness or shows no stable distinction between coherent-wrong and incoherent-correct trajectories.

Do not rescue the project by adding many error categories, alternate confidence definitions, prompts, or hand-matched subsets.

## Why a strong result matters

A robust result would identify a genuinely different signal in DLM decoding:

> **native diffusion confidence behaves partly like a detector of trajectory-internal consistency rather than merely a probability that the final answer is correct.**

That would matter scientifically before any downstream method is proposed, because it changes what the model's confidence can be interpreted as measuring.
