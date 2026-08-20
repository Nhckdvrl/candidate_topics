# DLM Trajectory Fate: Can Hidden States Predict Recovery and Overwrite Before They Happen?

## Status

Candidate topic. **Highest-priority pilot** because the phenomenon, probe setup, models, and code all already exist. The key test is whether hidden states contain information about the **future fate of the current denoising state**, not merely final correctness.

---

## 1. Background

Diffusion Language Models (DLMs) expose an explicit denoising trajectory, which makes it possible to study how an answer evolves over generation time.

### Seed paper 1: temporal oscillation is real

**Time Is a Feature** (ICLR 2026) shows that intermediate DLM predictions can oscillate during denoising. A sample may become correct and later be overwritten, e.g.:

`wrong -> correct -> wrong`

So current surface correctness is not monotonic. A correct intermediate answer is not necessarily stable, and an incorrect intermediate answer is not necessarily doomed.

Paper / project:
- https://arxiv.org/abs/2508.09138
- https://github.com/aim-uofa/dLLM-MidTruth

### Seed paper 2: hidden states predict eventual correctness

**Probing Functional Correctness in Diffusion Language Models** (ACL 2026 SRW) probes DLM hidden states at multiple denoising steps and finds that they increasingly encode whether the **final output** will be correct.

The paper uses a simple and reusable pipeline:

- LLaDA-8B-Instruct / Dream-7B-Instruct;
- hidden-state extraction at selected denoising steps;
- mean pooling over generation regions;
- PCA to 64 dimensions;
- logistic regression;
- AUC evaluation.

Paper / code:
- https://aclanthology.org/2026.acl-srw.15/
- https://github.com/guan404ming/dllm-probing

The missing neighboring question is:

> **Given the current surface state, does the hidden representation already encode what will happen to it next?**

This is a one-step rotation from **eventual correctness** to **trajectory fate**.

---

## 2. What we want to study

The main idea is to condition on the current surface correctness and ask about the future transition.

### Case A: the current answer is wrong

Among states with `current_correct = 0`, distinguish:

- **recoverable**: the trajectory later reaches a correct answer;
- **doomed**: the trajectory never recovers, or ends incorrect.

Question:

> Among two states that are both wrong right now, can the hidden state tell which one will recover?

### Case B: the current answer is correct

Among states with `current_correct = 1`, distinguish:

- **stable-correct**: the answer remains correct;
- **will-be-overwritten**: a later denoising step destroys the correct answer.

Question:

> Among two states that are both correct right now, can the hidden state tell which one is fragile and will later be overwritten?

This conditional design is important. A naive four-way classifier could cheat by mostly learning whether the current answer is correct or incorrect. We specifically want **future-fate information beyond current surface correctness**.

---

## 3. Exact measurements

### 3.1 Hidden-state probe

Follow the existing DLM probing pipeline as closely as possible.

For denoising step `t`, layer `l`, and pooled generation region `r`, extract:

`h_t^(l,r)`

Then use:

`hidden state -> PCA(64) -> standardization -> logistic regression`

Evaluate two conditional binary probes:

1. `AUC_recover(t,l)`: recoverable vs doomed among currently wrong states;
2. `AUC_overwrite(t,l)`: stable-correct vs will-be-overwritten among currently correct states.

The split must be by **problem ID**, never by `(problem, denoising step)`, otherwise neighboring states from the same problem can leak between train and test.

### 3.2 Lead time

The strongest claim would not be merely that fate is readable at the moment of transition, but that it is readable **before the surface transition happens**.

Let `t*` be the denoising step where recovery or overwrite becomes visible in the decoded answer. Define:

`lead_time = t* - t`

Then plot predictive performance as a function of lead time:

`AUC(lead_time)`

A useful signal tens of denoising steps before the transition would be much more interesting than a probe that only succeeds at `t*`.

### 3.3 Surface baselines

The hidden-state probe must beat simple observable quantities such as:

- mean token entropy;
- max probability / confidence;
- probability of the current answer tokens;
- fraction of unmasked tokens;
- denoising step index;
- simple prompt/task difficulty features.

The central question is whether the hidden state contains **extra trajectory-fate information**, not whether confidence correlates with future success.

---

## 4. Minimal validation experiment

### Model

Start with **LLaDA-8B-Instruct** only.

Do not begin with multiple DLM families. The first goal is to falsify or support the core premise as cheaply as possible.

### Dataset

Use **1,000 GSM8K test problems**.

This is not intended as a new benchmark. GSM8K is chosen because both seed lines already establish relevant DLM behavior on mathematical reasoning tasks.

### Denoising setup

Use 128 denoising steps.

For the pilot, save intermediate outputs and hidden states at a denser subset than the original probing paper, for example:

`0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 120, 124, 127`

### Layers

To reduce storage, begin with upper LLaDA layers where correctness information is already known to be strongest, e.g.:

- layer 22
- layer 25
- layer 28

### For every saved denoising step

Record:

1. current decoded answer;
2. current correctness;
3. final correctness;
4. whether a future recovery occurs;
5. whether a future overwrite occurs;
6. entropy / confidence baselines;
7. hidden-state features from the selected layers.

From this, automatically construct four state categories:

- wrong + recoverable;
- wrong + doomed;
- correct + stable;
- correct + overwritten later.

### Pilot outputs

The minimum useful pilot should produce four plots:

1. **Class counts over denoising time**
   - are there enough recover / overwrite states to make the question statistically meaningful?
2. **Recoverability AUC over denoising time**
3. **Overwrite-risk AUC over denoising time**
4. **AUC vs lead time**, compared with entropy/confidence baselines

---

## 5. Decision rule

### Strong positive result

Continue if hidden states predict one or both trajectory-fate variables **before the corresponding surface transition**, and clearly outperform simple uncertainty baselines.

The strongest case would look like:

- the decoded answer is still wrong;
- several denoising steps remain before recovery;
- yet `P(recoverable | h_t)` is already high.

Or:

- the decoded answer is currently correct;
- no visible corruption has happened yet;
- yet the hidden state strongly predicts that the answer will later be overwritten.

This would suggest that DLM trajectories contain a latent notion of **state fate / stability** that becomes readable before it is visible at the surface.

### Partial positive result

Still potentially interesting if only one direction is predictable:

- recovery predictable, overwrite not predictable;
- overwrite predictable, recovery not predictable.

That asymmetry itself could reveal different mechanisms for correction and corruption.

### Stop

Stop if:

- the probes only become predictive at or after the visible transition;
- entropy/confidence explains essentially all predictive power;
- the number of recover/overwrite examples is too small for stable estimation;
- performance disappears under problem-level splits.

Do not inflate a late, near-transition signal into a claim that the model "knows the future".

---

## 6. If the pilot works

Only after the basic representation result is established:

1. replicate on Dream-7B-Instruct;
2. extend to ARC / MATH / Countdown or other tasks already used by the seed work;
3. test whether a trajectory-fate direction generalizes across tasks or models;
4. optionally perform activation steering at intermediate denoising steps.

A causal extension could ask whether steering toward a "recoverable" or "stable" direction actually changes:

- probability of recovery;
- probability of later overwrite;
- final task correctness.

That would move the story from **readable fate representation** to **causally relevant trajectory state**.

---

## 7. Why this topic may matter

DLMs are unusual because their intermediate answers can improve and deteriorate within one generation trajectory. Existing work separately establishes:

- temporal answer oscillation at the surface;
- hidden-state information about eventual correctness.

The unanswered adjacent question is whether the model's internal state already distinguishes:

> **"wrong but recoverable" from "wrong and doomed", and "correct and stable" from "correct but fragile".**

This keeps the model, task, measurement paradigm, and denoising axis almost unchanged; the main novelty is the scientific variable being probed.
