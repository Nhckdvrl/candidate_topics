# Coverage Collapse vs. Latent Alternative Paths

## Status

Candidate topic. **High-risk / high-reward.** The broad claim "post-training causes representation collapse" is already crowded in 2026. This topic is only worth pursuing in a much narrower form:

> **At a known reasoning decision point, does post-training remove information about an unchosen viable branch, or merely suppress its accessibility at the output?**

The topic should be abandoned immediately if branch-specific alternative information cannot be reliably measured in the base model.

---

## 1. Background

### Seed paper 1: reasoning coverage collapses after post-training

**Why Do Reasoning Models Lose Coverage?** studies the familiar phenomenon that post-training can improve `pass@1` while reducing `pass@k` and solution diversity. The paper uses controlled graph-reasoning tasks with explicit **forks in the road**, making it possible to know exactly where multiple viable reasoning branches exist.

The established behavioral phenomenon is:

> post-training increasingly commits the model to a smaller subset of reasoning paths.

Paper / code:
- https://arxiv.org/abs/2605.17026
- https://github.com/NNHieu/reasoning_forks

The repository already provides synthetic graph tasks, training code, intermediate checkpoints, coverage evaluation, and a 1,000-example test set.

### Seed paper 2: unchosen futures can be represented internally

**Are Language Models Aware of the Road Not Taken? Token-level Uncertainty and Hidden State Dynamics** studies whether reasoning-model hidden activations contain information about futures that are not ultimately selected. It finds that hidden states can predict aspects of future outcome distributions and that alternative outcomes can sometimes be made more accessible through activation intervention.

Paper:
- https://arxiv.org/abs/2511.04527

So two separate facts are already established:

1. post-training can collapse behavioral reasoning coverage;
2. unchosen alternatives can be represented in hidden states.

The adjacent missing question is:

> **What happens to branch-specific latent alternatives while behavioral coverage is collapsing during training?**

---

## 2. Why the claim must be narrow

A generic story about "representation collapse" is not enough anymore.

Recent 2026 work already studies:

- output diversity collapse across post-training stages;
- hidden-state low-rank / anisotropy / homogenization during sequential post-training;
- suppression of exploratory reasoning primitives such as hypothesizing and backtracking.

Therefore this project should **not** claim novelty from showing that hidden representations become less diverse.

The target variable must be much more specific:

> **Does the model still encode whether a concrete unchosen branch at a concrete graph decision point is viable?**

The distinction we care about is:

- **erasure**: information about the alternative branch itself disappears;
- **suppression**: the branch remains internally represented, but normal decoding stops selecting it;
- **suppression -> erasure**: accessibility collapses first, branch information disappears later.

---

## 3. What we want to study

Consider a graph-reasoning state with two locally plausible branches:

```text
        root
       /    \
      A      B
             \
              ...
```

Suppose both branches are represented in the task structure, and one or more branches can lead to a valid solution.

At a known fork, define a branch-specific target:

`y(v, b) = 1` if branch `b` can still lead to a valid solution from decision state `v`, otherwise `0`.

The core question is:

> **As SFT / RL increases commitment to one branch, does the hidden state still contain information about the viability of the branches that are no longer sampled?**

This is intentionally different from measuring output entropy or softmax support. A low-probability token still existing in the vocabulary is not evidence that the model internally represents the downstream viability of that branch.

---

## 4. Exact measurements

### 4.1 Behavioral coverage

For each training checkpoint, measure:

- `pass@1`;
- `pass@k` (pilot: `k=16`, later increase if needed);
- first-decision branch entropy;
- empirical frequency with which each viable branch is sampled.

A simple branch entropy measure is:

`H_branch(t) = - Σ_b p_t(b) log p_t(b)`

This establishes when behavioral coverage is collapsing.

### 4.2 Branch-specific representation probe

At the first controlled decision point, extract the hidden state `h_t(v)`.

For each candidate branch `b`, pair that state with a branch representation `e_b` and predict branch viability:

`s_t(v,b) = f(h_t(v), e_b)`

A minimal implementation could use a bilinear or low-capacity probe:

`s(h,b) = sigmoid(h^T W e_b)`

The important evaluation quantity is not generic hidden-state rank. It is **branch-specific viability information**:

- AUROC for viable vs non-viable candidate branches;
- recall of viable but behaviorally unchosen branches;
- calibration of branch-viability scores.

Define a simple summary:

`R_latent(t) = recall of viable alternative branches from hidden representations`

Then compare:

`H_branch(t)` vs `R_latent(t)` across training.

### 4.3 Accessibility / causal test

A probe failure cannot prove that information is absent, so any strong paper claim should include an accessibility test.

Learn a branch-related activation direction or intervention from earlier checkpoints / controlled examples and intervene at the fork:

`h' = h + α d_b`

Measure:

- change in `log p(branch b)`;
- probability of entering branch `b`;
- probability of successfully completing the task through branch `b`.

The strongest evidence for **suppression rather than erasure** would be:

> natural decoding almost never takes branch `b`, but a small hidden-state intervention reliably restores the branch and its successful continuation.

---

## 5. Minimal validation experiment

This topic has a stricter G0 than the other candidates.

### G0: can branch viability be measured at all?

Before studying training dynamics, use the base / early model only.

If a low-capacity probe cannot reliably distinguish viable from non-viable candidate branches, then the proposed latent variable is not operationalized and the project should stop.

### Model

Start with **Qwen2.5-0.5B** using the setup from `reasoning_forks`.

The small model is enough for the first feasibility test and makes dense checkpoint analysis cheap.

### Dataset

Use the official **1,000-example graph test set** from the seed repository.

### Training trajectory

For the first real dynamics experiment, use SFT only. Do not begin with GRPO.

Suggested checkpoints:

`epoch 0, 1, 2, 4, 6, 8, 10, 12, 14, 16`

Only add RL checkpoints after the SFT result is interpretable.

### For every checkpoint

1. sample `k=16` reasoning trajectories per test graph;
2. compute `pass@1`, `pass@16`, and first-branch entropy;
3. identify the controlled first decision point;
4. save the hidden state at that point;
5. create `(state, candidate branch)` pairs with graph-ground-truth viability labels;
6. train/evaluate the branch-viability probe using graph-ID splits;
7. track alternative-branch recall over training.

### Pilot outputs

The first meaningful experiment should produce four plots:

1. **pass@1 / pass@16 vs training step**
2. **first-branch entropy vs training step**
3. **branch-viability probe AUROC vs training step**
4. **behavioral coverage vs latent alternative recall**

Only if these show a coherent pattern should we proceed to activation intervention.

---

## 6. Decision rule

### Result A: behavioral collapse, latent alternatives preserved

`pass@k ↓`, branch entropy `↓`, but branch-specific viability remains readable.

Interpretation:

> post-training changes which path is selected before it removes knowledge of the alternatives.

This is the strongest and cleanest version of the **suppression** story.

Paper potential: **high**, especially if intervention can recover the suppressed branch.

### Result B: behavioral and latent collapse occur together

`pass@k ↓` together with branch-viability information `↓`.

Interpretation:

> coverage loss is accompanied by branch-specific representational pruning, not merely output sharpening.

Paper potential: **moderate to high**, but only if the result is branch-specific and distinct from generic representation-collapse metrics.

### Result C: latent branch information collapses before behavior

The branch probe degrades before `pass@k` or output entropy visibly collapses.

Interpretation:

> loss of latent alternatives may be an early precursor to later behavioral mode collapse.

Paper potential: **high and surprising**, if robust across seeds/models.

### Result D: branch-specific information is not measurable or is unstable

Possible symptoms:

- base-model AUROC near chance;
- results depend strongly on probe capacity;
- no consistent relation to graph-ground-truth branch viability;
- apparent effects disappear under graph-ID splits.

Interpretation:

> the proposed latent variable is not established in this controlled setting.

Action: **stop the project**. Do not replace the failed branch-specific question with a generic CKA/effective-rank collapse paper.

---

## 7. Key falsification threshold

The project should not even enter the training-dynamics stage unless the base / early model passes this test:

> **Can we reliably decode the viability of an unchosen candidate branch at a known graph fork?**

A weak or chance-level result here kills the topic.

This is deliberate: the project should have a cheap way to falsify its central premise before investing in many checkpoints or RL runs.

---

## 8. Why this topic may matter

Post-training may reduce reasoning diversity for at least two very different reasons:

1. the model genuinely loses internal information about alternative routes;
2. the alternatives remain represented, but the policy becomes increasingly unable or unwilling to access them.

Those mechanisms imply very different interpretations of coverage collapse and very different possibilities for recovery.

The useful scientific question is therefore not simply:

> "Does post-training reduce diversity?"

That is already established.

It is:

> **When a reasoning path disappears from behavior, has it disappeared from the model?**
