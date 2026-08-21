# Archive Summary — Topic 05: Temporal Forgetting Re-entry

**Final status: ARCHIVED / STOPPED AT CONCEPTUAL IDENTIFICATION GATE**

**No hypothesis-level empirical conclusion was reached.** The project was stopped during partial checkpoint sampling, before scoring, robust forgotten-set construction, the preregistered gate, re-entry evaluation, or teacher-forced likelihood analysis.

## Original question

The topic started from a natural distinction:

> When a learner once solved a problem reliably but later fails, has the former skill actually been lost, or has it merely become inaccessible?

The proposed use of reasoning-model training checkpoints looked attractive because an earlier checkpoint from the same lineage can provide a concrete previously successful solution trace. The planned experiment therefore attempted to distinguish **genuine competence loss** from **lost access / lost route selection** by giving the final model prefixes from its own earlier correct trajectory.

The intended pipeline was:

1. establish robust temporal forgetting with repeated sampling across eight public Qwen2.5-7B RL checkpoints;
2. freeze forgotten / never-correct / stable-correct groups;
3. compare final-model continuation under old-self, other-correct, final-wrong, and matched never-correct prefixes;
4. use teacher-forced old-route suffix NLL as a second diagnostic;
5. optionally test relearning savings only after the primary behavioral analysis.

The infrastructure for this pipeline is preserved in this directory.

## Why the topic was stopped

The failure is **conceptual identification failure**, not a negative experimental result.

The central target quantity is something like:

> whether the final model still possesses the old skill when solving the original problem without external help.

But the proposed intervention changes the problem condition itself:

\[
P(\mathrm{solve}\mid x) \neq P(\mathrm{solve}\mid x+\mathrm{correct\ prefix}).
\]

A correct prefix can improve performance for many reasons that do not imply reactivation of a retained old skill:

- it reduces the remaining search space;
- it supplies intermediate variables or useful partial results;
- it rules out competing wrong trajectories;
- it changes local token compatibility and continuation probability;
- it changes the effective difficulty and conditional distribution of the task.

Therefore even a large rescue effect does not identify the intended mechanism.

## Why the proposed controls do not solve the problem

### 1. `old-self > baseline` is not diagnostic

A correct partial solution makes the task easier. Rescue relative to the uncued problem can therefore arise even when the old competence is no longer available without the cue.

### 2. `old-self > other-correct` still does not identify re-entry

An old-self prefix may have unusually favorable lexical, stylistic, token-level, or local continuation compatibility with the final model. A difference from another correct solution is therefore not sufficient evidence that the model has re-entered a historically preserved internal skill.

### 3. The “old route” is not a stable observable object

A generated continuation need not remain on one identifiable route. It may evolve as

```text
old-like state -> new state -> different strategy -> correct answer
```

or switch repeatedly among strategies. A correct continuation after an old prefix therefore does not establish that the former route itself was recovered.

### 4. `final-wrong` and `never-correct` controls remove only local alternatives

These controls can test generic guidability or whether some prefixes are more useful than others, but they do not reveal whether the uncued final model internally retains the former competence.

### 5. Teacher-forced NLL is conditional on the intervention

A favorable value of

\[
p_{final}(r^{old}_{k+1:n}\mid x,r^{old}_{1:k})
\]

only says that the old continuation is probable **after the old prefix has already been supplied**. It does not imply that the same skill is available, represented, or selectable under the original uncued condition \(x\).

Thus the behavioral rescue and the likelihood diagnostic share the same identification limitation.

## What was actually run

Only part of the checkpoint-sampling stage was executed.

The run was stopped before:

- primary scoring;
- full repeated-sampling state classification;
- forgotten / never-correct / stable-correct set construction;
- the hard feasibility gate;
- trace freezing and matched-control construction;
- re-entry experiments;
- teacher-forced NLL analysis;
- confirmation or relearning experiments.

Accordingly:

> **there is no scientific result to report about whether Temporal Forgetting reflects storage loss, retrieval failure, route-selection failure, or competence erosion.**

Partial generation logs/raw outputs were kept locally for forensic reference rather than interpreted scientifically. The downloaded public checkpoints were removed from the local Hugging Face cache and the GPUs were released from this project.

## What was still useful

Although the question was stopped, the implementation work produced reusable infrastructure:

- repeated-sampling checkpoint state construction;
- strict complete-checkpoint coverage checks;
- official MATH-500 prompt/stop alignment;
- PRIME/MATH scoring plus optional judge fallback;
- multi-GPU / multi-node embarrassingly parallel generation;
- deterministic trace freezing;
- matched forgotten/never-correct controls;
- assistant-prefix continuation generation;
- teacher-forced suffix likelihood measurement;
- discovery/confirmation separation and cluster bootstrap analysis;
- descriptive checkpoint state-dynamics tooling.

These components may be reused by future topics, but they should not be treated as evidence for Topic 05.

## Main lesson for topic selection

The important mistake happened **before the experiment**.

The natural question — "forgotten or merely inaccessible?" — is interesting, but an interesting distinction is not yet an identifiable research question. The proposed observable must uniquely discriminate the competing explanations.

Here, every apparent success of the key intervention remained compatible with a simpler statement:

> giving the model part of a correct solution changes the task and makes solving easier.

Adding more variants of the same cue cannot repair that problem. Old-self, other-correct, final-wrong, never-correct, more prefix lengths, or more NLL measurements all remain conditioned on a changed input distribution.

The selection lesson is therefore:

> **Before investing in a candidate, ask whether the proposed measurement can in principle identify the claimed latent distinction, not merely whether it has many controls.**

A future revival would require a genuinely different identification strategy that can distinguish retained competence from lost competence **without defining retention by performance after supplying the missing solution information**. Merely adding stronger matching, more checkpoints, more samples, hidden-state probes, or additional prefix controls is not sufficient reason to reopen this topic.

## Repository disposition

The folder is intentionally retained as an archive containing:

- the original motivation and design (`README.md`, historical validation/runbook files);
- the explicit identification-failure record (`VALIDATION_FAILURE.md`);
- reusable code and tests;
- this final archive summary.

**Final decision: do not continue validation.**
