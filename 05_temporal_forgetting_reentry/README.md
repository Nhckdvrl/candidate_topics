# Topic 05 — Temporal Forgetting: Lost Skill or Lost Entry Point?

**Status: ARCHIVED / STOPPED AT CONCEPTUAL IDENTIFICATION GATE**

> When a learner once solved a problem reliably but later fails, has the former skill actually been lost, or has it merely become inaccessible?

This topic is **no longer under validation**. It was stopped because the proposed experiment cannot identify the intended latent distinction, not because the hypothesis was empirically falsified.

See:

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — final project summary, failure analysis, and lessons;
- [`VALIDATION_FAILURE.md`](./VALIDATION_FAILURE.md) — contemporaneous stop record from the partial server run;
- [`VALIDATION.md`](./VALIDATION.md) — historical preregistered validation design;
- [`RUNBOOK.md`](./RUNBOOK.md) and [`SERVER_AGENT_PROMPT.md`](./SERVER_AGENT_PROMPT.md) — historical execution plan;
- `code/`, `scripts/`, and `tests/` — preserved reusable infrastructure.

## Original motivation

Li et al., **Temporal Sampling for Forgotten Reasoning in LLMs** (ACL 2026), reports that reasoning problems can be solved at intermediate training checkpoints and then fail at the final checkpoint. This suggested a natural question:

> Does later failure mean the former competence disappeared, or can the final model no longer access / select a route it once possessed?

The proposed validation used eight public Qwen2.5-7B GRPO checkpoints and planned to compare final-model continuation from:

- its own earlier correct reasoning prefix (`old-self`);
- another correct solution;
- its own final wrong prefix;
- a matched never-correct control;
- plus teacher-forced old-route suffix NLL.

The code was substantially hardened before execution: repeated-sampling state definitions, complete checkpoint coverage, official MATH-500 prompt/scorer alignment, token-budget matching, assistant-prefix continuation, F/N matching, discovery/confirmation separation, and multi-GPU sharding were all implemented.

## Why this cannot answer the question

The key problem is structural:

\[
P(\mathrm{solve}\mid x) \neq P(\mathrm{solve}\mid x+\mathrm{correct\ prefix}).
\]

Providing a correct prefix changes the task. It can shrink the search space, expose intermediate variables, exclude wrong strategies, or simply create favorable local continuation statistics. Therefore successful continuation does not uniquely imply that an uncued old skill remained available and was "re-entered."

The concept of an `old route` is also not a stable observable object: a continuation can move through old-like, new, and different reasoning states before reaching a correct answer. Likewise, favorable teacher-forced NLL only shows compatibility **conditional on already supplying the old prefix**.

Consequently, old-self / other-correct / final-wrong / never-correct controls can rule out some simpler confounds, but they do not solve the central identification problem.

## Experimental disposition

Only part of checkpoint sampling was completed. The project stopped before:

- scoring;
- robust forgotten-set construction;
- the preregistered feasibility gate;
- re-entry evaluation;
- teacher-forced NLL analysis;
- any confirmation experiment.

Therefore **no empirical conclusion is reported** about storage loss, retrieval failure, route-selection failure, or competence erosion.

Partial raw outputs and logs were kept locally only for forensic reference. Downloaded checkpoints were cleaned from the local cache and GPU resources were released.

## Final decision

**Do not continue this topic under the current identification strategy.**

A future revival would require a genuinely different way to distinguish retained from lost competence without defining retention by performance after supplying part of the missing solution. More prefix controls, more samples, more checkpoints, or a post-hoc hidden-state probe do not by themselves repair the conceptual problem.
