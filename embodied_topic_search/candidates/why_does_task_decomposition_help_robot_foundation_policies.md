# Why Does Task Decomposition Help Robot Foundation Policies?

> **DOWNGRADED / REMOVED FROM ACTIVE SHORTLIST (2026-08-24).**
>
> Source: Round 8. This question was never registered as a root Topic.

## Original question

> **When a hierarchical VLA greatly outperforms a flat VLA on long-horizon tasks, how much of the gain comes from genuine high-level planning/reasoning, and how much comes from repeatedly translating the global task back into atomic instructions that the low-level policy already knows how to execute?**

Sharper form:

> **Does the planner reason better, or does it keep the controller on-support?**

The intended decomposition was:

```text
planning / sequencing
vs
controller-support matching
vs
temporal handoff / reset
```

## Why it was downgraded

The broad mechanism angle became substantially more crowded during the 2026 audit:

- `What Matters in Orchestrating Robot Policies` directly studies hierarchical VLA components and low-level language steerability;
- `Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation` explicitly targets the semantic-plan / executable-kinematics gap using executable skill primitives and tractability constraints;
- hierarchical post-training work explicitly identifies planner-generated-subgoal / executor distribution misalignment and aligns the executor to planner outputs;
- compositional-generalization diagnosis already shows that low-level capability can exist while instruction steering fails to access it.

A carefully controlled causal decomposition could still be interesting, but the broad question no longer has enough empty conceptual space to justify a root Topic under this repository's selection rule.

Keeping it alive would increasingly require narrowing to a particular wording control, reset schedule, benchmark, or planner interface to avoid collision. That is exactly the pattern the search process is supposed to reject.

## Final status

```text
DOWNGRADED
NOT REGISTERED
NO EXPERIMENT PLANNED
```

The historical detailed version remains available in git history. Current search bookkeeping is in [`../README.md`](../README.md) and [`../2026-08-24_candidate_promotion_audit.md`](../2026-08-24_candidate_promotion_audit.md).