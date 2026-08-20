# Topic 05 — Temporal Forgetting: Lost Skill or Lost Entry Point?

**Status: CANDIDATE / VALIDATION NOT YET RUN**

> When a learner once solved a problem reliably but later fails, has the solution competence been erased, or has the learner mainly lost access to the old route?

## Why this is a natural question

A failure to recall does not uniquely imply that a memory or skill is gone. Cognitive theories have long distinguished **storage / availability** from **retrieval / accessibility**.

Modern reasoning-model training gives an unusually clean experimental system: for the *same model lineage and same problem*, we can keep both the later checkpoint that now fails and an earlier checkpoint that previously produced a correct reasoning trajectory.

That lets us ask:

> **Can the later model re-enter and complete its own former solution route?**

## Seed phenomenon

Li et al. (ACL 2026), *Temporal Sampling for Forgotten Reasoning in LLMs*, shows **Temporal Forgetting** across SFT and RL:

- many questions are correct at an intermediate checkpoint but wrong at the final checkpoint;
- over 20% of final errors are, on average, previously solved;
- the effect occurs across Qwen2.5 1.5B/7B, SFT/GRPO, and several reasoning benchmarks.

The authors released eight Qwen2.5-7B GRPO checkpoints (`step_32` ... `step_256`), 64 sampled responses per checkpoint for AIME24/AIME25/AMC, and evaluation infrastructure for MATH-500 and OlympiadBench.

Their remedy is **Temporal Sampling**: keep old checkpoints at inference time and sample from them.

Our question is different:

> **What kind of forgetting produced the old/new discrepancy in the first place?**

## Nearest collision

Li & Goyal (ICLR 2026), *Off-Trajectory Reasoning*, studies **Guidability**: can a model continue a correct partial trace supplied by another, often stronger, model on problems beyond its own capability? Models generally struggle.

The proposed experiment is deliberately different:

- the problem is known to have been within **this model lineage's own earlier capability**;
- the cue is **its own earlier successful trajectory**;
- forgotten problems are compared to matched **never-correct** problems.

If old-self prefixes help forgotten items no more than they help never-correct items, we must not call the result "retained skill."

## Why greedy flips are not enough

A problem is a primary **robust forgotten** item only if repeated sampling supports both states:

`P_old(correct) >= 0.75` and `P_final(correct) <= 0.125`.

Default pilot: at least 8 samples per checkpoint. With the released 64-sample data, use empirical rates directly.

This is the first kill gate: if strong temporal forgetting nearly disappears under repeated sampling, stop.

## Three complementary diagnostics

No single cueing experiment can prove that a latent skill is "stored." The validation therefore triangulates three observables **predeclared in advance**.

### A. Re-entry curve

For old correct trajectory `r_old`, feed the final model `problem + prefix(r_old,k)` and measure `R(k)=P(correct | old prefix k)`.

The key comparison is not `R(k)>R(0)` by itself. Any correct hint can make a problem easier.

Compare:

1. forgotten + **old-self correct prefix**;
2. forgotten + **other-correct prefix** matched in length/information;
3. forgotten + final model's **own wrong prefix**;
4. never-correct matched item + correct prefix.

A selective advantage for old-self / forgotten pairs is evidence for route-specific re-entry.

### B. Old-route continuation likelihood

Teacher-force the final model on the old correct trace and compute mean per-token NLL of the remaining old solution after each prefix point.

If free generation fails but the old continuation remains substantially more likely than matched never-correct continuations, the old route is still compatible with the final policy even though it is not self-selected.

### C. Relearning savings

Secondary only. Give identical tiny corrective updates to forgotten old solutions and matched never-correct solutions. Faster reacquisition of forgotten solutions is an orthogonal savings diagnostic.

This diagnostic is preregistered now and is **not** a post-hoc rescue if A/B fail.

## Validation pipeline

```text
G-1  Robust forgetting exists under repeated sampling?
     ↓
G0-A Re-entry curves with locked controls
     ↓
G0-B Teacher-forced old-route likelihood
     ↓
G1   Relearning savings (secondary triangulation)
     ↓
G2   Replicate on SFT trajectory / second model or dataset
```

## Control groups

**F — robust forgotten**: earlier pass rate >= .75, final <= .125; choose latest qualifying old checkpoint.

**N — never-correct**: every checkpoint <= .125.

**S — stable-correct**: final >= .75 and at least one earlier checkpoint >= .75.

Match F/N on benchmark and final-model difficulty proxies.

## Prefix construction

Primary prefixes are cut at **reasoning-step boundaries**, not arbitrary tokens.

Reject any prefix that contains the final boxed answer, an explicit normalized gold answer statement, or the complete old solution.

Primary prefix levels: `0%, 10%, 25%, 50%`.

## Four informative outcomes

| Result | Interpretation |
|---|---|
| Short old-self prefix selectively rescues forgotten items; old continuation NLL is favorable | **lost entry / selection** is a major component |
| Generic correct prefixes rescue equally; no old-self advantage | final model remains guidable, but no evidence of route-specific retention |
| Only long prefixes rescue; old-route NLL worsens | partial competence erosion |
| Even near-complete non-answer-leaking prefixes fail and old-route NLL looks like never-correct controls | strong evidence that these cases are closer to genuine route/skill loss |

A clean negative therefore still characterizes the seed phenomenon instead of merely saying "our probe did not work."

## G-1 cheap screen

Use the official 64-response release first. It covers small AIME/AMC sets, so it may only support a smoke test.

For a proper pilot:

```text
model       UWNSL Qwen2.5-7B GRPO checkpoints
checkpoints 8
dataset     MATH-500 (primary), optionally OlympiadBench
samples     8 per problem/checkpoint for classification
```

For 500 problems this is `500 × 8 × 8 = 32,000` trajectories. After F/N/S sets are frozen, re-entry inference uses only the final checkpoint.

## Kill conditions

- fewer than ~50 robust forgotten items in the main 500-problem pilot;
- forgetting largely vanishes once repeated-sampling robustness is imposed;
- old-self and matched control prefixes cannot be made comparable without answer leakage;
- an exact recent work already performs old-checkpoint self-prefix re-entry with forgotten-vs-never-learned controls.

## Novelty boundary

Do **not** claim temporal forgetting itself, old-checkpoint diversity, or that correct hints help.

The intended contribution is:

> **use a model's own former successful trajectory, repeated-sampling state definitions, and never-correct controls to distinguish self-initiation failure from deeper loss of a previously demonstrated reasoning route.**

## Initial code

```bash
python code/build_forgotten_set.py --input results/checkpoint_samples.jsonl --output results/groups.jsonl
python code/build_reentry_prompts.py --groups results/groups.jsonl --output results/reentry_prompts.jsonl
python code/analyze_reentry.py --input results/reentry_generations.jsonl
```

See `VALIDATION.md` for schemas and locked thresholds.
