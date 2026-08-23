# 14 — Does Power-Law Learning Need a Persistent Head?

**Status:** **ARCHIVED — valid full G0, `KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT`**

- [Frozen G0 result](./G0_RESULT.md)
- [Archive summary](./ARCHIVE_SUMMARY.md)
- [Locked validation contract](./VALIDATION.md)
- [Design history](./DESIGN_HISTORY.md)
- [Locked config](./LOCKED_CONFIG.json)

## Scientific question

> When a power-law skill distribution makes compositional learning possible, is local frequency asymmetry enough, or must the **same skills remain high-frequency for long enough** to scaffold the rest?

The seed work showed a strong power-law advantage in compositional S5 state tracking and stage-wise head-to-tail learning. Topic 14 isolated a natural temporal interpretation of that phenomenon: perhaps the same head skills must stay privileged for long enough to become a scaffold.

The project was designed around a deliberately simple causal test rather than a large mechanism stack.

## Decisive intervention

For every locked replication seed, Slow and Fast started from the same model and AdamW branch state and received the **same actual finite power-law minibatch multiset**.

Slow:

```text
A0 A1 ... A(P-1) B0 B1 ... B(P-1)
```

Fast:

```text
A0 B0 A1 B1 ...
```

Matched exactly:

- branch model state;
- AdamW state;
- finite minibatch contents and labels;
- A/B maps and counts;
- total optimizer steps;
- constant post-branch LR;
- frozen uniform evaluation panel.

Only temporal ordering / head persistence changed.

The five full-G0 seeds also used different predeclared random rank-to-skill base mappings so one arbitrary skill assignment could not be mistaken for a general law.

## Frozen full G0 result

All 5 seeds and all 20 arms passed integrity checks.

The prerequisite power-law effect was extremely strong:

```text
median Static - Uniform exact-AUC = +0.9300
positive seeds                    = 5/5
```

So the experimental object was clearly alive.

Primary Slow-vs-Fast exact-AUC contrasts:

```text
seed 0   -0.0325
seed 1   +0.7106
seed 2   +0.0095
seed 3   +0.0340
seed 4   -0.0395

median   +0.0095
4/5 seeds satisfy |gap| <= 0.06
```

Frozen decision:

```text
KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT
```

G1 was not run.

## Interpretation

This is a substantive null, not a prerequisite failure.

The clean testbed exhibits an enormous power-law learning advantage, but changing how long one head identity persists does not produce a stable effect across the locked replications.

The strongest supported statement is therefore:

> **In this S5 regime, local / instantaneous power-law asymmetry is sufficient for the main learning advantage; persistent head identity is not a generally important ingredient.**

This does not claim that temporal order never matters in compositional learning. It rejects the registered project-level claim that a stable, long-lived head is a broadly necessary or strong explanation for the power-law effect.

## Why seed 1 does not rescue the topic

Seed 1 showed a large `+0.7106` Slow advantage, and it is preserved in the result record.

However, four independently locked mapping/seed replications are near zero, the sign is not stable, and the frozen median is `+0.0095`.

Following seed 1 now would require searching over map identity, head structure, persistence schedules, or algebraic properties after seeing the outcome. That would be a new post-hoc question, not confirmation of Topic 14.

If future independent evidence predicts a specific structural property that should make persistent heads matter, register that as a new topic with its own locked test.

## Why the project stops

Topic 14 actually passed the standards we want from topic selection:

- the question was natural before mentioning a model or probe;
- the prerequisite phenomenon was strong;
- the main intervention was one clean contrast;
- the observable was direct;
- a positive result would have been surprising and valuable;
- the null result answers the registered question instead of destroying the experimental object.

The hypothesis simply did not hold broadly enough.

Do not reopen it by sweeping alpha, map pairs, head definitions, persistence lengths, models, architectures, or alternate primary metrics.

The full implementation, tests, validation contract and unused preregistered G1 launcher are preserved for reproducibility, but further experimentation under the existing hypothesis is closed.
