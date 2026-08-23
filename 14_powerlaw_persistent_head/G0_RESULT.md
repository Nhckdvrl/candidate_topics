# Topic 14 — Frozen full G0 result

## Final machine decision

```text
KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT
```

The formal G0 completed all five locked replication seeds and all 20 arms (`Uniform / Static / Slow / Fast` for each seed). All run-integrity checks passed. G1 was not run.

## Scientific question tested

> Once a power-law skill-frequency advantage is clearly present, does that advantage require the same high-frequency skills to remain privileged for a long time, or is local / instantaneous asymmetry sufficient?

The clean primary comparison was `Slow` versus `Fast` under an exact matched-data design:

- identical branch model and AdamW state within a seed;
- identical finite power-law minibatch multiset;
- identical A/B maps and counts;
- identical total optimizer steps and constant post-branch LR;
- identical frozen uniform evaluation panel;
- only the temporal order of the A/B batches differed.

Therefore this run directly tested temporal persistence, rather than confounding persistence with different data, counts, optimizer history, or LR schedule.

## Prerequisite: power-law advantage is unquestionably alive

Locked clean-regime prerequisite:

```text
Static - Uniform exact-accuracy AUC
median = +0.9300
positive seeds = 5/5
```

This is far above the preregistered prerequisite floor. The experiment therefore did **not** fail because the seed phenomenon disappeared in the clean branch/flat-LR regime.

That point matters for interpretation: the negative persistence result is scientifically meaningful because the underlying power-law learning advantage was extremely strong.

## Primary Slow-vs-Fast result

Per-seed exact-accuracy AUC contrasts:

| Seed | Slow - Fast AUC |
| ---: | ---: |
| 0 | -0.0325 |
| 1 | +0.7106 |
| 2 | +0.0095 |
| 3 | +0.0340 |
| 4 | -0.0395 |

Locked aggregate readout:

```text
median Slow - Fast = +0.0095
4/5 seeds have |gap| <= 0.06
```

This satisfies the preregistered no-meaningful-persistence condition.

The sign is also not stable: excluding the single large seed-1 excursion, the remaining four runs cluster tightly around zero with both positive and negative signs. The frozen protocol was intentionally designed so a large effect in one arbitrary rank-to-skill assignment could not be promoted into a general mechanism claim.

## Why seed 1 does not rescue the topic

Seed 1 produced a large `+0.7106` Slow advantage. It is real enough to preserve as an anomaly, but it does not establish a general temporal-persistence phenomenon:

1. four of five independently locked mapping/seed replications are near zero;
2. the median, which was frozen specifically to resist one-run domination, is `+0.0095`;
3. the direction is not consistent across seeds;
4. the project-level claim was that persistent head identity is a general causal ingredient of the power-law advantage, not that some specially favorable skill assignment can exhibit a persistence effect.

Trying additional mapping pairs, head subsets, group-structure descriptors, persistence schedules, or selecting seed-1-like assignments now would be post-hoc search.

If independent future evidence suggests that **which skills occupy the head** interacts systematically with temporal persistence — for example through subgroup/generator structure — that should be registered as a new scientific question with its own pre-data identification. It is not a valid rescue of Topic 14.

## Interpretation

The strongest supported statement in the locked testbed is:

> **Power-law asymmetry has a very large learning benefit, but that benefit does not generally require the same skills to remain high-frequency for a long period. Local / instantaneous asymmetry is sufficient for the main advantage in this regime.**

This falsifies the simple project-level hypothesis that temporal persistence of a privileged head is a necessary or broadly important ingredient of the power-law curriculum mechanism.

It does **not** claim that temporal order can never matter in any compositional-learning system. It says that, under the exact S5 regime where the seed phenomenon is very strong, the same-data Slow-vs-Fast intervention does not reveal a stable persistence law.

## Why G1 was not run

G1 was preregistered only after a replicated G0 temporal-order effect, to estimate a persistence timescale.

G0 did not establish such an effect. Running intermediate persistence lengths after observing the null — especially to explain seed 1 — would convert G1 from mechanism characterization into parameter search. It was therefore correctly skipped.

## Frozen conclusion

```text
PREREQUISITE: PASS, very strong
TEMPORAL PERSISTENCE: no stable effect
G1: not authorized
TOPIC 14: archive
```
