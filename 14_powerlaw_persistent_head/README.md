# 14 — Does Power-Law Learning Need a Persistent Head?

**Status:** REGISTERED CANDIDATE — CHEAP MECHANISM G-0

## Natural question

> When unequal practice frequencies help a learner acquire compositional skills, is it enough that some skills are locally more frequent at every moment, or must the **same** skills remain frequent long enough to become a stable scaffold for the rest?

This asks whether the benefit of a power-law training distribution comes from instantaneous distributional asymmetry or from a **persistent implicit curriculum**.

## Seed phenomenon

*The Power of Power Law* (ICML 2026) reports a striking phenomenon on compositional learning tasks: replacing a uniform skill-frequency distribution with a power-law distribution can turn tasks that are otherwise difficult or effectively unlearnable into learnable ones.

The work also reports a staged learning pattern in which high-frequency head skills are acquired first and learning subsequently accelerates on the tail. It tests several alternative skill orderings and supports the broader role of distributional asymmetry rather than a special semantic ordering of skills.

This leaves a temporal identification gap:

> Does the asymmetry need to stay attached to the same skills over time, or is a power-law-shaped minibatch/block distribution useful even when the identity of the head keeps changing?

That distinction separates two explanations:

1. **instantaneous symmetry breaking** — any locally unequal gradient signal helps optimization escape the uniform regime;
2. **persistent curriculum** — particular head skills must remain advantaged long enough to be mastered and then scaffold learning of the tail.

## Avoiding the obvious confound

A naive comparison between static power law and randomly reshuffled power law changes the **global per-skill training counts**: a static head receives more total examples, whereas a perfectly rotating head can become uniform in the long run.

The primary experiment should therefore compare schedules that preserve both:

- the same power-law frequency spectrum inside every training block;
- the same total rank occupancy / total exposure count for every skill over the whole run.

Only the **temporal persistence** of rank assignments changes.

## G-0: same counts, same local spectrum, different persistence

Partition training into equal-sized blocks. Each block uses the same frozen power-law rank-frequency vector.

Construct two balanced rank-assignment schedules:

### Slow rotation / high persistence

A skill that becomes the head remains near the head for a long contiguous run of blocks before rank assignments rotate.

### Fast rotation / low persistence

Use the exact same number of blocks at each rank for every skill, but temporally shuffle those rank assignments so head identity changes rapidly.

Across the complete run, the two schedules therefore have:

- identical total tokens;
- identical examples/skill counts;
- identical per-block power-law frequency spectra;
- identical number of head/mid/tail blocks per skill;
- different autocorrelation / persistence time of which skill occupies the head.

Include two anchors:

- **Uniform**: seed negative-control regime;
- **Static power law**: seed positive-control regime with a fixed head/tail assignment.

The causal comparison is `slow rotation vs fast rotation`; static and uniform only anchor interpretation.

## Primary first figure

Plot compositional-task learning curves for:

`Uniform / Static-PL / Slow-Rotation-PL / Fast-Rotation-PL`.

The first question is simply whether slow and fast balanced schedules separate.

No representation analysis, probe, task-specific mechanism classifier, or threshold search is needed.

## Interpretation

### Slow rotation > fast rotation

Persistent frequency advantage matters even when long-run per-skill counts are identical:

> **a temporary head must remain head long enough to become a scaffold.**

If slow rotation approaches static power-law performance, this is especially strong evidence for an implicit-curriculum timescale rather than mere global frequency imbalance.

### Slow rotation ≈ fast rotation > uniform

Rapidly changing local asymmetry is sufficient:

> **instantaneous symmetry breaking, not stable head identity, is the key ingredient.**

This would be surprising because the global marginal distribution of the balanced rotating schedules can be uniform even while every local block is power-law shaped.

### Static power law > both balanced rotating schedules ≈ uniform

Then the benefit requires more than local asymmetry or temporary persistence. A genuine long-run frequency hierarchy / permanently advantaged skill set is important.

### Ordered continuum with persistence timescale

If performance varies monotonically with head-persistence duration, the natural next object is a learning-timescale relation: how long must a skill remain advantaged before the power-law curriculum effect appears?

That follow-up should only be attempted after a large slow-vs-fast separation.

## Kill line

First reproduce the seed's static-power-law advantage over uniform in the chosen small setup. If that prerequisite fails, stop.

Then run one frozen slow-vs-fast balanced comparison. Kill the temporal-persistence hypothesis if those schedules are nearly indistinguishable and neither yields a scientifically interesting result relative to the anchors.

Do **not** rescue the topic by sweeping many power-law exponents, arbitrary rotation frequencies, skill taxonomies, or hidden-state measurements after a weak first contrast.

## What would make the result worth being excited about?

The interesting result is a principle about **how a curriculum exists in time**.

Either of the following would be strong:

> **A power-law-shaped local training stream can enable compositional learning even when every skill has the same long-run frequency.**

or

> **The same local frequency asymmetry only works when its head identity persists long enough, revealing a temporal scaffold behind the power-law effect.**

Both outcomes go beyond saying that "power-law data works better" and directly identify what part of that distributional structure actually matters.
