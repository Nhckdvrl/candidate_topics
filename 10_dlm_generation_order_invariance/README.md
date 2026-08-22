# 10 — Is DLM Generation Order Invariant to Problem Isomorphisms?

**Status:** ARCHIVED — positive 4×4 G0, failed to establish a meaningful non-toy 9×9 experimental object

## Final verdict

Topic 10 is archived. The core phenomenon is real in the published UPO 4×4 Sudoku setting, but we could not establish a competent 9×9 experimental object under the available literature-aligned routes without introducing model/data/configuration fishing.

This is not a hypothesis rejection. It is a significance / scalability stop: the project passed the “phenomenon exists” gate but did not pass the “phenomenon exists in a scientifically meaningful non-toy regime” gate.

See [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) for the full postmortem.

## Question

> When a problem is changed only by an exact symmetry that preserves its underlying structure, does a diffusion language model preserve how it solves the problem?

For Sudoku this has two levels:

1. **outcome equivariance** — does an exactly solved puzzle remain solved after an exact spatial isomorphism?
2. **order equivariance** — when both versions are solved, is mapped blank-cell finalization order preserved?

## What worked

### G0-v3: published UPO 4×4 setting

We first reproduced the public UPO 4×4 setting with `GSAI-ML/LLaDA-8B-Instruct`:

- blank-cell accuracy: `2907/4000 = 72.675%`
- exact-puzzle accuracy: `295/500 = 59.0%`

Then we froze 64 discovery + 64 untouched confirmation source puzzles with four digit-preserving spatial transforms each.

Discovery:

- identity exact: `44/64 = 68.75%`
- isomorph exact: `157/256 = 61.33%`
- solve/fail flip: `39.45%`, 95% CI `[31.64%, 47.27%]`
- both-exact pairs: `116`
- mapped tau: `0.111`
- tau − row-major null: `0.054`, CI `[-0.052, 0.160]`
- tau − boundary-first null: `0.130`, CI `[0.015, 0.247]`

Confirmation:

- identity exact: `34/64 = 53.13%`
- isomorph exact: `148/256 = 57.81%`
- solve/fail flip: `45.31%`, 95% CI `[37.89%, 52.73%]`
- both-exact pairs: `84`
- mapped tau: `0.118`
- tau − row-major null: `0.207`, CI `[0.060, 0.356]`
- tau − boundary-first null: `0.119`, CI `[-0.015, 0.265]`

The robust result is outcome non-equivariance: exact spatial isomorphisms substantially reshuffle solve/fail outcomes. Conditional mapped generation order retains only a weak positive structural component, and the positional-null contrasts do not both replicate cleanly.

## What failed

### G0-v2: 9×9 LLaDA-8B prerequisite failure

The original zero-shot 9×9 fixed-grid object was not competent:

- identity exact: `0/8`
- blank-cell accuracy: `38.33%`
- same-serialization tau: `1.0`
- native scheduler agreement: `0.958`

The measurement pipeline was healthy, but the model/task object was not usable for the intended science.

### G1-v4: Dream-7B 9×9 seed-aligned reconstruction failure

We then tried to recover a 7B-class 9×9 object using the public Dream model/trainer and the seed paper's described 50-train / 100-test setup. The seed paper does not release its 9×9 corpus/generator or fully resolve Base-vs-Instruct provenance, so this was explicitly a seed-aligned reconstruction, not an exact reproduction.

Ordinary held-out exact-solve competence:

- epoch 2: `6/100`
- epoch 5: `3/100`

Training losses became very small while held-out exact solve stayed extremely low and declined. Raw outputs frequently repeated prompt instructions or emitted malformed, flat, or truncated matrices. The run was stopped before epoch 10 and before any 9×9 symmetry test.

## Why the project stops here

The remaining rescue routes require changing one or more of:

- Dream Base vs Instruct variant;
- unreleased 9×9 dataset distribution / generator;
- training configuration;
- prompt / response format;
- decoding details;
- or moving to a much larger model such as LLaDA2.0-flash-100B.

Those are no longer clean confirmations. They create too many degrees of freedom just to recover the prerequisite object.

The 4×4 result is real, but by itself it is too toy-scale to meet the significance bar for continuing this project as a primary paper direction.

## Transferable lesson

A research candidate needs two separate gates:

1. **phenomenon existence** — can the effect be cleanly demonstrated at all?
2. **meaningful-regime existence** — can the same question be studied in a regime large/natural enough to support the intended scientific claim?

Topic 10 passed the first gate and failed the second under the currently recoverable experimental objects.

## Historical artifacts

- `V3_PUBLISHED_REPRO.md` — published 4×4 reproduction
- `LOCKED_CONFIG_V3.json` — frozen 4×4 protocol
- `results/g0_v3_4x4_discovery_summary.json`
- `results/g0_v3_4x4_confirmation_summary.json`
- `G1_V4_DREAM_REPRO.md` — 9×9 Dream reconstruction design
- `G1_V4_TRAINING_RUN.md` / `G1_V4_ENV_AUDIT.md`
- `G1_V4_STOP_NOTE.md`
