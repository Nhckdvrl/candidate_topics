# 13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?

**Status:** VALIDATION CODE READY — re-audited clean G-0

## Scientific question

> If two language-model training runs see exactly the same documents the same number of times, can the model generalize differently only because repeated copies of the same document reappear at different times?

The seed phenomenon is *Internal Data Repetition Destroys Language Models* (2026): with 10% of the token budget assigned to exact-document replay, held-out loss is worst at an intermediate repeat count. For the 34M Qwen3-style model the damage peak is around `R ≈ 1400`, and the reported peak location is only weakly dependent on training duration across the measured OT grid.

The seed paper shuffles the final document-index list. It therefore establishes **repetition damage**, but does not identify whether temporal placement of identical copies matters once the training multiset is fixed.

Our question is narrower and causal:

> Is duplicate damage only a property of the multiset, or is **inter-exposure time across optimizer updates** itself a training variable?

## What the re-audit changed

The first implementation was clean at the document-multiset level but not clean enough at the optimizer level. In the old `clustered` schedule, the same repeated document could occur several times inside one effective optimizer batch, while `even` usually contained fewer same-document collisions. A positive result could therefore be explained as ordinary within-batch diversity loss rather than temporal memory/plasticity.

The revised G-0 removes that objection by construction:

> **Every optimizer step contains at most one repeat slot in every condition.**

The effective batch is frozen at:

```text
micro_batch = 8 blocks
grad_accum  = 1
seq_len     = 2048
```

Repeat slots are first assigned to distinct optimizer steps. Only after those temporal positions are frozen do we assign repeated identities.

Thus `clustered` versus `even` changes **when the same identity comes back across parameter updates**, not how many copies of that identity are averaged into one update.

## Frozen G-0 geometry

We retain the cheapest seed-covered regime:

```text
model:             34,061,856-parameter Qwen3-style LM
OT:                0.25
token budget:      20 * OT * N
context:           2048
repeat fraction:   10%
R:                 1386
```

At this budget the realized schedule is approximately:

```text
total training blocks:     83,158
repeat slots:               8,316
repeated documents:             6
copies per repeated doc:     1,386
optimizer-step batch:            8 blocks
optimizer steps:            10,395
max repeat slots / step:         1
```

`R=1386` is not a post-hoc search. It is near the seed paper's reported 34M peak around `R≈1400`, while making the 10% repeated budget divide exactly into six fixed-length repeated documents.

## The four conditions

### 1. `fresh`

The 10% reserved repeat slots are filled by one-time fresh documents. This is the matched no-repetition control.

### 2. `random`

The six repeated documents each appear exactly 1386 times and their identities are randomly assigned to the fixed repeat slots.

This is the prerequisite reproduction condition. Before interpreting spacing, we require `random` to have higher held-out loss than `fresh` on the same held-out documents.

### 3. `clustered`

Each repeated identity occupies a contiguous run in the ordered list of repeat slots. Because repeat slots themselves live in distinct optimizer steps, this creates short **cross-update** revisit intervals without within-step duplicate collisions.

### 4. `even`

A fixed random permutation of the six identities is repeated round-robin through the repeat slots. Each identity is therefore revisited regularly across the full training trajectory.

On the locked pilot schedule the realized mean spacing is roughly:

```text
clustered: ~1.25 optimizer steps
even:      ~7.50 optimizer steps
```

while both have exactly the same repeated documents and exact multiplicities.

## Causal invariants checked in code

Within each trial, `clustered`, `random`, and `even` have:

- identical repeated document IDs;
- identical count `R` for every repeated document;
- identical non-repeated document at every non-repeat slot;
- identical repeat-slot positions;
- identical total tokens and optimizer steps;
- at most one repeat slot per optimizer step;
- identical model initialization for the four conditions.

Only:

```text
repeat slot -> repeated document identity
```

changes.

`schedule.py` writes SHA-256 audits for these invariants and refuses to build a schedule if the one-repeat-slot-per-update constraint is impossible.

## A deliberate corpus simplification

For the causal G-0, every schedule atom is one long FineWeb-Edu-Dedup document represented as:

```text
first 2047 Qwen3 tokens + EOS
```

Short documents are excluded. This is deliberate: every temporal slot then contains exactly the same number of real training tokens, so moving a document does not also move variable amounts of compute/time.

This is **not identical to the seed paper's full document-length distribution**. Therefore:

> failure of `random > fresh` is a **setup/reproduction failure**, not evidence that the spacing hypothesis is false.

If that prerequisite fails without an engineering bug, the correct next action is to improve seed-paper reproduction fidelity, not to archive Topic 13 from that result.

Document IDs are stable (`id`, URL, or text hash), so the held-out split does not silently depend on streaming row order.

## Repeated-pool replication

The previous confirmation accidentally reused the same six repeated documents in every seed. That is not enough: a positive result could be driven by an unusual repeated pool.

The revised schedule uses each trial seed to choose a new repeated pool from the prepared corpus. The four conditions inside a trial still share that exact pool.

Confirmation uses four frozen trials:

```text
20260822
20260823
20260824
20260825
```

This jointly varies model initialization, training order, and repeated-pool sample between trials while preserving perfect matching within each trial.

## GPU blocking

With four GPUs, the confirmation rotates condition-to-GPU assignment cyclically across the four trials. Each condition therefore runs once on each physical GPU. The intended mapping is explicit before job reuse/resume and is recorded in each `metrics.json`, so restarting a partial trial cannot silently change the hardware block.

This prevents a small systematic GPU-specific numerical/performance difference from being permanently confounded with one condition.

There is no DDP/NCCL communication. Each GPU trains one independent 34M model.

## Evaluation and statistics

Every condition is evaluated once at the final checkpoint on the exact same 2048 held-out document blocks.

`train.py` saves the loss of every held-out block. Therefore within a trial we analyze paired differences:

```text
seed reproduction:  loss(random, doc_i)    - loss(fresh, doc_i)
spacing effect:     loss(clustered, doc_i) - loss(even, doc_i)
```

The default 95% interval uses the standard error of these paired document-level differences. This is much more efficient than treating the two evaluation sets as independent.

These intervals establish that an effect generalizes across held-out documents **within a trained pair**. They do not replace independent training replications; the four-trial confirmation handles training/pool variability.

No learned judge, hidden-state probe, memorization threshold, alternate model family, or post-result schedule sweep is part of G-0.

## Decision logic: evidence first, not kill first

### Pilot

One matched trial is used as an engineering/prerequisite check.

If `random > fresh` is clearly reproduced:

- if spacing is already visible: run confirmation;
- if spacing is null: **still run confirmation once**.

A single training seed is not allowed to kill the topic.

If seed repetition damage is not reproduced:

```text
PILOT_SETUP_FAIL_REPRODUCTION_NOT_TOPIC_FAIL
```

Do not interpret the spacing result. Check implementation/data fidelity. This is explicitly not a scientific negative for Topic 13.

### Confirmation

All four preregistered trials must complete.

Strong evidence requires:

1. repetition damage reproduces in at least 3/4 matched trials;
2. every valid trial has the same sign for `clustered - even`;
3. at least three valid trials individually resolve that same direction on the paired held-out set (or all valid trials if only three pass the prerequisite reproduction gate).

Then:

```text
GO_STRONG_SPACING_IS_CAUSAL
```

If all valid trials agree in direction but the per-trial intervals are not yet tight enough:

```text
PROMISING_SPACING_EFFECT_NEEDS_MORE_REPLICATION
```

That is a real lead, not a kill. A larger replication is justified before any method paper.

Only when repetition damage itself is stable but the clean spacing manipulation has inconsistent directions do we return:

```text
NO_EVIDENCE_SPACING_IN_LOCKED_TEST
```

That is the meaningful negative result.

### No arbitrary practical threshold

The old rule required spacing to exceed 25% of the random repetition damage. That threshold has been removed.

We still report:

```text
|clustered - even| / |random - fresh|
```

because magnitude matters scientifically, but it is descriptive rather than a hand-chosen gate.

## Stale-run protection

`run_g0.py` hashes:

- the frozen config;
- `schedule.py`;
- `train.py`;
- `prepare_corpus.py`;
- `analyze.py`;
- `run_g0.py`.

The hash becomes `experiment_id`. Existing results are reused only when the ID matches. A different code/config version in the same work directory fails fast instead of silently mixing experiments.

Because this re-audit changes the experiment schema, use a fresh work directory if an older Topic-13 run already exists.

## Files

```text
13_repetition_temporal_spacing/
├── README.md
├── configs/g0.json
├── preflight.py
├── prepare_corpus.py
├── schedule.py
├── train.py
├── analyze.py
├── run_g0.py
├── requirements.txt
└── tests/
    ├── test_schedule.py
    └── test_analysis.py
```

## Run

Use the existing PyTorch/Transformers environment if compatible; do not create another environment unnecessarily.

First:

```bash
cd 13_repetition_temporal_spacing
python -m unittest discover -s tests -v
```

Pilot on four GPUs:

```bash
python run_g0.py \
  --mode pilot \
  --num-gpus 4 \
  --work-dir runs/g0_v2
```

If the pilot reproduces the prerequisite seed effect, run the frozen confirmation regardless of whether the pilot spacing contrast is positive or null:

```bash
python run_g0.py \
  --mode confirm \
  --num-gpus 4 \
  --work-dir runs/g0_v2
```

The pilot seed is reused; confirmation adds the other three trials.

## What this experiment proves if it works

The strongest result is not “a different data loader is slightly better.” It is:

> Two pretraining runs consumed exactly the same repeated multiset with the same multiplicities and the same compute, with no within-update duplicate collision confound, yet held-out generalization changed systematically because identical documents were revisited at different distances across optimizer updates.

That would establish temporal organization of repeated exposure as a genuine pretraining variable and justify a broader study of the learning-timescale mechanism.
