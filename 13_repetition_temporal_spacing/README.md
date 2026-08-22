# 13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?

**Status:** REGISTERED CANDIDATE — HIGH-PRIORITY G-0

## Natural question

> If a learner sees the exact same examples the same number of times, does it matter how far apart those repeated exposures occur?

For language-model pretraining:

> Is duplicate-data damage determined only by the training multiset, or also by the temporal spacing between repeated copies of the same document?

This is a learning-dynamics question, not a data-cleaning question. The key object is whether two training streams containing **exactly the same documents with exactly the same multiplicities** can produce meaningfully different models solely because duplicate occurrences are temporally arranged differently.

## Seed phenomenon

*Internal Data Repetition Destroys Language Models* (2026) reports a strong and counterintuitive repetition effect in pretraining. When a fixed fraction of training compute is spent on repeated data, the damage is not simply monotonic in repeat count; intermediate repetition regimes can be especially harmful and can waste a substantial fraction of effective training compute.

The seed establishes that exact-document repetition can be a first-order training variable.

However, the repeated copies are interleaved through the training stream rather than used to identify the role of **inter-exposure spacing** itself. Therefore the seed does not answer:

> Would the same repeated multiset be equally harmful if identical copies were deliberately clustered together or deliberately spaced apart?

Adjacent work on spaced repetition in continual pretraining provides independent motivation that temporal review schedules can matter, but it studies replay/retention policies rather than fixed-multiset duplicate damage.

## Exact question

Let every training run contain the same document IDs with the same counts:

`M = {d_1 x n_1, d_2 x n_2, ..., d_k x n_k}`.

Construct training streams that differ only in the temporal positions assigned to repeated identities.

The cleanest design fixes two things across all conditions:

1. the positions and order of all unique-document examples;
2. the global set of positions reserved for repeated-document exposures.

Only the mapping

`repeated slot -> repeated document identity`

changes.

This avoids turning the experiment into a generic curriculum/order comparison.

## G-0: same multiset, different spacing

Choose one seed-paper regime in which repetition damage is already large and reproducible. Freeze the model, optimizer, total tokens, repeated fraction, repeated document set, multiplicities, and repeated-slot locations.

Compare three schedules:

### Clustered

For each repeated document, assign its copies to nearby repeated slots, minimizing inter-exposure distance.

### Random

Randomly assign repeated identities to the fixed repeated slots, approximating ordinary shuffled interleaving.

### Evenly spaced

Assign each repeated document's copies across the repeated slots to make its inter-exposure distances as even and large as possible.

Every run therefore has:

- identical unique examples;
- identical repeated examples;
- identical example counts;
- identical total token/compute budget;
- identical positions at which the learner sees *some* repeated example;
- different temporal spacing of repeated copies of the **same** document.

## Primary first figure

Plot held-out validation loss against training compute for the three frozen schedules.

The primary endpoint is the final or compute-matched held-out loss difference. A secondary descriptive quantity may reproduce the seed paper's compute-equivalent loss metric.

No representation analysis, memorization probe, or alternate repetition regime is needed for G-0.

## Interpretation

### Spacing strongly changes damage

If evenly spaced repetition is substantially less harmful than clustered/random repetition despite an identical training multiset, then duplicate-data damage is not merely a static reweighting property:

> **the temporal organization of repeated exposure is itself a causal pretraining variable.**

This would imply that two datasets identical as multisets can have materially different effective compute simply because of order.

### Spacing does not matter

If clustered, random, and evenly spaced schedules have essentially the same damage, the natural conclusion is equally useful:

> repetition damage is dominated by what probability mass is duplicated, not by when identical copies reappear.

This directly answers the same question; it does not erase it.

### Non-monotonic spacing effect

A middle spacing being worst or best would be especially interesting because it would connect the seed paper's non-monotonic repeat-count phenomenon to a temporal learning-timescale effect.

## Kill line

The first fixed-multiset schedule comparison must carry the topic.

Kill the topic if the seed repetition damage cannot first be reproduced in the chosen small-scale setup. Also kill it if spacing changes only produce tiny/noisy differences that are not visible in held-out loss.

Do **not** rescue it by sweeping many models, repeat ratios, document categories, memorization thresholds, or data-order heuristics after the frozen comparison fails.

## What would make the result worth being excited about?

The exciting result is not merely "spacing helps a bit." It is a large clean separation under an identical training multiset:

> **same data, same multiplicities, same compute — different temporal spacing, materially different pretraining efficiency.**

That would turn duplication from a purely dataset-statistics issue into a learning-dynamics principle.
