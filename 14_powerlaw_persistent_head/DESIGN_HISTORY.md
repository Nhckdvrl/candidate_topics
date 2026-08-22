# Topic 14 design history

This file records protocol changes made before interpreting Topic-14 scientific results.

## v1 — balanced cyclic rotation

The first registered implementation used all 120 cyclic rank shifts so each skill occupied every power-law rank equally often. Slow ordered those shifts smoothly; Fast shuffled them.

Audit found two identification weaknesses: the effective head persistence could be shorter than the seed paper's learning transition, and a cosine LR made data order inseparable from optimizer time.

## v2 — exact two-map matched-multiset intervention

Before scientific data were interpreted, the protocol was replaced by two frozen maps A/B and deterministic batch keys. Slow and Fast receive the exact same finite minibatch multiset; Slow keeps a map for 40k/80k steps while Fast alternates every step. A shared uniform branch checkpoint and constant post-branch LR make the core comparison unusually clean.

## v3 — second-pass fairness / implementation audit (2026-08-22)

A second code-and-design audit kept the v2 scientific intervention but fixed ways it could falsely kill or misread the topic:

1. `paper_anchor` is now genuinely near-paper: random initialization, no shared data warmup, own data from step 0, 1000-step LR warmup, cosine-to-0.1x, 200k steps, fp16 on CUDA.
2. A weak Static-vs-Uniform anchor in the deliberately clean shared-warmup/constant-LR regime no longer kills the persistence question. It yields `CORE_ANCHOR_WEAK_NO_PERSISTENCE_CONCLUSION`; the paper anchor then distinguishes reproduction failure from clean-regime incompatibility.
3. The A/B relation remains a fixed half-cycle shift, but the base rank→skill mapping is predeclared to vary across replication seeds. This tests that a result is not tied to one arbitrary skill assignment while preserving exact pairing within each seed.
4. Analyzer integrity checks now cover architecture, precision, optimizer/data seeds, alpha, mapping, evaluation panel, metric grid, protocol version and run signatures.
5. `--resume` now restores real model/AdamW/fp16-scaler checkpoints rather than merely skipping completed runs.
6. GPU launching respects external `CUDA_VISIBLE_DEVICES` masks and runs waves instead of oversubscribing cards when fewer GPUs than arms are available.

These changes were motivated by pre-result audit logic: give a true phenomenon a fair chance and separate scientific nulls from engineering/regime failures. They do not add post-hoc scientific controls or change the primary Slow-vs-Fast question.

If any older Topic-14 outputs exist from v1/v2, keep them separate. v3 outputs carry protocol signatures and should use a fresh output root if stale files trigger a mismatch.
