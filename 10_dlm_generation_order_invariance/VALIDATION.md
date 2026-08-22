# Validation contract — Topic 10

## Scientific question

Does a masked diffusion LM preserve its Sudoku solution order under exact spatial isomorphisms of the same constraint problem?

The primary claim is **equivariance of the decoding policy under exact problem isomorphisms**, not a perfect causal `do(position)` claim.

## Why this G0 is identifiable

For every blank cell `c` in puzzle `x`, a spatial Sudoku automorphism `T` gives a known mapped cell `T(c)` in the logically identical puzzle `T(x)`. Row/band, column/stack and transpose transformations preserve the complete CSP. Primary transforms keep digit labels unchanged, so digit-token identity is held fixed while cells move through row-major serialization.

The output is not parsed from free-form text. It is an 81-slot answer grid appended to the prompt. Given cells are clamped to their known digit token; blank cells are literal mask tokens. Only blanks participate in decoding. Exactly one blank is finalized per diffusion step, so every blank has an unambiguous finalization rank.

## Frozen G0 sequence

### G0-P: preflight

1. Generate deterministic unique Sudoku puzzles.
2. Verify each has exactly one solution.
3. Apply 20 random spatial isomorphisms and verify mapped solutions and blank sets exactly.
4. Verify tokenizer encodes digits `1..9` as nine distinct single tokens.
5. Verify configured mask ID.

Any failure stops the run.

### G0-A: seed-phenomenon replication

Before testing invariance, verify the chosen model/protocol actually exhibits the phenomenon being interpreted. On correctly solved identity puzzles, compute the association between initial Sudoku candidate count and finalization step.

Locked diagnostic:

`Spearman(initial_candidate_count, finalization_step)`.

Positive values mean less constrained cells are finalized later. `rho >= 0.15` is the preregistered minimal G0 signal. If this is absent, do not interpret later invariance as evidence about the seed paper's "easy-first" phenomenon.

### G0-B: instrument negative control

On a small subset, replace confidence-based token selection with random-remasking while leaving the model, puzzles and fixed-slot grammar unchanged. Compare random-control order with identity confidence order. Mean absolute Kendall tau should be <= 0.20. A strong non-zero result indicates a pipeline/order artifact and stops interpretation.

### G0-C: exact isomorphism test

For each correctly solved identity puzzle and each correctly solved spatial isomorph, map blank cells back through the known transform and compute pairwise Kendall tau-b between their finalization steps.

Primary result:

`tau_iso` distribution across exact puzzle/isomorph pairs.

No layer, prompt, threshold, transform family or subset search is allowed after seeing the discovery result.

### G0-D: positional diagnostic

Secondary only: correlate cell movement toward/away from row-major sequence boundaries with mapped finalization-rank change.

`Spearman(delta_boundary_distance, delta_finalization_step)`.

This is explanatory follow-up, not required for the primary phenomenon.

## Data split and stopping discipline

`LOCKED_CONFIG.json` reserves 64 deterministic discovery puzzles and 64 independent confirmation puzzles, four spatial isomorphs each. Run discovery first. Confirmation stays untouched until the discovery decision is written down.

A puzzle/isomorph pair enters the primary tau analysis only when both outputs exactly solve their respective puzzles. This prevents comparing generation orders of failed trajectories that may not correspond to the same solved CSP.

At least 50 exact identity/isomorph pairs are required before interpreting G0-C. If the model cannot produce that many under the frozen protocol, the experimental object is not available cheaply enough and the topic should be stopped or the model replaced **before** inspecting alternative measurements.

## Interpretation, not significance fishing

- high positive `tau_iso`: solution order behaves approximately as a structural invariant;
- near-zero `tau_iso`, especially with systematic positional rank shifts: the apparent logical order is substantially serialization/sampler dependent;
- intermediate stable `tau_iso`: structural scheduling and sampler scheduling both matter.

All three answer the same question. The project is killed only by failed identifiability/prerequisite gates, not because tau points in an inconvenient direction.

## Efficiency choices

- one 8B DLM for G0 rather than an early model sweep;
- exactly one token finalized per step to eliminate tied-rank ambiguity;
- no external Sudoku dataset: deterministic unique-puzzle generator plus reserved seed;
- originals are decoded once and reused across their four transforms;
- no hidden states, probes, judges or training;
- confirmation set is generated upfront but not run until the discovery decision.
