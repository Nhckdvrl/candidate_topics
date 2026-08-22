# Validation contract — Topic 10 (G0 v2)

## Scientific question

Does a masked diffusion LM preserve its Sudoku solving behavior when the same constraint problem is rewritten by an exact spatial Sudoku isomorphism?

There are two observables, in this order:

1. **outcome equivariance** — does an exactly solved puzzle remain exactly solved after an isomorphism?
2. **order equivariance** — conditional on both versions being solved, is the mapped blank-cell finalization order preserved?

This is deliberately stronger and cleaner than treating failed transformed runs as unusable data. A `solve -> fail` or `fail -> solve` flip under a mathematically exact isomorphism is itself evidence that serialization matters.

## What is held fixed

For every blank cell `c` in puzzle `x`, a row/band, column/stack, or transpose automorphism `T` gives a known mapped cell `T(c)` in `T(x)`. The complete CSP and unique solution are preserved. Primary transforms keep digit labels unchanged, so a mapped logical cell keeps the same answer-token identity while moving through row-major serialization.

The model does not free-generate text. A readable 9x9 answer-grid template is appended after the prompt. Separators and given digits are fixed. Exactly the original blank cells are mask tokens, and exactly one blank is finalized per step. Therefore cell ↔ token position ↔ finalization rank is exact rather than inferred post hoc.

## Decoder fidelity

The v1 code incorrectly renormalized logits over digits `1..9`, which could create artificial confidence. G0 v2 fixes this.

For each masked cell we choose the most likely *valid Sudoku digit*, but its scheduling score is its probability under the **full vocabulary**:

`p(best_valid_digit | current masked sequence)`.

Thus a digit with 1% full-vocabulary probability remains 1% confidence. If the native full-vocabulary argmax is itself a digit, the score equals standard LLaDA confidence for that position. We log `native_digit_argmax_fraction` so any remaining grammar-projection gap is visible rather than hidden.

The public LLaDA generator likewise ranks masked positions by the full-vocabulary probability of the proposed token under confidence remasking; our only task-specific restriction is that a Sudoku cell must ultimately contain a digit.

## Frozen G0 sequence

### G0-P — preflight

Before loading the 8B model:

- generate a deterministic unique Sudoku and verify exactly one solution;
- verify 20 sampled spatial automorphisms preserve the solution and mapped blank set.

With the tokenizer:

- verify digits `1..9` are nine distinct exact single tokens;
- verify the configured mask token;
- build the fixed template and assert that mask tokens occur **only** at intended blank-cell positions.

Any failure is an engineering/measurement failure and must be repaired before looking at scientific results.

### G0-S — same-serialization stability

On the first 12 discovery puzzles, decode the exact same serialization twice under the locked temperature-0 protocol.

Primary stability diagnostic:

`Kendall(finalization_order_run1, finalization_order_run2)`.

Mean tau must be at least `0.95`. If the same serialized problem does not have a stable measurable order, low isomorphism tau cannot be interpreted as a structural effect.

This replaces the old `candidate_count -> finalization` kill gate. Static candidate count is only a crude local Sudoku difficulty proxy and is now reported as secondary characterization.

### G0-O — outcome equivariance

For every identity/isomorph pair, record exact-solve status before any filtering.

Report:

- identity exact accuracy;
- isomorph exact accuracy;
- isomorph exact retention conditional on identity success;
- total solve-flip rate;
- flip directions (`identity correct -> iso wrong` and the reverse).

Do **not** discard solve flips. A substantial transformation-dependent success change is already a direct failure of policy equivariance.

At least 16/64 identity puzzles must be exactly solved before the controlled protocol is considered usable for scientific interpretation. If the base protocol almost never solves Sudoku, replace/fix the model or task formulation before drawing conclusions.

### G0-R — order equivariance

Only pairs where both identity and isomorph exactly solve the unique Sudoku enter the order analysis. For each such pair:

`tau_iso = Kendall(r_x(c), r_T(x)(T(c)))`.

Because four transforms share one source puzzle, uncertainty is clustered by **puzzle**: average transform-level tau within each puzzle, then bootstrap across puzzle means. Do not treat the four transforms as independent samples.

The locked minimum for a quantitative order conclusion is 30 both-correct pairs spanning at least 12 source puzzles. Failure to reach this threshold because isomorphs frequently turn successes into failures is not a null result; it falls back to G0-O outcome non-equivariance.

### G0-N — positional nulls

For every exact pair, compute two parameter-free null trajectories using the *same transform geometry*:

- **surface-order null:** a scheduler that always follows absolute row-major cell index;
- **boundary-first null:** a scheduler driven only by distance to the two ends of the 81-cell row-major sequence.

Report observed `tau_iso` beside both null taus and the per-puzzle excess:

`tau_observed - tau_surface_null`

`tau_observed - tau_boundary_null`.

This makes an intermediate tau interpretable. We no longer call `tau=0.4` "mixed" merely by inspection; we ask whether it preserves substantially more mapped order than simple positional schedulers would under the exact same automorphisms.

### G0-C — secondary controls and characterization

On 12 discovery puzzles, randomize which masked position is revealed while keeping the same model and Sudoku digit grammar. Its order should have no systematic correlation with confidence order. This is a sanity check, not the central identification argument.

Also report, within each correctly solved identity puzzle, Spearman correlation between initial candidate count and finalization step. Summarize these **per-puzzle** correlations; never pool cells across puzzles. This indicates whether the controlled protocol resembles the seed paper's easy-first story, but it is not allowed to kill an otherwise well-identified invariance result.

## Frozen data discipline

`LOCKED_CONFIG.json` creates 64 discovery and 64 reserved confirmation puzzles, each with four digit-preserving spatial automorphisms. The manifest is generated once from the locked seed.

The runner is idempotent by construction:

- existing output requires explicit `--resume` or `--overwrite`;
- `--resume` skips completed trace keys;
- duplicate trace keys cause analysis to fail loudly rather than inflate sample size.

Do not run confirmation until the discovery interpretation has been written down. Do not search prompts, thresholds, transform subsets, or alternative order metrics after seeing discovery.

## What establishes the topic

The project is healthy if the controlled decoder solves enough identity puzzles and same-serialization order is stable. From there, either scientific direction is useful:

- **high outcome retention + tau far above positional nulls:** strong evidence that generation order tracks a structural property of the CSP;
- **solve flips and/or tau near positional nulls:** strong evidence that apparent adaptive ordering is substantially serialization/sampler dependent;
- **high outcome retention with intermediate tau significantly above positional nulls:** both structural and positional forces are present.

The goal is not to manufacture a negative result. The goal is to make the first experiment strong enough that a positive invariance result genuinely establishes a reusable property, while a negative result cannot be dismissed as a measurement artifact.
