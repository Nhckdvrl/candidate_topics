# 10 — Is DLM Generation Order Invariant to Problem Isomorphisms?

**Status:** G0 v2 IMPLEMENTED / READY FOR FROZEN DISCOVERY

## Question

> When a problem is changed only by an exact symmetry that preserves its underlying structure, does a diffusion language model preserve how it solves the problem?

For Sudoku this becomes two concrete questions:

1. does an exactly solved puzzle remain solved after an exact spatial isomorphism?
2. conditional on both being solved, is the mapped blank-cell finalization order preserved?

This distinguishes **problem-structural scheduling** from **serialization / sampler scheduling** without hidden states, probes, learned alignment, or subjective labels.

## Why this question exists

Findings of ACL 2026, *Parallelism and Generation Order in Masked Diffusion Language Models*, reports adaptive generation order across eight MDLMs and 58 benchmarks; Sudoku is highlighted as a case where easier blanks tend to be filled first. ACL 2026, *Empirical Analysis of Decoding Biases in Masked Diffusion Models*, independently shows that uncertainty-based decoding can exhibit rigid boundary and trivial-token biases.

Those observations leave a direct identification gap:

> If the underlying Sudoku CSP is literally unchanged up to a mathematical isomorphism, does the decoder follow the mapped logical structure or the new serialization?

## Exact intervention

Primary transforms use legal Sudoku automorphisms:

- row permutations within bands and band permutations;
- column permutations within stacks and stack permutations;
- transpose.

Digit labels are **not** changed in the primary experiment. Therefore every blank `c` maps to a known blank `T(c)` with the same correct digit token, while its row-major position changes.

## Measurement design

The output is a fixed readable 9x9 grid template. Separators and given digits are clamped; only original blank cells are `[MASK]`. Exactly one blank is finalized per step.

For every blank:

`r_x(c) = irreversible reveal rank of c`.

The decoder chooses the best legal digit for each masked cell, but the scheduling score is the probability of that digit under the **full vocabulary**:

`p(best valid digit | current masked sequence)`.

This is important. G0 v1 incorrectly renormalized confidence over digits `1..9`, which could manufacture high confidence. G0 v2 preserves the model's absolute full-vocabulary confidence and logs how often the native full-vocabulary argmax is itself a digit.

## G0 in one page

### 1. Measurement validity

`preflight.py` checks unique Sudoku solutions, exact automorphism preservation, tokenizer digit identities, mask ID, and that mask tokens occur only at intended cell slots.

### 2. Same-serialization stability

The first 12 discovery puzzles are decoded twice with identical serialization. Mean order Kendall tau must be at least `0.95` before low isomorphism tau can be interpreted structurally.

This is the real prerequisite. The old pooled `candidate count -> finalization` correlation is no longer a kill gate; it is only a secondary easy-first characterization.

### 3. Outcome equivariance

Every identity/isomorph pair contributes a solve/fail comparison **before filtering**.

Report:

- identity exact accuracy;
- isomorph exact accuracy;
- isomorph retention given identity success;
- solve-flip rate and direction.

A `solve -> fail` change caused only by an exact Sudoku isomorphism is already direct evidence of non-equivariance. It must not disappear because the pair cannot enter an order-correlation analysis.

### 4. Order equivariance

For pairs where both versions exactly solve the unique Sudoku:

`tau_iso = KendallTauB(r_x(c), r_T(x)(T(c)))`.

Four transforms share one source puzzle, so inference is clustered by source puzzle: transform taus are averaged within puzzle, then the bootstrap resamples puzzles rather than pretending all transforms are independent.

### 5. Positional nulls

Observed tau is compared against two parameter-free null schedules under the exact same transforms:

- pure row-major order;
- pure boundary-first order.

The analysis reports both null taus and:

`tau_observed - tau_positional_null`.

This is what makes an intermediate tau interpretable rather than a storytelling exercise.

### 6. Secondary diagnostics

- random-remasking order vs confidence order;
- within-puzzle candidate-count vs finalization correlation;
- native-digit-argmax fraction, measuring how close the task-constrained decoder is to native full-vocabulary confidence decoding.

None of these is allowed to replace the primary outcome/order equivariance result.

## Frozen budget

`LOCKED_CONFIG.json` fixes:

- `GSAI-ML/LLaDA-8B-Instruct`;
- 45 blanks per puzzle;
- 64 discovery puzzles;
- 64 untouched confirmation puzzles;
- 4 digit-preserving spatial isomorphs per puzzle;
- temperature 0;
- one blank finalized per step;
- 12 same-serialization repeats and 12 random controls on discovery.

The first scientific run therefore remains cheap: roughly 320 primary trajectories plus 24 controls.

## Interpretation

A usable experiment first requires enough identity successes and stable same-serialization order.

Then:

- **high solve retention + tau well above positional nulls** supports a structural/equivariant generation policy;
- **solve flips and/or tau near positional nulls** supports strong serialization/sampler dependence;
- **high solve retention + intermediate tau still clearly above positional nulls** supports a real mixture of structural and positional scheduling.

The goal is not to make the topic die. The goal is to make either direction hard to dismiss.

## Quick start

```bash
cd 10_dlm_generation_order_invariance
pip install -r requirements.txt
python src/preflight.py --config LOCKED_CONFIG.json
pytest -q tests
python src/make_manifest.py --config LOCKED_CONFIG.json --out data/manifest.jsonl

# plumbing only
python src/run_g0.py --split discovery --limit 4 --overwrite --out results/smoke_traces.jsonl
python src/analyze_g0.py --config LOCKED_CONFIG.json --traces results/smoke_traces.jsonl --manifest data/manifest.jsonl --out results/smoke_summary.json

# frozen discovery
python src/run_g0.py --split discovery --overwrite --out results/g0_discovery_traces.jsonl
python src/analyze_g0.py --config LOCKED_CONFIG.json --traces results/g0_discovery_traces.jsonl --manifest data/manifest.jsonl --out results/g0_discovery_summary.json
```

If the run is interrupted, use `--resume` rather than appending or restarting manually. Duplicate trace keys cause analysis to fail loudly.

See [`VALIDATION.md`](./VALIDATION.md) for the locked scientific contract and [`SERVER_HANDOFF.md`](./SERVER_HANDOFF.md) for the server execution sequence.
