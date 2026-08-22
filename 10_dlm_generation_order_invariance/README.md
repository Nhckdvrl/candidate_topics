# 10 — Is DLM Generation Order Invariant to Problem Isomorphisms?

**Status:** G0 v2 prerequisite failure; G0 v3 4x4 published-setting discovery + confirmation completed

G0-v2 is retained as an engineering/prerequisite audit, not a scientific rejection: LLaDA-8B-Instruct had 0/8 exact identity solves on our unvalidated 9x9 zero-shot setting. G0-v3 moves to the literature-established UPO 4x4 setting and keeps all v2 traces separate.

## Question

> When a problem is changed only by an exact symmetry that preserves its underlying structure, does a diffusion language model preserve how it solves the problem?

For Sudoku this has two levels:

1. **outcome equivariance** — does an exactly solved puzzle remain solved after an exact spatial isomorphism?
2. **order equivariance** — when both versions are solved, is mapped blank-cell finalization order preserved?

This separates problem-structural scheduling from serialization / sampler scheduling without hidden states, probes, learned alignment, or subjective labels.

## Why this question exists

Findings of ACL 2026, *Parallelism and Generation Order in Masked Diffusion Language Models*, reports adaptive generation order across eight MDLMs and 58 benchmarks; Sudoku is highlighted as a case where easier blanks tend to be filled first. ACL 2026, *Empirical Analysis of Decoding Biases in Masked Diffusion Models*, independently shows that uncertainty-based decoding can exhibit rigid boundary and trivial-token biases.

The missing contrast is therefore direct:

> If the Sudoku CSP is unchanged up to a mathematical isomorphism, does decoding follow the mapped problem structure or the new serialization?

## Exact intervention

Primary transforms are legal Sudoku automorphisms: row/band permutations, column/stack permutations, and transpose. Digit labels are not changed in G0, so every blank `c` maps to a known blank `T(c)` with the **same correct digit token** while moving through row-major serialization.

Base solved grids are sampled by randomized Sudoku backtracking rather than generated only from one canonical solution orbit. A deterministic clue-removal procedure then produces unique 45-blank puzzles.

## Measurement

The output is a fixed readable 9x9 grid template. Separators and givens are clamped; only original blanks are mask tokens. Exactly one blank is finalized per step, so:

`r_x(c) = irreversible reveal rank of cell c`

is exact.

The decoder selects the best legal digit for each blank but schedules positions using that digit's **full-vocabulary probability**:

`p(best valid digit | current masked sequence)`.

This fixes a material G0-v1 bug, where confidence had been renormalized over digits `1..9` and could become artificially large.

G0 v2 also computes, at every confidence-decoding step, which position the completely native full-vocabulary LLaDA scheduler would have selected. The result logs:

- `native_digit_argmax_fraction`;
- `native_scheduler_pick_same_fraction`.

The latter is the key fidelity diagnostic: if grammar-constrained and native schedulers choose different positions too often, repair the prompt/template before interpreting the scientific result. The locked discovery fidelity floor is `0.80`.

## G0 decision path

### Measurement validity

`preflight.py` verifies unique Sudoku solutions, exact automorphism preservation, tokenizer digit identities, mask ID, and exact mask/cell-slot alignment.

### Same-serialization stability

The first 12 discovery puzzles are decoded twice with identical serialization. Mean order Kendall tau must be at least `0.95`. If the same representation has no stable measurable order, isomorphism non-invariance is not interpretable.

### Outcome equivariance

Every identity/isomorph pair contributes a solve/fail comparison before any filtering. `solve -> fail` under an exact isomorphism is itself evidence of non-equivariance and is never discarded merely because it cannot enter an order-correlation analysis.

Outcome flip rate and isomorph retention both receive source-puzzle-cluster bootstrap confidence intervals.

### Order equivariance

For pairs where both versions exactly solve the unique Sudoku:

`tau_iso = KendallTauB(r_x(c), r_T(x)(T(c)))`.

Four transforms share one source puzzle, so transform-level taus are averaged within source puzzle and bootstrap uncertainty is computed across source puzzles.

### Positional nulls

Observed tau is compared against two parameter-free schedulers under the exact same transforms:

- pure row-major order;
- pure boundary-first order.

The analysis reports observed-minus-null effects and puzzle-cluster CIs. This prevents an intermediate tau from being interpreted by eyeballing.

### Secondary characterization

Random remasking and within-puzzle initial-candidate-count correlations are reported as diagnostics. Candidate count is **not** a kill gate; it is only a crude easy-first proxy.

## Frozen budget and provenance

`LOCKED_CONFIG.json` fixes LLaDA-8B-Instruct, 45 blanks, 64 discovery puzzles, 64 untouched confirmation puzzles, four digit-preserving isomorphs per puzzle, temperature 0, 12 same-serialization repeats, and 12 random controls.

Both manifest and traces are stamped `g0-v2`. Old v1 manifests/results are rejected rather than silently resumed or mixed with v2. Runners support deterministic multi-GPU sharding and true pre-inference `--resume`; analysis accepts multiple shard JSONL files and rejects duplicate trace keys.

## What establishes the topic

The first run is designed to make either direction hard to dismiss:

- **high solve retention + mapped-order tau clearly above positional nulls** establishes genuine structural/equivariant scheduling;
- **solve flips and/or mapped-order tau close to positional nulls** establishes substantial serialization/sampler dependence;
- **high retention + intermediate tau still clearly above positional nulls** supports a real mixture of structural and positional forces.

The goal is not to kill the topic. It is to make a positive result strong enough to become a paper premise and a negative result too clean to explain away as measurement noise.

## Quick start

### G0-v3 published 4x4 route

The v3 route first reproduces the public UPO 4x4 test CSV and prompt/template. The 500-row baseline reached 72.675% blank-cell accuracy and 59.0% exact-puzzle accuracy. It then freezes a 64/64 discovery/confirmation manifest with four digit-preserving spatial transforms per puzzle.

See [`V3_PUBLISHED_REPRO.md`](./V3_PUBLISHED_REPRO.md), [`LOCKED_CONFIG_V3.json`](./LOCKED_CONFIG_V3.json), and the v3 result summaries under `results/`.

```bash
cd 10_dlm_generation_order_invariance
pip install -r requirements.txt
python src/preflight.py --config LOCKED_CONFIG.json
pytest -q tests
rm -f data/manifest.jsonl
python src/make_manifest.py --config LOCKED_CONFIG.json --out data/manifest.jsonl

# plumbing only
python src/run_g0.py --split discovery --limit 4 --overwrite --out results/smoke_traces.jsonl
python src/analyze_g0.py --config LOCKED_CONFIG.json --traces results/smoke_traces.jsonl --manifest data/manifest.jsonl --out results/smoke_summary.json

# frozen discovery
python src/run_g0.py --split discovery --overwrite --out results/g0_discovery_traces.jsonl
python src/analyze_g0.py --config LOCKED_CONFIG.json --traces results/g0_discovery_traces.jsonl --manifest data/manifest.jsonl --out results/g0_discovery_summary.json
```

See [`VALIDATION.md`](./VALIDATION.md) for the scientific contract and [`SERVER_HANDOFF.md`](./SERVER_HANDOFF.md) for execution details.
