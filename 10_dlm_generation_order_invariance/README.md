# 10 — Is DLM Generation Order Invariant to Problem Isomorphisms?

**Status:** G0 IMPLEMENTED / READY FOR FROZEN DISCOVERY

## Natural question

> When a problem is changed only by an exact symmetry that preserves its underlying structure, does a diffusion language model preserve the order in which it solves the problem?

This targets a core interpretation of arbitrary-order generation: whether observed generation order reflects **problem structure** or is substantially induced by **serialization / sampler bias**.

## Seed tension

### Seed A — generation order appears adaptive

*Parallelism and Generation Order in Masked Diffusion Language Models* (Findings of ACL 2026) studies eight MDLMs over 58 benchmarks and reports that generation order varies with task, reasoning stage, and correctness. Sudoku is the most direct motivating case: easier cells tend to be finalized earlier, motivating the interpretation that arbitrary-order decoding follows computational/problem structure.

### Seed B — confidence decoding has non-semantic ordering biases

Recent ACL 2026 work on uncertainty/confidence decoding reports rigid boundary and trivial-token biases. Therefore, an early token is not automatically evidence that the model judged it logically prior or easier.

The unresolved identification question is:

> **If the Sudoku constraint problem is changed only by an exact spatial isomorphism, is the mapped-cell finalization order preserved?**

## What is now implemented

G0 is no longer a free-form-output pilot. The code uses a deliberately strict measurement contract:

- deterministic unique Sudoku generation with an exact solver;
- exact row/band, column/stack and transpose automorphisms;
- **spatial-only primary transforms**: digit identity is held fixed;
- an 81-cell fixed answer region appended to the prompt;
- givens clamped as digit tokens; only blanks are `[MASK]`;
- digit grammar restricted to `1..9` to prevent tokenization ambiguity;
- exactly **one blank finalized per diffusion step**, giving an exact cell-level rank;
- LLaDA-style low-confidence reveal with per-cell finalization-step instrumentation;
- random-remasking negative control;
- frozen discovery and untouched confirmation splits;
- dependency-free Kendall tau-b / Spearman / bootstrap analysis;
- unit tests for solver, automorphisms, invariants, manifest determinism and analysis.

See [`VALIDATION.md`](./VALIDATION.md) for the scientific contract and [`SERVER_HANDOFF.md`](./SERVER_HANDOFF.md) for exact commands.

## Why the fixed-slot protocol is important

Parsing generation order from free-form text would require guessing which generated token corresponds to which Sudoku cell, and tokenization changes could themselves create apparent order effects.

Instead, the suffix contains exactly 81 cell slots. If a puzzle has blank set `B`, only positions in `B` are masks. For every blank `c`, the decoder records:

`r_x(c) = irreversible reveal step of cell c`.

A spatial Sudoku automorphism `T` gives a known one-to-one map:

`c -> T(c)`.

So the primary pairwise statistic is directly defined:

`tau_iso = KendallTauB({r_x(c)}, {r_T(x)(T(c))})`.

No learned alignment, hidden representation, judge or hand annotation is involved.

## G0 gates — interpreted in this order

### G0-P — preflight / identifiability

The pipeline must verify:

1. each generated puzzle has exactly one solution;
2. every sampled spatial transform preserves the mapped solution and blank set;
3. the tokenizer represents digits `1..9` as nine distinct one-token symbols;
4. the configured mask-token ID is valid.

Failure here is an implementation/measurement stop.

### G0-A — reproduce the seed phenomenon

Before talking about invariance, identity puzzles that are solved correctly must show the basic easy-first ordering that motivates the topic.

Frozen inexpensive structural diagnostic:

`Spearman(initial candidate count, finalization step)`.

Positive means cells with more legal candidates initially tend to be finalized later. The preregistered minimal signal is `rho >= 0.15`.

If this signal is absent, the current model/protocol has not instantiated the phenomenon to be explained. Do not continue into a mechanism story.

### G0-B — random-remasking negative control

Run the same fixed-slot decoder but choose the next blank uniformly at random rather than by model confidence.

The ordering relative to confidence-based identity decoding should have mean absolute Kendall tau near zero (`<= 0.20`). A strong residual ordering would reveal a pipeline artifact.

### G0-C — primary isomorphism test

For each exact-solved identity/isomorph pair, compute mapped-cell Kendall tau-b.

Primary result:

`distribution(tau_iso)`.

At least 50 exact solved identity/isomorph pairs are required before interpretation.

Possible outcomes all answer the question:

- **high positive tau** — solution order behaves approximately like a structural invariant;
- **near-zero tau** — the visually logical order is not preserved under equivalent serialization and is substantially decoder/serialization dependent;
- **intermediate stable tau** — structural scheduling and sampler scheduling coexist.

### G0-D — positional diagnostic, secondary only

After G0-C, test whether mapped rank changes track movement toward/away from the row-major sequence boundary:

`Spearman(delta boundary distance, delta finalization step)`.

This helps explain a low/mixed invariance result but is not needed to make the phenomenon exist.

## Frozen data budget

[`LOCKED_CONFIG.json`](./LOCKED_CONFIG.json) specifies:

- model: `GSAI-ML/LLaDA-8B-Instruct`;
- 45 blanks per puzzle;
- 64 discovery puzzles;
- 64 reserved confirmation puzzles;
- 4 spatial isomorphs per puzzle;
- deterministic temperature-0 confidence decoding;
- one reveal per step.

The discovery cost is therefore about `64 x (1 original + 4 isomorphs)` confidence trajectories, plus a cheap random-remasking control. There is no initial model/layer/prompt sweep.

## Repository layout

```text
10_dlm_generation_order_invariance/
├── LOCKED_CONFIG.json
├── LITERATURE_AUDIT.md
├── README.md
├── SERVER_HANDOFF.md
├── VALIDATION.md
├── requirements.txt
├── src/
│   ├── analyze_g0.py
│   ├── instrumented_llada.py
│   ├── make_manifest.py
│   ├── metrics.py
│   ├── preflight.py
│   ├── run_g0.py
│   ├── schema.py
│   └── sudoku.py
└── tests/
    ├── conftest.py
    ├── test_analysis.py
    ├── test_manifest.py
    ├── test_metrics.py
    └── test_sudoku.py
```

## Quick start

```bash
cd 10_dlm_generation_order_invariance
pip install -r requirements.txt
python src/preflight.py --config LOCKED_CONFIG.json
pytest -q tests
python src/make_manifest.py --config LOCKED_CONFIG.json --out data/manifest.jsonl

# plumbing only
python src/run_g0.py --split discovery --limit 8 --include-random-control --out results/smoke_traces.jsonl
python src/analyze_g0.py --traces results/smoke_traces.jsonl --manifest data/manifest.jsonl --out results/smoke_summary.json

# frozen discovery
python src/run_g0.py --split discovery --include-random-control --out results/g0_discovery_traces.jsonl
python src/analyze_g0.py --traces results/g0_discovery_traces.jsonl --manifest data/manifest.jsonl --out results/g0_discovery_summary.json
```

## Kill line

The project must not be rescued by broad prompt/model/sampler/metric search after discovery.

Stop the current protocol if:

- fixed cell-token mapping fails preflight;
- exact Sudoku solving is too weak to yield 50 matched successful isomorph pairs;
- the seed easy-first signal is absent under the frozen protocol; or
- random remasking exhibits a strong non-random order correlation, indicating an instrumentation artifact.

If those prerequisites pass, **do not kill because `tau_iso` has an inconvenient sign or magnitude**. High, low and mixed invariance are all scientifically interpretable outcomes of the same clean question.

## What would actually be exciting?

Not "a cell near the left edge goes first."

The scientifically meaningful result is one of two strong principles:

> **Generation order is approximately equivariant to exact changes of representation of the same constraint problem.**

or

> **A claimed logical solution order changes substantially when only the serialization of an isomorphic problem changes.**

Either would materially change how arbitrary-order DLM decoding should be interpreted.
