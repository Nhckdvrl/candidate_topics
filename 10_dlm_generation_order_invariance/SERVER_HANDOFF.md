# Server handoff — Topic 10 G0 v2

Run from `10_dlm_generation_order_invariance/`.

## 1. Environment and preflight

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/preflight.py --config LOCKED_CONFIG.json
pytest -q tests
```

Use a recent CUDA PyTorch build and enough VRAM for `GSAI-ML/LLaDA-8B-Instruct` in BF16. Do not proceed past a failed preflight: a tokenizer/template/isomorphism failure means the measurement itself is invalid.

## 2. Freeze the manifest once

```bash
python src/make_manifest.py --config LOCKED_CONFIG.json --out data/manifest.jsonl
```

This materializes both discovery and reserved confirmation puzzles. Do not regenerate with another seed after looking at results.

## 3. Plumbing smoke

```bash
python src/run_g0.py \
  --config LOCKED_CONFIG.json \
  --manifest data/manifest.jsonl \
  --split discovery \
  --limit 4 \
  --overwrite \
  --out results/smoke_traces.jsonl

python src/analyze_g0.py \
  --config LOCKED_CONFIG.json \
  --traces results/smoke_traces.jsonl \
  --manifest data/manifest.jsonl \
  --split discovery \
  --out results/smoke_summary.json
```

Smoke is only for engineering. Because it is tiny, ignore scientific decision flags.

## 4. Frozen discovery

```bash
python src/run_g0.py \
  --config LOCKED_CONFIG.json \
  --manifest data/manifest.jsonl \
  --split discovery \
  --overwrite \
  --out results/g0_discovery_traces.jsonl

python src/analyze_g0.py \
  --config LOCKED_CONFIG.json \
  --traces results/g0_discovery_traces.jsonl \
  --manifest data/manifest.jsonl \
  --split discovery \
  --out results/g0_discovery_summary.json
```

If interrupted, rerun the first command with `--resume` instead of `--overwrite`. Never concatenate partial files manually; duplicate trace keys are deliberately rejected by analysis.

## 5. Read the result in the correct order

1. `n_identity_exact` / `identity_exact_accuracy`: is the controlled Sudoku protocol usable at all?
2. `same_serialization_repeat_tau`: is generation order stable when serialization is literally unchanged?
3. `solve_flip_rate` and `solve_flip_directions`: does a mathematical isomorphism change solve/fail outcome?
4. If enough both-correct pairs remain, inspect `tau_iso_per_puzzle` and its puzzle-cluster bootstrap CI.
5. Compare observed tau with `surface_order_positional_null_per_puzzle` and `boundary_first_positional_null_per_puzzle`, plus the corresponding excess-tau summaries.
6. Treat `easy_first_candidate_count_spearman_per_puzzle`, random-remasking tau, and `native_digit_argmax_fraction` as diagnostics/characterization, not the headline gate.

A low number of both-correct pairs caused by many `identity correct -> iso wrong` flips is **not** an unusable run; it is direct outcome non-equivariance. Only low identity competence or unstable same-serialization order prevents interpretation.

Do not change prompt, blank count, transform subset, model, or metric after discovery to rescue a preferred story.

## 6. Confirmation

Only after writing down the discovery interpretation, run the untouched confirmation half:

```bash
python src/run_g0.py \
  --config LOCKED_CONFIG.json \
  --manifest data/manifest.jsonl \
  --split confirmation \
  --overwrite \
  --out results/g0_confirmation_traces.jsonl

python src/analyze_g0.py \
  --config LOCKED_CONFIG.json \
  --traces results/g0_confirmation_traces.jsonl \
  --manifest data/manifest.jsonl \
  --split confirmation \
  --out results/g0_confirmation_summary.json
```

Confirmation intentionally does not repeat the discovery-only controls. Interpret it against the already frozen discovery claim.
