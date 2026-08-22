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

Use a recent CUDA PyTorch build and enough VRAM for `GSAI-ML/LLaDA-8B-Instruct` in BF16. Do not proceed past a failed preflight: tokenizer/template/isomorphism failures are measurement failures, not scientific results.

## 2. Regenerate the v2 frozen manifest

G0 v2 changed both the controlled decoder and base Sudoku generator. **Do not reuse any old g0-v1 manifest or result file.** Protocol versions are now stamped and mismatches fail loudly.

```bash
rm -f data/manifest.jsonl
python src/make_manifest.py --config LOCKED_CONFIG.json --out data/manifest.jsonl
```

This materializes both discovery and reserved confirmation puzzles. Do not regenerate with another seed after looking at discovery results.

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

Smoke is engineering-only. Check that Sudoku outputs are plausible, `native_scheduler_pick_same_fraction` is not catastrophically low, and no protocol/template assertion fires. Do not tune scientific thresholds on four puzzles.

## 4. Frozen discovery

Single GPU:

```bash
python src/run_g0.py --split discovery --overwrite --out results/g0_discovery_traces.jsonl
python src/analyze_g0.py --config LOCKED_CONFIG.json --traces results/g0_discovery_traces.jsonl --manifest data/manifest.jsonl --split discovery --out results/g0_discovery_summary.json
```

For four GPUs, run one process per GPU with deterministic puzzle sharding, e.g. shard `i` uses `--num-shards 4 --shard-index i --device cuda:i` and a distinct `results/g0_discovery_shard{i}.jsonl`. The analyzer accepts all shard files directly:

```bash
python src/analyze_g0.py \
  --config LOCKED_CONFIG.json \
  --traces results/g0_discovery_shard0.jsonl results/g0_discovery_shard1.jsonl results/g0_discovery_shard2.jsonl results/g0_discovery_shard3.jsonl \
  --manifest data/manifest.jsonl \
  --split discovery \
  --out results/g0_discovery_summary.json
```

If a runner is interrupted, rerun that same shard with `--resume`; completed trace keys are skipped **before** GPU inference. Do not concatenate files manually. Duplicate keys and stale protocol versions are rejected by analysis.

## 5. Read the result in this order

1. `n_identity_exact` / `identity_exact_accuracy`: does the controlled Sudoku protocol have enough competence?
2. `same_serialization_repeat_tau`: is order measurable and stable when serialization is unchanged?
3. `native_scheduler_pick_same_fraction`: does the digit grammar preserve the position choices of native full-vocabulary confidence scheduling well enough for a native-DLM interpretation?
4. `solve_flip_rate`, its puzzle-cluster CI, and flip directions: does exact isomorphism change solve/fail outcome?
5. If enough both-correct pairs remain, inspect `tau_iso_per_puzzle`, its puzzle-cluster CI, and observed-minus-positional-null CIs.
6. Treat candidate-count/easy-first and random-remasking summaries as diagnostics, not headline gates.

Many `identity correct -> iso wrong` flips are already a scientific non-equivariance result, not missing data. Conversely, a strong positive claim needs high outcome retention and order preservation clearly above the row-major / boundary positional nulls.

Do not change prompt, blank count, transform subset, model, or primary metric after seeing discovery merely to rescue a preferred story. Engineering bugs may be fixed, but if they alter the scientific protocol, bump the protocol version and regenerate the frozen manifest.

## 6. Confirmation

Only after writing down the discovery interpretation, run the untouched confirmation half with the same protocol. Confirmation intentionally omits discovery-only repeat/random controls and must be interpreted against the already frozen claim.
