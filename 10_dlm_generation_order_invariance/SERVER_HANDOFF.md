# Server handoff — Topic 10 G0

Run from `10_dlm_generation_order_invariance/`.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A recent CUDA PyTorch build and enough VRAM for `GSAI-ML/LLaDA-8B-Instruct` in BF16 are required.

## 2. Preflight

```bash
python src/preflight.py --config LOCKED_CONFIG.json
pytest -q tests
```

Do not proceed if either fails.

## 3. Materialize the frozen manifest

```bash
python src/make_manifest.py --config LOCKED_CONFIG.json --out data/manifest.jsonl
```

The manifest contains both discovery and untouched confirmation puzzles. Do not regenerate it with another seed after seeing results.

## 4. Cheap smoke run

```bash
python src/run_g0.py \
  --config LOCKED_CONFIG.json \
  --manifest data/manifest.jsonl \
  --split discovery \
  --limit 8 \
  --include-random-control \
  --out results/smoke_traces.jsonl

python src/analyze_g0.py \
  --traces results/smoke_traces.jsonl \
  --manifest data/manifest.jsonl \
  --split discovery \
  --out results/smoke_summary.json
```

Smoke is for plumbing only. Do not tune scientific thresholds on it.

## 5. Frozen discovery G0

```bash
python src/run_g0.py \
  --config LOCKED_CONFIG.json \
  --manifest data/manifest.jsonl \
  --split discovery \
  --include-random-control \
  --out results/g0_discovery_traces.jsonl

python src/analyze_g0.py \
  --traces results/g0_discovery_traces.jsonl \
  --manifest data/manifest.jsonl \
  --split discovery \
  --out results/g0_discovery_summary.json
```

## 6. Decision order

Read fields in this order:

1. `identity_exact_accuracy` — can the model solve enough cases?
2. `n_valid_exact_isomorph_pairs >= 50` — enough matched successful pairs?
3. `seed_replication_candidate_count_vs_finalization_spearman >= 0.15` — seed easy-first signal present?
4. `random_remasking_tau_vs_identity.mean` near zero — instrumentation negative control sane?
5. only then interpret `tau_iso` and the positional diagnostic.

Do **not** swap prompts, change blank count, add model families, change the transform set, or change the metric after seeing discovery unless the current protocol is formally killed and a new version is registered.

## 7. Confirmation

Only if discovery yields a large, interpretable result worth a paper, freeze that interpretation in a short result note and run:

```bash
python src/run_g0.py \
  --config LOCKED_CONFIG.json \
  --manifest data/manifest.jsonl \
  --split confirmation \
  --out results/g0_confirmation_traces.jsonl

python src/analyze_g0.py \
  --traces results/g0_confirmation_traces.jsonl \
  --manifest data/manifest.jsonl \
  --split confirmation \
  --out results/g0_confirmation_summary.json
```
