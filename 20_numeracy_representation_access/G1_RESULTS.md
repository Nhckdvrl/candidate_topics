# Frozen G1 result

## Verdict

`STOP_G1_NONREPLICATION`

G1-P0 was executed on the fresh upstream seed `20260824`. The fresh object
contains enough hard critical cases, and the exploratory notation pattern
replicates descriptively, but the frozen seed-0 probe does not meet the
predeclared fresh hard accuracy threshold of `0.90`:

```text
fresh unique hard probe accuracy = 0.8985507246
```

Per the frozen protocol, no rank reflection or random-direction intervention
was run. This is not a causal-null result and must not be reported as one.

## Integrity and environment

- Repository was synchronized with `git pull origin main`.
- Model: local Qwen/Qwen3-8B snapshot
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Upstream repository: `VCY019/Numeracy-Probing`
- Upstream revision: `9e1be04b69965662886c79d543936389c5407d27`
- Fresh generator: official `src/construct_data.py`, `int-sci`, seed `20260824`
- Python: `3.12.13`
- conda environment: `/home/xiang/miniconda3/envs/fgvd`
- torch: `2.11.0+cu130`; transformers: `5.12.1`; scikit-learn: `1.9.0`
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- Prompt, tokenizer padding, greedy decoding, and parsing remained the frozen
  G0 contract.

Exact command:

```bash
UPSTREAM_ROOT=/tmp/Numeracy-Probing-g0 \
MODEL=/home/xiang/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
PYTHON_BIN=/home/xiang/miniconda3/envs/fgvd/bin/python \
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash 20_numeracy_representation_access/run_g1.sh
```

Seed-0 integrity was rechecked against G0:

- train SHA-256: `8a995020ecd21dc23f3a3ac97880652c78c85573fa95b53305a1f89004092914`
- val SHA-256: `73f0a6703283d186243b4f4db4238712e0e6b523757693553e3b33b202d33d2e`

Fresh test SHA-256:

```text
e58a210c6809b62641d97ec49d21f4c9bdbef64d9dba1e0b10f924dda32f6d67
```

The fresh test had 1600 raw rows, 1598 unique displayed pairs, 0 ties, and 2
excluded exact displayed duplicates. The preserved raw fresh test is in
`artifacts/g1/fresh_seed20260824_test.jsonl`.

## Fresh G1-P0 baseline

The probe was trained only on the original seed-0 train split at the frozen
`L_sat` layer, block 19 zero-based / layer 20 one-based. Its seed-0 validation
accuracy was exactly `0.990625`, matching the predeclared value.

| subset | N | probe accuracy | generation accuracy | invalid rate |
|---|---:|---:|---:|---:|
| fresh unique full | 1598 | 0.989987 | 0.776596 | — |
| fresh unique hard | 138 | 0.898551 | 0.565217 | 0.000000 |

Fresh hard critical object:

- `n_critical = 51`
- critical rate: `51/138 = 0.369565`
- hard generation errors: `60`
- exact-operand hard errors: `60`
- fresh object gate: **FAIL** only because hard probe accuracy is below `0.90`

The predeclared fresh-object conditions were:

| condition | result |
|---|---:|
| hard N >= 100 | PASS (`138`) |
| hard probe accuracy >= 0.90 | **FAIL (`0.898551`)** |
| unique hard critical >= 25 | PASS (`51`) |
| critical rate >= 0.20 | PASS (`0.369565`) |
| invalid rate < 0.05 | PASS (`0.0`) |

## Fresh notation confirmation

Among fresh hard generation errors whose first parsed answer exactly matched an
input operand, `55/60 = 0.916667` selected the scientific-notation operand.
Thus the descriptive notation follow-up threshold (`>= 0.80`) is met. This
does not override the failed fresh-object gate and no notation intervention
was run.

## Stopping decision

The correct diagnosis is fresh-object nonreplication under the frozen
probe-quality gate, not a causal null and not an implementation failure. The
G1 rank-reflection and eight norm-matched random-null populations were not
constructed or evaluated because P0 failed. Do not change layer, threshold,
prompt, model, or seed to rescue this run.

Artifacts:

- `artifacts/g1/fresh_data_audit.json`
- `artifacts/g1/fresh_baseline_records.jsonl`
- `artifacts/g1/fresh_baseline_summary.json`
- `artifacts/g1/fresh_seed20260824_test.jsonl`
- `artifacts/g1/fresh_seed20260824_test.sha256`
- `artifacts/g1/rank_probe_lsat.npz`
- `artifacts/g1/rank_causal_summary.json`
