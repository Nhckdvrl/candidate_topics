# Frozen G2 notation-competition result

## Verdict

`NOTATION_READABLE_BUT_NOT_CAUSAL_AT_LSAT`

The notation-side attractor replicated on untouched seed `20260825`, and the
seed-0-only ranking and notation coordinates passed their representation checks.
However, neutralizing the notation coordinate at the frozen layer-20 site did
not rescue any of the 32 primary errors. All eight norm-matched orthogonal nulls
also produced zero rescues.

This is a clean frozen null for the tested one-dimensional notation coordinate
at `L_sat`; it is not a failure of the behavioral notation observation and it
does not justify a layer/token/strength rescue search.

## Integrity and environment

- Primary causal seed: `20260825`
- Upstream: `VCY019/Numeracy-Probing`
- Upstream revision: `9e1be04b69965662886c79d543936389c5407d27`
- Model: local Qwen/Qwen3-8B snapshot `b968826d9c46dd6066d109eabc6255188de91218`
- Python: `3.12.13`
- conda environment: `/home/xiang/miniconda3/envs/fgvd`
- torch: `2.11.0+cu130`; transformers: `5.12.1`; scikit-learn: `1.9.0`
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- Exact prompt and greedy generation remained the frozen G0 contract.

Exact formal command:

```bash
UPSTREAM_ROOT=/tmp/Numeracy-Probing-g0 \
MODEL=/home/xiang/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
PYTHON_BIN=/home/xiang/miniconda3/envs/fgvd/bin/python \
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash 20_numeracy_representation_access/run_g2.sh
```

The fresh causal test had SHA-256:

```text
65820bbb99c8d8b474085b48c8b5645d251eb4310cf7e9c84685410a3fe4eb9f
```

The exact checksum is also preserved in
`artifacts/g2/fresh_seed20260825_test.sha256`.

## G2-P0 fresh object

After excluding exact displayed duplicates and ties:

- unique hard: `123`
- hard exact-operand generation errors: `41`
- hard errors choosing scientific operand: `39`
- scientific-side error rate: `39/41 = 0.951220`
- object gate: PASS

Fresh baseline metrics:

| subset | probe accuracy | generation accuracy |
|---|---:|---:|
| full unique test (`1598`) | 0.989362 | 0.790989 |
| hard unique test (`123`) | 0.861789 | 0.666667 |

The causal primary population contains `32` unique hard cases satisfying:

1. frozen ranking probe correct;
2. baseline generation wrong;
3. baseline output exactly equals an input operand;
4. selected operand is the scientific-notation operand.

## Frozen representation checks

- `L_sat`: transformer block 19 zero-based / layer 20 one-based
- seed-0 ranking validation accuracy: `0.990625`
- seed-0 notation validation accuracy after orthogonalization: `1.000000`
- raw notation probe validation accuracy: `1.000000`
- notation/ranking cosine: `1.96e-17`
- representation checks: PASS

The notation direction was trained on seed-0 train only, orthogonalized against
the frozen ranking direction, and its scalar threshold was fit without fresh
test fitting.

## Causal intervention result

Notation neutralization moved each notation projection to its frozen threshold
while preserving the ranking projection. Manipulation checks passed:

- maximum notation threshold residual: `1.82e-06`
- maximum ranking-logit change under neutralization: `1.29e-05`
- maximum ranking-logit change under random nulls: `2.00e-05`
- prefill calls: `4/4`

| intervention | rescue rate |
|---|---:|
| notation neutralization (`R_not`) | 0/32 = 0.000000 |
| random 20260901 | 0.000000 |
| random 20260902 | 0.000000 |
| random 20260903 | 0.000000 |
| random 20260904 | 0.000000 |
| random 20260905 | 0.000000 |
| random 20260906 | 0.000000 |
| random 20260907 | 0.000000 |
| random 20260908 | 0.000000 |

Therefore:

- `R_null = 0.000000`
- `DeltaR = 0.000000`
- paired bootstrap 95% CI: `[0.000000, 0.000000]`
- invalid/neither rate under neutralization: `0%`
- changed valid outputs: `0`

## Interpretation

The evidence now supports a narrower two-part conclusion:

1. The scientific-notation operand is a robust behavioral attractor on the
   discovery seed and two untouched confirmation/causal seeds.
2. The specific seed-0-trained, layer-20, one-dimensional notation coordinate
   tested here is not causally sufficient to rescue those errors when
   neutralized while preserving the ranking projection.

This does not prove that no notation-related mechanism exists anywhere in the
network. It does establish the preregistered null at `L_sat`. Per G2, do not
open a layer/token/strength/subspace/model search in this stage.

## Artifacts

- `artifacts/g2/notation_causal_summary.json`
- `artifacts/g2/fresh_baseline_summary.json`
- `artifacts/g2/fresh_baseline_records.jsonl`
- `artifacts/g2/fresh_data_audit.json`
- `artifacts/g2/fresh_seed20260825_test.jsonl`
- `artifacts/g2/notation_representation_checks.json`
- `artifacts/g2/rank_probe_lsat.npz`
- `artifacts/g2/notation_probe_lsat.npz`
- `artifacts/g2/notation_neutralization_records.jsonl`
- `artifacts/g2/random_null_records.jsonl`
