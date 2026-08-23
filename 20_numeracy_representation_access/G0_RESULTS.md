# Frozen same-prompt G0 result

## Verdict

`GO_CAUSAL_G1`

The primary frozen G0 was completed on Qwen/Qwen3-8B with the official
seed-0 `int_sci_compare` data and exact five-shot int-sci prompt. The critical
cell is sufficiently dense: 38 of 129 hard test cases are
probe-correct / generation-wrong.

## Integrity

- Repository synchronized with `git pull origin main` before execution.
- Official upstream repository: `VCY019/Numeracy-Probing`
- Upstream revision: `9e1be04b69965662886c79d543936389c5407d27`
- Official generator: `src/construct_data.py`, seed `0`, `int-sci`.
- Split sizes: train `8000`, validation `1600`, test `1600`.
- Data checksums (SHA-256):
  - train: `8a995020ecd21dc23f3a3ac97880652c78c85573fa95b53305a1f89004092914`
  - val: `73f0a6703283d186243b4f4db4238712e0e6b523757693553e3b33b202d33d2e`
  - test: `3688742a69a1b629447cce611d56d3ae73a68a26484042c9700a768b9a24b7b2`
- Static data audit: PASS. `int_sci_compare` has zero displayed ties, zero
  original ties, and zero ordering changes from scientific formatting. The
  frozen test hard count is `129/1600`; test answer-A rate is `0.4651163`.
- No `dec_sci_compare` data was used in the primary gate.

The Qwen3-8B snapshot was loaded locally from Hugging Face cache at revision
`b968826d9c46dd6066d109eabc6255188de91218`.

Environment:

- Python `3.12.13`
- conda environment: `/home/xiang/miniconda3/envs/fgvd`
- torch `2.11.0+cu130`, CUDA `13.0`
- transformers `5.12.1`
- scikit-learn `1.9.0`
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- model was run with `model.eval()`, deterministic greedy decoding, and
  `do_sample=False`.

Exact formal command:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/xiang/miniconda3/envs/fgvd/bin/python -u \
advisor_topic_search/g0/numeracy_same_prompt_g0.py \
--data-root /tmp/numeracy_data_seed0.s8Rp7d \
--model /home/xiang/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
--out-dir /home/xiang/candidate_topics/20_numeracy_representation_access/artifacts/g0 \
--batch-size 8 --max-new-tokens 40
```

## Frozen protocol

- Prompt: official `verbalization.py` int-sci prompt, `n_few_shot=5`,
  operator `larger`, `use_alt_prompt=false`.
- Demonstration answer positions: A, B, A, B, A.
- Hidden state: last input token before generation; left padding was used and
  the final sequence position was indexed directly.
- Probe: one Logistic Regression probe per transformer layer, trained on
  train; validation-selected layer; earliest layer on exact ties; test used
  once after layer selection.
- Hard regime: `abs(log2(a/b)) < 0.1`.
- Generation: greedy, `max_new_tokens=40`; raw completions and parseability
  are retained in the records.

## Layer selection

Selected layer: zero-based `35`, one-based `36`.

Validation accuracy by layer is recorded in
`artifacts/g0/layer_validation.csv` and in `artifacts/g0/summary.json`.
The selected validation accuracy was `0.999375`.

## Primary test metrics

| subset | N | probe accuracy | generation accuracy | gap | invalid rate |
|---|---:|---:|---:|---:|---:|
| full test | 1600 | 0.996875 | 0.817500 | 0.179375 | 0.000000 |
| hard test | 129 | 0.961240 | 0.682171 | 0.279070 | 0.000000 |

Hard subset critical metrics:

- `N_critical = n10 = 38`
- `R_critical = 38/129 = 0.294574`
- generation errors: `41`
- error coverage: `38/41 = 0.926829`

The hard 2x2 table is:

| | generation correct | generation wrong |
|---|---:|---:|
| probe correct | n11 = 86 | **n10 = 38** |
| probe wrong | n01 = 2 | n00 = 3 |

For completeness, the full-test 2x2 table is:

| | generation correct | generation wrong |
|---|---:|---:|
| probe correct | n11 = 1306 | **n10 = 289** |
| probe wrong | n01 = 2 | n00 = 3 |

## Decision and diagnosis

All frozen survival conditions pass:

1. full probe accuracy `0.996875 >= 0.90`;
2. hard probe accuracy `0.961240 >= 0.80`;
3. hard gap `0.279070 >= 0.15`;
4. hard critical count `38 >= 30`;
5. hard gap is positive;
6. hard invalid rate `0.0% < 5%`.

This is a clean same-prompt critical object, not a failure mode A/B/C/D/E.
The project may enter G1. G1 must preserve the frozen layer/token rule and
must not expand into an unbounded layer, token, prompt, threshold, or model
search. No G1 intervention was run in this G0 task.

## Artifacts

- `artifacts/g0/summary.json`
- `artifacts/g0/test_records.jsonl` (1600 complete test records)
- `artifacts/g0/records.csv`
- `artifacts/g0/critical_cases.csv` (289 full-test critical cases)
- `artifacts/g0/hard_critical_cases.csv` (38 hard critical cases)
- `artifacts/g0/layer_validation.csv`
- `artifacts/g0/data_audit.json`
