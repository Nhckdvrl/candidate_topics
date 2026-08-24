# G0 Results — Topic 22

## Current verdict

**MEASUREMENT_RUNTIME_FAILURE — repaired G0b completed, but the measurement is still invalid.**

The pair-structure result from G0a remains valid. The first Qwen3-14B CoT behavioral screen must not be used to decide the topic because the inference/scoring stack violated Qwen3 thinking-mode best practices and produced an 81.25% invalid-output rate.

## G0a pair locality — VALID

| Metric | Value | Gate |
|---|---:|---|
| valid pairs | 5383 | pass (>=5000) |
| malformed pairs | 0 | pass |
| ground-truth flip rate | 1.0000 | pass (>=0.99) |
| age/sex match rate | 1.0000 | pass (>=0.99) |
| changed-token fraction median | 0.0726 | pass (<=0.12) |
| changed-token fraction p90 | 0.2516 | pass (<=0.30) |

G0a verdict remains: `PAIR_STRUCTURE_OK`.

## First G0b run — INVALIDATED MEASUREMENT

Original run metadata:

- Model: `Qwen/Qwen3-14B`
- Host/GPU: `fvcrc15`, four NVIDIA A100 80GB PCIe GPUs
- Seed: `20260823`
- Fixed random sample: 256 test pairs
- Dataset: `zhui711/MedEinst`, `test`

Observed metrics were:

| Metric | Value |
|---|---:|
| control accuracy | 0.2070 (53/256) |
| exact Bias Trap count | 18 |
| Bias Trap rate among control-correct | 0.3396 |
| 95% Wilson lower bound | 0.2269 |
| diagnosis transitions | 6 |
| invalid-output rate | **0.8125** |

These numbers are retained for provenance only and are **not** a scientific verdict.

## Why the run is invalid

Three implementation choices made the run non-faithful as a Qwen3 thinking-mode measurement:

1. **Greedy decoding was used with `enable_thinking=True`.** The official Qwen3 model card explicitly recommends sampling (`temperature=0.6`, `top_p=0.95`, `top_k=20`) for thinking mode and warns against greedy decoding because it can degrade performance and cause pathological repetition.
2. **The reasoning budget was only 1024 new tokens.** The official Qwen3 example allocates up to 32768 new tokens for thinking. A capped reasoning trace may never reach the post-`</think>` final answer.
3. **Scoring required our custom literal `FINAL_DIAGNOSIS:` marker.** A valid final diagnosis phrased differently was counted as invalid even if the model had completed reasoning and supplied a canonical dataset diagnosis.

The combination is fully capable of explaining an 81.25% invalid rate. Therefore the previous `STOP_SEED_PHENOMENON_NOT_REPRODUCED` label is withdrawn.

## Frozen measurement repair v2

The scientific object is unchanged:

- same model: `Qwen/Qwen3-14B`;
- same dataset split;
- same random 256 pair IDs;
- same seed `20260823`;
- same Bias Trap definition;
- same G0b gate thresholds.

Only the broken measurement implementation is repaired:

- thinking mode uses Qwen3-recommended sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`;
- control/trap in each pair use the same deterministic per-pair sampling seed (common random numbers);
- CoT budget is 32768 new tokens, matching the official Qwen3 example budget ceiling;
- token-level `</think>` separation is used, and diagnosis extraction reads **only post-thinking final-answer content**;
- the preferred `FINAL_DIAGNOSIS:` marker is still accepted but no longer mandatory;
- conservative canonical-label extraction accepts an unambiguous diagnosis in the post-thinking final answer without using an LLM judge;
- invalid reasons are explicitly separated into `hit_max_tokens`, `thinking_not_closed`, and `unresolved_final`;
- extraction-method counts and thinking/truncation diagnostics are written to `summary.json`.

No threshold has been relaxed and no model/prompt/sample search is authorized.

## Repaired G0b rerun — completed, measurement-invalid

The frozen repair v2 rerun completed all 256 pairs on `fvcrc15` using `Qwen/Qwen3-14B`, seed `20260823`, four A100 GPUs, sampling `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `max_new_tokens=32768`.

| Metric | Value | Gate |
|---|---:|---|
| control accuracy | 0.3555 (91/256) | pass (>=0.35) |
| Bias Trap count | 34 | pass (>=20) |
| Bias Trap rate among control-correct | 0.3736 | pass (>=0.30) |
| 95% Wilson lower bound | 0.2812 | pass (>=0.20) |
| diagnosis transitions | 12 | pass (>=8) |
| invalid-output rate | **0.6250 (160/256)** | **fail (<=0.10)** |

All 256 control and trap thinking traces closed; neither side hit the token limit. The invalid outputs were `unresolved_final` (control 109, trap 124). Because the invalid-rate gate failed, the verdict is `MEASUREMENT_RUNTIME_FAILURE`; no direct-mode G0c was run and no scientific hypothesis verdict is assigned.

The repaired artifact is `artifacts/g0_behavior_cot/summary.json`. The result is not a scientific negative: the behavioral signal gates pass on the valid subset, but 62.5% invalid output makes the measurement unusable under the frozen protocol.

## Follow-up

Under the frozen protocol, direct mode must not run after this runtime failure. Any further attempt requires another explicitly reviewed measurement repair; the current result cannot be used as either reproduction or non-reproduction.

The command used for the completed repair rerun was:

```bash
cd 22_medeinst_evidence_update
MODEL=Qwen/Qwen3-14B \
N_PAIRS=256 \
SEED=20260823 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_g0.sh
```

The frozen interpretation is: because invalid rate remained above 0.10, report `MEASUREMENT_RUNTIME_FAILURE` and do not claim the MedEinst phenomenon is false.
