# G0 Results — Topic 22

## Current verdict

**RERUN REQUIRED — the first G0b run is measurement-invalid, not a scientific negative.**

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

## Next action

Rerun:

```bash
cd 22_medeinst_evidence_update
MODEL=Qwen/Qwen3-14B \
N_PAIRS=256 \
SEED=20260823 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_g0.sh
```

Interpret the repaired G0b exactly as frozen:

- if G0b passes, continue automatically to direct-mode G0c;
- if G0b fails with a healthy invalid rate (`<=0.10`), treat that as a real reproduction failure and stop;
- if invalid remains high because thinking still fails to terminate within 32768 tokens, report `MEASUREMENT_RUNTIME_FAILURE` rather than claiming the MedEinst phenomenon is false.
