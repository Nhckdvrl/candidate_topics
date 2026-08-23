# G0 Results — Topic 22

## Verdict

**STOP at G0b: `SEED_PHENOMENON_NOT_REPRODUCED`.** Per the frozen protocol, G0c direct mode was not run.

## Run metadata

- Repository commit: `7fede5af018a9f6943385be9c70dcc70c843cb71`
- Model: `Qwen/Qwen3-14B` (local snapshot used for inference; exact model ID unchanged)
- Host/GPU: `fvcrc15`, four NVIDIA A100 80GB PCIe GPUs
- PyTorch/CUDA: `2.6.0+cu124` / CUDA 12.4
- Transformers: 5.15.1 (isolated target path); datasets 5.0.1; accelerate 1.14.0
- Seed: `20260823`; fixed random sample: 256 test pairs
- Pair dataset: `zhui711/MedEinst`, `test`
- CoT: thinking enabled, zero-shot prompt, max_new_tokens=1024, greedy decoding

## G0a pair locality

| Metric | Value | Gate |
|---|---:|---|
| valid pairs | 5383 | pass (>=5000) |
| malformed pairs | 0 | pass |
| ground-truth flip rate | 1.0000 | pass (>=0.99) |
| age/sex match rate | 1.0000 | pass (>=0.99) |
| changed-token fraction median | 0.0726 | pass (<=0.12) |
| changed-token fraction p90 | 0.2516 | pass (<=0.30) |

G0a verdict: `PAIR_STRUCTURE_OK`.

## G0b seed-faithful CoT screen

| Metric | Value | Gate |
|---|---:|---|
| control accuracy | 0.2070 (53/256) | fail (>=0.35) |
| control-correct count | 53 | pass (>=50) |
| exact Bias Trap count | 18 | fail (>=20) |
| Bias Trap rate among control-correct | 0.3396 | pass (>=0.30) |
| 95% Wilson lower bound | 0.2269 | pass (>=0.20) |
| diagnosis transitions | 6 | fail (>=8) |
| invalid-output rate | 0.8125 | fail (<=0.10) |

G0b verdict: `STOP_SEED_PHENOMENON_NOT_REPRODUCED`.

The relatively high conditional Bias Trap rate does not rescue the failed control accuracy, transition-density, and invalid-output gates. No prompt, model, parser, or decoding rescue was attempted.

Artifacts are retained locally under `artifacts/g0_pair_locality/` and `artifacts/g0_behavior_cot/`; the large raw records file is not intended for commit unless required by the repository policy.
