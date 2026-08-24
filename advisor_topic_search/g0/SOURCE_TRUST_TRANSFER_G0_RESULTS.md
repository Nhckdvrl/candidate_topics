# Source-Trust Transfer G0 — Result

Date: 2026-08-24

Verdict: `KILL_SOURCE_LEVEL_TRANSFER_PAPER_SCALE`

The frozen G0 was run exactly as specified after syncing `main` to
`b07d6a4ad14bf5476785c5bc27d06cf360b8d172`.

## Frozen run

- upstream artifact commit: `87dd466f10a76ea1cadc21a552d423d2d60c0cce`
- model: `google/gemma-3-4b-it`
- seed: `20260824`
- independent target items: `128`
- total one-token probability prompts: `2048`
- inference unit: independent target item

## Result

| Metric | Observed | Frozen requirement |
| --- | ---: | ---: |
| mean transfer delta | `-1.319 pp` | `>= 5 pp` |
| bootstrap 95% CI | `[-2.882, +0.290] pp` | lower `> 0` |
| positive item fraction | `49.22%` | `>= 60%` |
| discrete crossover count | `6` | `>= 12` |

Counterbalance means were all negative:

```text
history forward     -0.458 pp
history reversed    -2.181 pp
target canonical     -0.984 pp
target swapped       -1.654 pp
answer canonical     -1.571 pp
answer swapped       -1.068 pp
```

The paired prompt audit passed before inference. Across the audited paired
conditions, target sections were byte-identical and total prompt lengths were
identical. The complete outputs are in:

```text
artifacts/source_trust_transfer_g0/summary.json
artifacts/source_trust_transfer_g0/records.jsonl
artifacts/source_trust_transfer_g0/prompt_audit.jsonl
```

The result does not support paper-scale transfer of repetition association to
an unrelated novel claim. Per the frozen protocol, do not register Topic 23,
do not run hidden-state mechanism work, and do not tune the model, source type,
history, prompt, sample size, or gate to rescue this result.
