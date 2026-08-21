# Archive Summary — Topic 07: Old Blocks New, or New Erases Old?

**Final status: ARCHIVED / INCONCLUSIVE**

The frozen discovery pilot completed for all four matched M-A-P 1.3B/100B checkpoints. The Transformer reproduced the motivating positive PI>RI asymmetry, but the preregistered Transformer–GatedDeltaNet architecture gap was neither a robust positive signal nor a practically null effect. The topic is archived without locked confirmation and without post-hoc tuning.

## Frozen pilot integrity

- 4 architectures × 192 rows = 768 scoring rows;
- zero skipped rows;
- zero duplicate rows;
- 192 paired episode/query/level/condition cells shared across architectures;
- tokenizer fingerprint shared across the family;
- maximum boundary shift: 0;
- mean target token count: 2.77;
- all sampled prompts remained within context safety limits.

## Primary results

| architecture | U=1 I | U=3 I | U=7 I | U=15 I | mean I |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | -0.0833 | 0.2500 | 0.2500 | 0.2083 | 0.1563 |
| GLA | -0.0833 | 0.0000 | 0.2083 | -0.0417 | 0.0208 |
| DeltaNet | 0.1250 | 0.1250 | 0.0833 | 0.1667 | 0.1250 |
| Gated DeltaNet | 0.0833 | 0.0417 | 0.1250 | 0.0833 | 0.0833 |

Primary contrast:

```text
Delta_I = mean(I_Transformer) - mean(I_GatedDeltaNet)
         = 0.0729
paired bootstrap 95% CI = [-0.0313, 0.1771]
```

The Transformer mean `I` was positive, so this is not `PARADIGM_FAIL`. The contrast did not meet `Delta_I >= 0.10` with a positive 95% lower bound, so it is not `GO_TO_LOCKED_CONFIRMATION` or `STRONG_GO`. It also did not meet `abs(Delta_I) < 0.05`, so it is not `KILL`.

## Decision

**INCONCLUSIVE_DO_NOT_TUNE**

This status is deliberately not labeled “falsified.” The seed Transformer PI>RI phenomenon was observed, but the discovery pilot did not resolve whether the update architecture creates the preregistered separation. No locked confirmation was run, and no additional models, prompts, metrics, update levels, or mechanistic probes should be added as a rescue operation.

## Preserved outputs

The complete pilot outputs are preserved under `outputs/architecture_pi_ri_pilot/`, including `results.jsonl`, `resolved_config.json`, `summary.csv`, `pairwise_bootstrap.json`, `token_audit.json`, `intrusions.json`, `merge_audit.json`, and `decision.json`.
