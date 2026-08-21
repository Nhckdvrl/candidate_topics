# Archive Summary — Topic 07: Old Blocks New, or New Erases Old?

**Final status: ARCHIVED / INCONCLUSIVE AT FROZEN DISCOVERY GATE**

The frozen discovery pilot completed for all four matched M-A-P 1.3B/100B checkpoints. The Transformer reproduced the motivating positive PI>RI asymmetry, so the task and measurement were behaviorally active. However, the preregistered Transformer–GatedDeltaNet architecture gap was neither a robust positive signal nor a practically null effect. Under the frozen decision contract, the correct outcome is `INCONCLUSIVE_DO_NOT_TUNE`.

This is therefore **not a falsification of PI>RI**, and it is also **not evidence strong enough to justify continuing the architecture explanation**. The topic is archived without locked confirmation and without post-hoc tuning.

## Frozen pilot integrity

- 4 architectures × 192 rows = 768 scoring rows;
- zero skipped rows;
- zero duplicate rows;
- 192 paired episode/query/level/condition cells shared across architectures;
- tokenizer fingerprint shared across the family;
- maximum boundary shift: 0;
- mean target token count: 2.77;
- all sampled prompts remained within context safety limits.

The merge audit therefore passed cleanly. The decision is not attributable to missing rows, broken pairing, tokenizer boundary failures, or a collapsed Transformer baseline.

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
paired RI/PI units = 96
sign-transition levels = 0 / 4
```

The Transformer mean `I` was positive, so this is not `PARADIGM_FAIL`. The contrast did not meet `Delta_I >= 0.10` with a positive 95% lower bound, so it is not `GO_TO_LOCKED_CONFIRMATION` or `STRONG_GO`. It also did not meet `abs(Delta_I) < 0.05`, so the frozen decision rule does not call it a clean `KILL` either.

## Why this is still a stop rather than “just add more samples”

The goal of this repository is candidate selection, not forcing every uncertainty interval to a definitive hypothesis test. Three observations matter here.

1. **The point estimate itself is below the preregistered minimum worthwhile GO effect.** `Delta_I=0.0729` is not merely a >0.10 effect with insufficient power. Increasing `n` would narrow the interval, but if the center stayed similar it would only make a sub-threshold architecture effect more precisely estimated.
2. **The hoped-for qualitative regime change did not appear.** There were `0/4` frozen levels with the strong sign-transition pattern `I_Transformer>0` and `I_GatedDeltaNet<0`.
3. **The broader architecture ordering is not a clean monotonic editability story.** DeltaNet (`mean I=0.1250`) remained close to the Transformer (`0.1563`), while GLA was much lower (`0.0208`). The pilot therefore does not support a simple graded narrative in which increasingly writable/updateable memory smoothly moves the system from primacy/PI toward recency/RI.

A tempting post-hoc observation is that Transformer–GLA is numerically larger than the preregistered Transformer–GatedDeltaNet contrast. That comparison was **not** the frozen primary test and must not replace it after seeing the results. Doing so would be architecture shopping.

## Decision

**INCONCLUSIVE_DO_NOT_TUNE → ARCHIVE**

This status is deliberately not labeled “falsified.” The seed Transformer PI>RI phenomenon was observed, but the discovery pilot did not establish the preregistered claim that the memory-update architecture is a strong determinant of interference direction. No locked confirmation was run because the discovery gate did not authorize it.

Do not rescue this candidate by adding GDN2/Mamba, searching prompts or update levels, changing the metric, switching the primary architecture pair, increasing sample size solely to chase significance, or introducing mechanistic probes. A genuinely new external result could motivate a separately registered question; it would not retroactively turn this pilot into a GO.

## Main lesson

> **A phenomenon can be real while the proposed explanatory axis is not important enough to build a paper around.**

Topic 07 successfully reproduced the motivating interference phenomenon. What failed to become compelling was the next claim: that the chosen memory-update architecture axis produces a large, stable, qualitatively different PI/RI regime.

For future candidate selection, validate both layers separately:

```text
Does the phenomenon exist in the selected system?
        ↓
Does the explanatory axis create a large, clean separation?
```

Passing the first question is not evidence that the second is worth continued investment.

## Preserved outputs

The complete pilot outputs are preserved under `outputs/architecture_pi_ri_pilot/`, including:

- `results.jsonl`
- `resolved_config.json`
- `summary.csv`
- `pairwise_bootstrap.json`
- `token_audit.json`
- `intrusions.json`
- `merge_audit.json`
- `decision.json`

The frozen validation protocol, runnable code, and literature audit remain in the Topic 07 directory for provenance and to prevent the same candidate from being rediscovered through a post-hoc architecture/metric variant.
