# Topic 11 G-0 result

## Frozen verdict

**`KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL`**

The locked v3 G-0 completed successfully. Both protocol prerequisites passed by a large margin, but the preregistered primary retroactive consistency effect was essentially zero and its confidence interval excluded the predeclared minimum-worthy effect by orders of magnitude.

This is a valid scientific negative, not a protocol failure.

## Prerequisites

| Probe | Mean gap | 95% CI | Positive pairs | Locked floor | Result |
|---|---:|---:|---:|---:|---|
| Arithmetic result substitution | `0.426361` | `[0.390624, 0.462064]` | `1.000` | `0.100` | PASS |
| Semantic-alias comprehension | `0.214693` | `[0.186151, 0.244417]` | `1.000` | `0.020` | PASS |

The scorer therefore had strong arithmetic discrimination and understood the semantic-alias construction used by the factorial. The primary null cannot reasonably be attributed to a broken confidence readout or failure to understand the arithmetic aliases.

## Sample

- Eligible mirrored anchor pairs: **140**
- Digit-level tokenizer compatibility repair was applied only to preserve the locked one-token eligibility semantics.
- The scientific factors, primary metric, thresholds, and verdict logic were not changed.

## Primary result

Primary metric: `confidence_result_middle`, the Step-2/Step-3 result-token confidence measured **before** the future consistency-check suffix.

```text
Delta_consistency = -0.000003
95% CI            = [-0.000055, 0.000025]
positive pairs    = 0.943
sign-flip p       = 0.5022
```

The predeclared meaningful-effect floor was:

```text
Delta_consistency >= 0.010
```

The upper confidence bound, `0.000025`, is roughly **400× smaller** than the frozen `0.010` floor. Increasing the number of pairs cannot plausibly turn this into the preregistered scientific effect.

The corresponding correctness effect on the same primary tokens was small but detectable:

```text
Delta_correctness = 0.000584
95% CI            = [0.000030, 0.001685]
```

The strongest cross-cell contrast also went opposite to the hoped-for headline:

```text
CW - IC = -0.000587
95% CI  = [-0.001737, -0.000009]
```

So coherent-but-wrong did **not** outrank incoherent-but-correct on the locked retroactive readout.

## Full locked table

| Metric | Effect | Mean | 95% CI | Positive pairs | sign-flip p |
|---|---|---:|---:|---:|---:|
| confidence_result_middle | delta_consistency | -0.000003 | [-0.000055, 0.000025] | 0.943 | 0.5022 |
| confidence_result_middle | delta_correctness | 0.000584 | [0.000030, 0.001685] | 0.943 | 5e-05 |
| confidence_result_middle | consistency_when_correct | -0.000041 | [-0.000167, 0.000024] | 0.943 | 0.502 |
| confidence_result_middle | consistency_when_wrong | 0.000035 | [0.000020, 0.000062] | 0.914 | 5e-05 |
| confidence_result_middle | coherent_wrong_minus_incoherent_correct | -0.000587 | [-0.001737, -0.000009] | 0.150 | 1 |
| confidence_result_middle | prompt_check_match_interaction | -0.000076 | [-0.000225, 0.000000] | 0.436 | 0.9535 |
| confidence_result_first | delta_consistency | 0.000261 | [0.000221, 0.000305] | 0.993 | 5e-05 |
| confidence_result_first | delta_correctness | 0.000533 | [0.000464, 0.000609] | 0.993 | 5e-05 |
| confidence_result_first | consistency_when_correct | 0.000095 | [0.000076, 0.000116] | 0.979 | 5e-05 |
| confidence_result_first | consistency_when_wrong | 0.000427 | [0.000361, 0.000502] | 0.986 | 5e-05 |
| confidence_result_first | coherent_wrong_minus_incoherent_correct | -0.000272 | [-0.000322, -0.000224] | 0.079 | 1 |
| confidence_result_first | prompt_check_match_interaction | -0.000333 | [-0.000398, -0.000274] | 0.050 | 1 |
| confidence_result_final | delta_consistency | 0.000083 | [0.000027, 0.000175] | 0.936 | 5e-05 |
| confidence_result_final | delta_correctness | 0.000235 | [0.000100, 0.000474] | 0.993 | 5e-05 |
| confidence_result_final | consistency_when_correct | 0.000122 | [0.000041, 0.000275] | 0.914 | 5e-05 |
| confidence_result_final | consistency_when_wrong | 0.000044 | [0.000000, 0.000076] | 0.850 | 0.0016 |
| confidence_result_final | coherent_wrong_minus_incoherent_correct | -0.000152 | [-0.000313, -0.000049] | 0.107 | 1 |
| confidence_result_final | prompt_check_match_interaction | 0.000078 | [-0.000012, 0.000234] | 0.421 | 0.1973 |
| confidence_result_all | delta_consistency | 0.000076 | [0.000065, 0.000087] | 0.986 | 5e-05 |
| confidence_result_all | delta_correctness | 0.000470 | [0.000149, 0.001093] | 1.000 | 5e-05 |
| confidence_result_all | consistency_when_correct | 0.000034 | [0.000012, 0.000047] | 0.971 | 5e-05 |
| confidence_result_all | consistency_when_wrong | 0.000118 | [0.000096, 0.000144] | 0.964 | 5e-05 |
| confidence_result_all | coherent_wrong_minus_incoherent_correct | -0.000394 | [-0.001019, -0.000072] | 0.100 | 1 |
| confidence_result_all | prompt_check_match_interaction | -0.000084 | [-0.000126, -0.000055] | 0.129 | 1 |
| confidence_trajectory | delta_consistency | 0.003418 | [0.002837, 0.003985] | 0.871 | 5e-05 |
| confidence_trajectory | delta_correctness | 0.006506 | [0.005653, 0.007363] | 0.893 | 5e-05 |
| confidence_trajectory | consistency_when_correct | 0.002094 | [0.001402, 0.002786] | 0.736 | 5e-05 |
| confidence_trajectory | consistency_when_wrong | 0.004743 | [0.003931, 0.005543] | 0.843 | 5e-05 |
| confidence_trajectory | coherent_wrong_minus_incoherent_correct | -0.003088 | [-0.004012, -0.002194] | 0.264 | 1 |
| confidence_trajectory | prompt_check_match_interaction | -0.002648 | [-0.003633, -0.001663] | 0.307 | 1 |
| confidence_full | delta_consistency | 0.013765 | [0.013069, 0.014462] | 1.000 | 5e-05 |
| confidence_full | delta_correctness | 0.002845 | [0.002196, 0.003503] | 0.771 | 5e-05 |
| confidence_full | consistency_when_correct | 0.016244 | [0.015370, 0.017132] | 1.000 | 5e-05 |
| confidence_full | consistency_when_wrong | 0.011285 | [0.010417, 0.012104] | 0.957 | 5e-05 |
| confidence_full | coherent_wrong_minus_incoherent_correct | 0.010920 | [0.009987, 0.011851] | 0.957 | 5e-05 |
| confidence_full | prompt_check_match_interaction | 0.004959 | [0.003944, 0.005981] | 0.771 | 5e-05 |

## Why the positive `confidence_full` result does not rescue the topic

`confidence_full` includes the manipulated future consistency-check suffix. Its strong consistency effect:

```text
Delta_consistency = 0.013765
95% CI            = [0.013069, 0.014462]
```

therefore shows that aggregate full-output confidence is highly sensitive to whether the complete sequence is consistent. That is useful as a diagnostic, but it does **not** establish the preregistered claim that the consistency signal is meaningfully represented across earlier reasoning tokens.

The decisive comparison is precisely the contrast between:

```text
full output, includes manipulated suffix    -> +0.013765
unchanged middle reasoning results          -> -0.000003
```

The experiment was designed so that this divergence answers the question rather than creating an invitation to switch metrics.

## Scientific conclusion

The frozen G-0 supports the following bounded statement:

> Under a controlled retroactive intervention, LLaDA's native final-forward confidence does not exhibit a meaningful global consistency signal on unchanged middle reasoning-result tokens. Strong sequence-level consistency sensitivity therefore should not by itself be interpreted as evidence that confidence is globally distributed over the reasoning trajectory.

This does **not** imply that DLM confidence contains no consistency information anywhere. It rules out the stronger project-level interpretation that motivated Topic 11.

## Final action

- Archive Topic 11.
- Do not run G-1.
- Do not increase `n`, switch the primary metric, search token regions, change pooling, or sweep models to rescue the same claim.
- Reopen only if an independent external result motivates a genuinely new question rather than a post-hoc metric pivot.
