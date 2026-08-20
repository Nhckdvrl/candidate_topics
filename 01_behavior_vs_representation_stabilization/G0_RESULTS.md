# G0 validation results

## Decision

```yaml
G0-A: PASS
G0-B: FAIL
final_decision: KILL TOPIC
```

The behavioral premise is clear, but the representation screen does not show
the required temporal decoupling. From the first to the last fixed-1k pair,
robust behavioral movement falls to `0.0537x`, while cosine and standardized
representation drift fall further, to `0.0302x` and `0.0343x`. No G1
crosscoder experiment was run.

These results were produced from commit
`8b704cde82f757648a043b490d54f9e459397174` on the `main` branch.

## Environment

- Model: `EleutherAI/pythia-410m`
- GPU: 4 x NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB each
- NVIDIA driver: 580.82.07
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- Transformers: 5.15.1
- Batch size: 16
- Representation location: middle block, layer 12, `resid_pre`
- Positions per text: 4
- Corpus examples: 1,000
- Corpus SHA-256:
  `2e121b2a329df30b4d50212da610f0ce7496a610fc8316019ce6fa8f578a534e`
- Observed token-length range: 52-556; no truncation
- All 14 checkpoint revisions loaded successfully
- No OOM, NaN, or infinity was observed

The exact source corpus can be regenerated with `src/prepare_corpus.py`; raw
Pile text and activation NPZ files are intentionally not committed.

## Runtime

| Stage | Wall time |
| --- | ---: |
| Engineering smoke, 100 examples | 358.08 s |
| Formal corpus construction | 7.47 s |
| G0-A extraction and analysis | 44.89 s |
| G0-B extraction and analysis | 52.21 s |

The formal checkpoint extractions were distributed independently over four
GPUs. Each checkpoint used the same corpus and experiment settings as
`run_pilot.sh`; the standard analysis scripts were run after all checkpoint
files passed metadata checks.

## G0-A: behavior premise

All values are bits per byte. Confidence intervals are 95% cluster-bootstrap
intervals over text examples.

| Pair | Raw KL | Raw CI | Robust KL | Robust CI |
| --- | ---: | ---: | ---: | ---: |
| 2k -> 3k | 0.549120 | [0.460571, 0.674881] | 0.367483 | [0.335198, 0.400062] |
| 5k -> 6k | 0.099157 | [0.084288, 0.116647] | 0.090732 | [0.079067, 0.110439] |
| 10k -> 11k | 0.073646 | [0.066463, 0.081059] | 0.069836 | [0.062339, 0.077109] |
| 20k -> 21k | 0.070549 | [0.061891, 0.080842] | 0.067124 | [0.056668, 0.076967] |
| 50k -> 51k | 0.055134 | [0.048378, 0.063741] | 0.053024 | [0.046290, 0.061840] |
| 100k -> 101k | 0.024534 | [0.022256, 0.027390] | 0.025927 | [0.023527, 0.028254] |
| 142k -> 143k | 0.020785 | [0.018876, 0.022948] | 0.019723 | [0.017714, 0.021789] |

- Raw late/early ratio: `0.03785` (`26.4x` decay)
- Robust late/early ratio: `0.05367` (`18.6x` decay)
- Raw and robust trajectories agree qualitatively.
- The largest single-example contribution to the early raw metric was 8.95%.
  The robust result retains a large decay after clipping and trimming, so the
  trajectory is not explained by a few outliers.

Result: **G0-A PASS**.

## G0-B: representation screen

`1 - CKA` intervals are obtained by reversing the reported projected-CKA
interval endpoints. CKA is a control and uses the protocol's smaller
20-replicate bootstrap.

| Pair | Cosine drift (95% CI) | Standardized drift (95% CI) | 1 - CKA (95% CI) |
| --- | ---: | ---: | ---: |
| 2k -> 3k | 0.190127 [0.187834, 0.192341] | 0.485942 [0.467917, 0.506163] | 0.572874 [0.502552, 0.619647] |
| 5k -> 6k | 0.084738 [0.083588, 0.086037] | 0.190453 [0.184459, 0.197417] | 0.199285 [0.140159, 0.316711] |
| 10k -> 11k | 0.057721 [0.057059, 0.058514] | 0.142005 [0.139817, 0.144693] | 0.048953 [0.011944, 0.094874] |
| 20k -> 21k | 0.040362 [0.039816, 0.041015] | 0.122768 [0.119375, 0.126800] | 0.085005 [0.030605, 0.146482] |
| 50k -> 51k | 0.027450 [0.026938, 0.028037] | 0.094393 [0.091267, 0.099439] | 0.066595 [0.005293, 0.097071] |
| 100k -> 101k | 0.010789 [0.010458, 0.011224] | 0.033436 [0.031977, 0.035560] | 0.014904 [0.001896, 0.037807] |
| 142k -> 143k | 0.005740 [0.005631, 0.005841] | 0.016658 [0.016365, 0.016960] | 0.000212 [0.000030, 0.000480] |

| Metric | Late / early |
| --- | ---: |
| Raw behavior KL | 0.03785 |
| Robust behavior KL | 0.05367 |
| Cosine drift | 0.03019 |
| Standardized drift | 0.03428 |
| CKA movement | 0.000370 |

As a corpus-stability check, the ratios were recomputed for 30 deterministic
random half-samples of 500 texts. Median ratios and their empirical 2.5%-97.5%
ranges were:

| Metric | Median | Empirical range |
| --- | ---: | ---: |
| Robust behavior | 0.05244 | [0.04816, 0.06066] |
| Cosine | 0.03024 | [0.02966, 0.03067] |
| Standardized | 0.03437 | [0.03348, 0.03510] |

In zero of the 30 half-samples did cosine or standardized drift retain more
early-to-late movement than robust behavior.

Result: **G0-B FAIL**. This is Result C from the validation protocol:
representations stabilize alongside behavior, with no meaningful late-training
decoupling to justify sparse-feature or crosscoder work.

## Included artifacts

The committed `artifacts/analysis` directory contains the primary CSV/JSON
outputs and plots. `artifacts/checkpoints` contains metadata JSON only, which
records the shared corpus hash and extraction settings. Large NPZ activation
and log-likelihood arrays remain ignored.
