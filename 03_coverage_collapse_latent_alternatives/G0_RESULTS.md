# G0-A Experimental Results

Run ID: `g0_flashinfer2_20260821_233956`  
Run date: 2026-08-21 (Asia/Tokyo)  
Branch: `topic03-counterfactual-audit`

## Scope

This is the paper-exact behavior premise gate for Topic 03. The run used the trained Qwen2.5-0.5B SFT trajectory and evaluated 200 held-out behavior-gate problems with 16 samples at checkpoints e01, e02, e04, and e16.

The paper-exact SFT run completed 16 epochs / 3200 steps with learning rate `2e-5`, batch size `32`, gradient accumulation `1`, and warmup ratio `0.1`.

## Results

| checkpoint | problems | parse rate | mean viable-first probability | pass@1 | pass@8 | pass@16 |
|---|---:|---:|---:|---:|---:|---:|
| e01 | 200 | 1.000 | 1.000 | 0.2931 | 0.5406 | 0.680 |
| e02 | 200 | 1.000 | 1.000 | 0.2403 | 0.7763 | 0.915 |
| e04 | 200 | 1.000 | 1.000 | 0.2988 | 0.8504 | 0.965 |
| e16 | 200 | 1.000 | 1.000 | 0.3503 | 0.9011 | 0.965 |

The preregistered reference comparison is e04 versus e16:

- paired pass@8 change (reference minus late): `-0.0507`
- 95% bootstrap interval: `[-0.0880, -0.0129]`
- first-fork entropy change: `0.0000`
- parse rate at both endpoints: `1.000`

## Gate decision

**`stop_or_redesign`**

The behavior premise does not pass: sampled coverage increases rather than decreases from e04 to e16, and first-fork entropy does not decrease. Per the protocol, G0-B latent probing was not run. This result should not be reframed as evidence for latent viability after coverage collapse.

## Reproducibility artifacts

The tracked result summary is in `results/G0_METRICS.csv` and `results/G0_GATE.json`. The full raw generations and per-sample outputs remain in the local run directory because the upstream `reasoning_forks` checkout is a separate ignored repository:

`external/reasoning_forks/inference_runs/topic03/g0_flashinfer2_20260821_233956/`

The authoritative local machine-readable decision is `artifacts/behavior/g0_flashinfer2_20260821_233956/gate.json`.
