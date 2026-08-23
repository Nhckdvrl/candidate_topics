# Topic 18 frozen G0 result

**Decision: `INCONCLUSIVE / STOP_NO_RESCUE_SWEEP`**

The fully crossed, three-family local run does not support the intended
model-general “intrinsic inhibition bottleneck” claim. It also does not produce
a clean matched null: two families show substantial gaps and one does not.
Under the frozen gate, this mixed result cannot be promoted or relabeled as a
kill.

## Frozen design

- 64 base cells, each expanded to positive, negative, and equal-outcome baseline
  conditions (192 prompts per model);
- eight symbol pairs × marked identity × observation order × answer-option order;
- deterministic decoding, optional thinking mode disabled, maximum 8 new tokens;
- exact cached model revisions recorded in every prediction row;
- panel: Qwen2.5-7B-Instruct, Phi-4-mini-instruct, Gemma-3-12B-it.

## Primary result

| Family / model | Positive accuracy | Negative accuracy | Paired delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Google Gemma / Gemma3-12B | 1.000 | 0.797 | 0.203 | [0.109, 0.297] |
| Microsoft Phi / Phi-4-mini | 0.969 | 0.578 | 0.391 | [0.266, 0.531] |
| Qwen / Qwen2.5-7B | 0.953 | 0.906 | 0.047 | [-0.047, 0.141] |
| **Stimulus-clustered panel** | — | — | **0.214** | **[0.135, 0.292]** |

All outputs were parseable. Every model's equal-outcome baseline selected the
marked action at exactly 0.5 after counterbalancing. Thus the mixed result is not
explained by aggregate label or position preference.

## Why the decision is not `SURVIVE`

The frozen survival rule requires every model and every identity/order stratum
to have delta at least 0.10. Qwen's delta is 0.047 and some strata are below the
bar. A pooled average cannot override a failed model-consistency prerequisite.

## Why the decision is not `KILL`

The pooled upper confidence bound is not below 0.10, and two—not at most one—of
the three independent model families exceed a 0.10 gap. The clean-null kill
condition therefore also fails.

## Interpretation and stop rule

The strongest warranted conclusion is **model-family heterogeneity**. This tiny
G0 does not identify why Phi and Gemma differ from Qwen, and the project contract
forbids a prompt/task/model rescue sweep. Topic 18 therefore stops without a
mechanistic or method-development stage.

Machine-readable artifacts:

- `g0_design_v2.jsonl` — exact frozen prompts and answer keys;
- `g0_predictions_final.jsonl` — all 576 outputs with family/model/revision;
- `G0_LOCAL_RESULTS.json` — full scores, strata, controls, intervals, and gate.
