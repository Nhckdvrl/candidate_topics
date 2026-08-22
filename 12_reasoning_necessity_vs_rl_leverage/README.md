# 12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?

**Status: ARCHIVED — frozen G-0 ended in `INCONCLUSIVE_DO_NOT_TUNE`.**

See the full [archive summary](./ARCHIVE_SUMMARY.md) and preserved G-0 outputs under [`results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/`](./results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/).

## Scientific question

> Are the transformer layers that are required for mathematical competence also the layers in which isolated RL updates most efficiently improve that competence?

Topic 12 compared two distinct causal layer-wise quantities on the same `Qwen3-1.7B-Base` model:

- **functional necessity** `I_l = P(ablated wrong | baseline correct)` under a complete layer-bypass sweep;
- **RL adaptation leverage** `C_l`, taken from the published complete single-layer GRPO sweep and matched to the same MATH500+GSM8K task support.

The point of the experiment was not merely to ask whether both quantities vary with depth. The project required the **same individual layers** to line up beyond any broad middle-/depth-level structure.

## Frozen G-0

The final locked run used:

```text
model               Qwen/Qwen3-1.7B-Base
revision            912d2727784ca0a6f718845aa14d4d9e5f48fe26
layers              28
tasks               MATH500 + GSM8K
examples/task       256
seed                 20260822
decoding             greedy
max input            2048
max new tokens       1536
residual_scale       0.0
GPUs                 4
engineering batch    128
```

Integrity passed: `28/28` layer outputs, no input truncation, frozen run contract matched, and post-fix Math-Verify fallback was `0%`.

Baseline scores were compatible with the published base model:

```text
GSM8K    78.1%
MATH500  58.6%
```

## Result

```text
Spearman rho(I, C)                         0.355
paired bootstrap 90% CI                  [0.300, 0.402]
Kendall tau                                0.225
rho after removing quadratic depth trend -0.238
partial-rank depth diagnostic             -0.093
circular-shift p                           0.071
top-5 overlap                              1
random top-5 overlap expectation           0.89
MATH500 vs GSM8K necessity-profile rho     0.878
```

The raw relation was moderately positive, but the preregistered fine-grained correspondence disappeared after removing the broad depth trend and even changed sign. The top-5 overlap was essentially random.

At the same time, the necessity profile itself was highly stable across MATH500 and GSM8K (`rho=0.878`). This makes the negative informative: the problem was not simply that the necessity measurement was too noisy. Rather, **a stable functional-necessity structure did not map cleanly onto the published RL-leverage structure at individual-layer resolution.**

The frozen gate therefore returned:

`INCONCLUSIVE_DO_NOT_TUNE`

## Grader repair

A real evaluator bug was found for MATH500 bare gold expressions such as `(-1,6)`. `Math-Verify` could return an empty parse for valid unboxed dataset gold strings.

The fix normalized only the **gold serialization** when the raw gold failed to parse. Model responses were unchanged, frozen responses were regraded rather than regenerated, and a regression test was added. The repair reduced the spurious MATH500 fallback rate from about `8.98%` to `0%` without changing the scientific protocol.

## Why no G-1 / 4B rescue

Full layer bypass caused substantial generation pathology for some individual layers, but the preregistered global destructive-intervention threshold was not crossed; the run was classified `INFORMATIVE`, not `INCONCLUSIVE_INTERVENTION`.

Therefore `alpha=0.5` was **not** run merely because the substantive result was unattractive. Likewise, a second Qwen model was not used to search for a more favorable correlation after a valid no-tune discovery.

The archive decision is intentional:

- no layer-subset search;
- no task reweighting;
- no alternative correlation metric chosen post hoc;
- no milder ablation as a rescue;
- no 4B/model sweep as a lottery ticket.

## Main lesson

> **Two stable layer-wise structures do not imply a meaningful mapping between them.**

Topic 12 is a useful example of why complete-profile comparison plus a broad-depth control should come before mechanism work. A moderate raw correlation can be real while still being too coarse to support the stronger architectural principle a paper would need.

For the complete scientific interpretation, failure analysis, intervention caveats, and reopen conditions, read [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md).

## Preserved code and outputs

The validation harness, tests, frozen published RL table, and optional G-1 launcher are preserved for reproducibility. They are not an invitation to continue tuning this hypothesis.

Key preserved outputs:

- [`REPORT.md`](./results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/REPORT.md)
- [`integrity_report.json`](./results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/integrity_report.json)
- [`relation_metrics.json`](./results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/relation_metrics.json)
- [`layer_relation.csv`](./results/g0_qwen3_1p7b_bypass_gold_normalized_bs128/layer_relation.csv)

## References

- Aadim Nepal et al., *Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training* (2025).
- Zijian Zhang et al., *Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training*, arXiv:2607.01232v2 (2026).