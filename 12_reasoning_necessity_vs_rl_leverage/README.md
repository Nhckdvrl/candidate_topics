# 12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?

**Status:** VALIDATION HARDENED — RUN THE LOCKED G-0 BEFORE MECHANISM WORK

## Scientific question

> Are the transformer layers that are required for mathematical competence also
> the layers in which isolated RL updates most efficiently improve that competence?

The project compares two different causal quantities on the **same Qwen3-1.7B
base model**:

- **functional necessity** `I_l`: how often competence that the base model
  demonstrably has is destroyed when decoder layer `l` is bypassed;
- **RL adaptation leverage** `C_l`: how much of full-parameter RL gain is
  recovered when only decoder layer `l` is trainable.

This is not a weight-change study. We do not infer learning from `||ΔW||`, and
we do not claim reasoning literally “lives” in a layer.

## Why this experiment can decide the topic cheaply

The 2026 paper *Is One Layer Enough? Training A Single Transformer Layer Can
Match Full-Parameter RL Training* already publishes a complete **28-layer GRPO
scan for Qwen3-1.7B-Base**. Appendix Table 13 gives MATH500 and GSM8K scores for
the base model, full RL model, and every independently trained single-layer
model. Therefore we should not pay to repeat 28 RL runs before knowing whether
the missing relation exists.

For a score `S`, the paper defines

`C_l = (S_l - S_base) / (S_full - S_base)`.

The locked primary `C_l` in this repository applies that exact formula to the
published **MATH500+GSM8K average**, matching the task support of our necessity
measurement. The paper's published four-math-benchmark `C_math` is retained as
a frozen robustness check.

## Why the necessity metric is paired

A naive importance score is

`baseline accuracy - ablated accuracy`.

That is useful descriptively, but it can hide real necessity: a layer deletion
might destroy one problem that the base solves while accidentally changing a
previously wrong answer into a correct answer. The two transitions cancel in
net accuracy even though the first transition is direct evidence that the
existing competence depended on the layer.

The **primary** Topic-12 necessity score is therefore

`I_l,t = P(ablated wrong | baseline correct, task=t)`

and the final curve is the equal-weight average over MATH500 and GSM8K.

The old net accuracy-drop curve is still reported as a locked robustness check.
No result is discarded because it disagrees with the primary metric.

## Locked model / data / prompt contract

G-0 is pinned to:

- model: `Qwen/Qwen3-1.7B-Base`;
- model revision: `912d2727784ca0a6f718845aa14d4d9e5f48fe26`;
- decoder layers: exactly 28;
- MATH500: `HuggingFaceH4/MATH-500` revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`;
- GSM8K: `openai/gsm8k`, `main/test`, revision
  `7cf1290ed87c28a31f867e0f47a7cb62a61d502e`;
- prompt: the Qwen mathematical-reasoning prompt used by the seed
  layer-ablation work (`qwen_math_seed`);
- decoding: greedy, deterministic, KV cache enabled;
- 256 frozen examples per task, selected by SHA-256 over stable example IDs;
- seed: `20260822`;
- primary full-bypass intervention: `residual_scale=0.0`;
- maximum input length: 2048 tokens; **any input truncation fails integrity**;
- maximum generated length: 1536 tokens;
- Math-Verify primary grading.

Every output directory contains `run_contract.json`. Its hash is embedded in
every generated row. A stale result from a different seed, prompt, model
revision, sample size, ablation strength, or token limit is refused rather than
silently “resumed”.

## Intervention

For a decoder block with visible residual-stream mapping

`h_out = h_in + delta_l`,

we expose downstream layers to

`h'_out = h_in + alpha * delta_l`.

The target layer is still executed internally, so Hugging Face can maintain its
KV-cache bookkeeping, but its visible contribution to the downstream residual
stream is modified.

- `alpha=0.0`: exact block bypass, primary G-0;
- `alpha=0.5`: predeclared mild intervention.

The implementation special-cases `alpha=0` and `alpha=1` so the endpoints are
exact in bf16/fp16 rather than merely algebraically equivalent in real
arithmetic. `scripts/preflight_model.py` verifies on the real Qwen checkpoint
that `alpha=1` is exact identity, `alpha=0` actually changes logits, the model
has 28 layers, and cached generation still works under the hook.

## Protocol gate happens before the expensive sweep

The external `C_l` curve is only meaningful if our evaluation protocol is
reasonably compatible with the published base model scores. Table 13 reports:

- MATH500 base: 57.4;
- GSM8K base: 74.4.

The launcher therefore runs the unablated baseline **first** and immediately
calls `check_integrity.py --baseline-only`. If this fails, the 28-layer sweep is
not started.

The baseline gate checks:

- one frozen run contract;
- model/data/prompt ledger consistency;
- no silent input truncation;
- <=5% Math-Verify fallback;
- <=10% output truncation;
- compatibility with each published baseline score.

For published-score compatibility, the allowed absolute gap is

`max(5 percentage points, 2 * binomial SE around the published score)`.

With the locked `n=256`, this is much tighter than the previous arbitrary 15pp
window while still allowing expected subset sampling noise. If this gate fails,
that is **INCONCLUSIVE MEASUREMENT**, not evidence for or against the research
hypothesis. Fix the evaluation protocol before running layers.

## G-0

Run every decoder layer once. There is no layer discovery or selected subset.

Primary output:

`rho = Spearman(I_conditional, C_matched[MATH500+GSM8K])`.

Also frozen before seeing results:

- paired item bootstrap, 2,000 replicates, 90% CI;
- Kendall tau;
- top-5 overlap;
- MATH500-only and GSM8K-only relations;
- correlation between the two task-specific necessity profiles;
- robustness to the old net accuracy-drop definition of `I`;
- robustness to the published four-task `C_math`;
- parser-failure and max-token diagnostics.

### Broad depth shape is not enough

The RL paper already shows a strong middle-layer concentration. If our necessity
curve is also just “middle layers matter”, raw Spearman can look impressive
without establishing that the **same individual layers** are special.

The locked fine-grained check fits a quadratic function of normalized depth to
each raw curve separately and correlates the two residual curves. This asks
whether deviations among neighboring layers line up beyond the broad depth
profile. A circular-shift null is also reported because the curves are smooth
along depth.

A true partial-rank depth statistic is additionally reported as descriptive
diagnostics, but it is not the gate: a quadratic model of ranked values does not
perfectly absorb a nonlinear U-shaped rank profile.

## Do not mistake destructive ablation for a scientific null

Hard deletion is a strong intervention. Layer-induced parser failure or runaway
generation is retained as a causal outcome; those examples are **never filtered
out** after seeing the result. However, if full deletion destroys so much of the
model that layers cannot be ranked meaningfully, the run is not allowed to kill
the topic.

`alpha=0` is automatically labeled `TOO_DESTRUCTIVE_USE_MILD_SWEEP` if either:

- at least 25% of layers destroy >=90% of baseline-solved items; or
- at least 25% of layers have >=50% parser fallback; or
- at least 25% of layers have >=50% output truncation.

That yields `INCONCLUSIVE_INTERVENTION`. The next and only predeclared
measurement is the full `alpha=0.5` sweep.

Crucially, G-0 and the mild sweep now use the **same 256/task ledger, same prompt,
same model revision, same token limits, same statistics**. The only scientific
variable changed is `alpha: 0.0 -> 0.5`.

## Frozen result labels

Operational labels are defined in `src/topic12/stats.py`; they are not tunable
after results arrive.

- `STRONG_LAYER_LEVEL_ALIGNMENT`: large positive rho, bootstrap lower bound
  positive, and a nontrivial relation remains after broad depth removal.
- `BROAD_DEPTH_ALIGNMENT_ONLY`: raw positive alignment is strong but fine
  layer-to-layer alignment disappears after depth-shape removal.
- `STRONG_NEGATIVE_RELATION`: necessity and isolated RL plasticity run in
  opposite directions.
- `DISSOCIATION_CANDIDATE`: near-zero relation with a sufficiently narrow paired
  bootstrap interval. This is a candidate structural dissociation, not yet a
  paper-level law because the published RL curve itself has finite experimental
  uncertainty.
- `INCONCLUSIVE_INTERVENTION`: hard deletion is too destructive; use the locked
  mild sweep rather than interpreting the run scientifically.
- `INCONCLUSIVE_DO_NOT_TUNE`: no clean conclusion. Do not search layer subsets,
  task weights, thresholds, correlations, or new ablation definitions.

The purpose of these labels is to prevent post-hoc storytelling, not to maximize
the probability that the topic survives.

## Fast execution

Use an existing CUDA/PyTorch environment; `requirements.txt` intentionally does
not reinstall PyTorch.

```bash
cd 12_reasoning_necessity_vs_rl_leverage
pip install -r requirements.txt
pytest -q
bash scripts/smoke_test.sh
bash scripts/launch_g0_4gpu.sh
```

The G-0 launcher does:

1. real-model intervention preflight on GPU 0;
2. unablated baseline on the frozen 256+256 ledger;
3. **baseline protocol gate before expensive layer compute**;
4. 28-layer sweep sharded by `layer % 4` across four independent GPUs;
5. complete integrity audit;
6. locked statistics and `REPORT.md`;
7. emits whether the mild predeclared intervention is required.

No cross-GPU or cross-node communication is used. Each GPU loads its own 1.7B
model and handles seven layers, which is appropriate for machines connected by
slow inter-node networking.

Main outputs:

```text
results/g0_qwen3_1p7b_bypass/
├── run_contract.json
├── integrity_report.json
├── baseline.jsonl
├── layer_00.jsonl ... layer_27.jsonl
├── layer_relation.csv
├── relation_metrics.json
├── relation.png
└── REPORT.md
```

If `REPORT.md` says `INCONCLUSIVE_INTERVENTION`, or if a strong G-0 should be
confirmed at lower intervention strength:

```bash
bash scripts/launch_g1_confirmation_4gpu.sh
```

Do not run this simply because an otherwise valid G-0 is an ordinary null.

## What is enough to keep the research topic alive?

A useful G-0 is not “some p-value passed.” We want a whole-depth pattern that is
hard to explain away:

- strong alignment that survives the broad-depth diagnostic; or
- strong negative relation; or
- a narrow, stable dissociation with similar conclusions on both tasks and an
  independently replicated model family later.

If the only story is “both curves peak somewhere in the middle”, the result is
weaker than the proposed architectural principle. If the task-specific necessity
curves disagree strongly, treat the aggregate cautiously. If necessity mostly
tracks parser/truncation collapse, describe it as broad generation fragility,
not mathematical reasoning machinery.

Only after a clean G-0 should we spend compute on an independent model family,
mechanism analysis, or a fresh RL reproduction.

## References

- Aadim Nepal et al., *Layer Importance for Mathematical Reasoning is Forged in
  Pre-Training and Invariant after Post-Training* (2025). The released setup
  motivates exact layer bypass and supplies the locked Qwen math prompt.
- Zijian Zhang et al., *Is One Layer Enough? Training A Single Transformer Layer
  Can Match Full-Parameter RL Training*, arXiv:2607.01232v2 (2026). Appendix
  Table 13 supplies the frozen Qwen3-1.7B per-layer RL evaluation curve.

## Repository layout

```text
12_reasoning_necessity_vs_rl_leverage/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── qwen3_1p7b_table13_math.csv
├── scripts/
│   ├── preflight_model.py
│   ├── run_ablation.py
│   ├── check_integrity.py
│   ├── analyze_relation.py
│   ├── launch_g0_4gpu.sh
│   ├── launch_g1_confirmation_4gpu.sh
│   └── smoke_test.sh
├── src/topic12/
│   ├── __init__.py
│   ├── ablation.py
│   ├── benchmarks.py
│   └── stats.py
└── tests/
    ├── test_ablation.py
    ├── test_benchmarks.py
    ├── test_paper_table.py
    └── test_stats.py
```
