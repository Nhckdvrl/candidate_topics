# 12 — Does Functional Necessity Predict Causal RL Adaptation Leverage?

**Status:** VALIDATION IMPLEMENTED — RUN G-0 BEFORE ANY MECHANISM WORK

## Question

> Are the transformer layers that are most necessary for mathematical reasoning
> also the layers where an RL update can most efficiently improve reasoning?

The two quantities are deliberately causal but different:

- **functional necessity** `I_l`: paired accuracy loss when decoder layer `l` is
  bypassed at inference;
- **RL adaptation leverage** `C_l`: fraction of full-parameter RL gain recovered
  when **only** layer `l` is trainable.

We do **not** use weight-change magnitude as a proxy for learning, and we do not
claim that reasoning literally "lives" in a layer.

## Why Qwen3-1.7B-Base is the locked G-0

The 2026 single-layer-RL paper provides a **full 28-layer scan** for
`Qwen/Qwen3-1.7B-Base`, not just selected layers. Its Appendix Table 13 also
publishes the per-layer MATH500 and GSM8K scores, so we can compare our
necessity intervention against **task-matched** RL leverage without retraining 28
RL models.

The RL paper's primary Qwen3 setup uses GRPO on NuminaMath-CoT; it defines

`C(l) = (S_l - S_base) / (S_full - S_base)`.

For the primary comparison we apply this exact formula to the published
MATH500+GSM8K columns, matching the task support of our necessity curve.
The paper's published four-task `C_math` is a locked robustness check. Table 13
also reports layer 10 at `C_math=1.14` and layer 24 at `C_math=0.28`. The exact
values are frozen in `data/qwen3_1p7b_table13_math.csv`.

This choice is important: G-0 should spend compute on the missing quantity
`I_l`, not needlessly reproduce an already published 28-run RL sweep.

## Primary intervention

For each layer, let the ordinary residual block output be

`h_out = h_in + delta_l`.

The runner replaces only the visible residual-stream output with

`h'_out = h_in + alpha * delta_l`.

Primary G-0 locks `alpha=0`, i.e. a complete decoder-block bypass. The original
layer still executes internally so Hugging Face can maintain a valid KV cache;
downstream layers only see the bypassed residual stream. This is dramatically
faster than disabling KV caching for autoregressive generation.

The **only** predeclared intervention confirmation is `alpha=0.5`, a milder
half-residual perturbation. It is not run to rescue a null result. It is run only
after a large interpretable G-0 to check that the ordering is not created solely
by catastrophic deletion.

## G-0 protocol

Locked before seeing results:

- model: `Qwen/Qwen3-1.7B-Base`;
- layers: **all 28**, no cherry-picking;
- tasks: MATH500 + GSM8K;
- 128 examples per task chosen by stable SHA-256 sampling with seed `20260822`;
- same exact examples for baseline and every layer;
- deterministic greedy decoding;
- fixed prompt asking for step-by-step reasoning and a final `\boxed{}`;
- Math-Verify 0.8+ grading;
- full block bypass (`alpha=0`);
- equal weight for MATH500 and GSM8K when constructing `I_l`;
- primary external target: Table-13 contribution recomputed on the exact matched MATH500+GSM8K task pair;
- locked robustness target: the paper’s published four-task `C_math`;
- primary statistic: Spearman `rho(I, C_matched[MATH500+GSM8K])`;
- paired item bootstrap, 2,000 replicates, 90% CI;
- descriptive task-matched correlations against MATH500-specific and
  GSM8K-specific RL leverage;
- top-5 overlap, because the RL paper itself uses top-5 layer groups.

There is no search over layer subsets, task weights, thresholds, or alternate
correlation metrics.

## A necessary extra check: broad depth shape vs exact layer relation

Both curves might simply be "middle layers are special." A high raw correlation
would then sound stronger than it is.

So the analysis predeclares two non-rescue diagnostics:

1. **depth-residual rho**: fit a quadratic function of normalized depth to each
   curve separately, then correlate the residuals;
2. **circular-shift null**: circularly shift one full curve along depth and
   recompute `|rho|`, preserving smooth depth autocorrelation.

If raw rho is high but the depth-residual relation vanishes, the code reports
`BROAD_DEPTH_ALIGNMENT_ONLY`, not a layer-specific architectural law.

## Measurement-integrity gates

`check_integrity.py` refuses to analyze:

- missing layers;
- duplicate or different example ledgers across conditions;
- >5% Math-Verify fallback on the **unablated baseline**;
- >10% max-token truncation on the **unablated baseline**;
- >15 percentage-point mismatch between our unablated task accuracy and the published Table-13 base score.

Ablation-induced parser failure or overlong generation is **not filtered out**:
destroying answer format or termination can be a real consequence of deleting a
critical block. Those rates are recorded per layer in `integrity_report.json`
and flagged for interpretation. This prevents a strong causal effect from being
silently censored while still revealing when "reasoning necessity" is actually
broad generation damage.

The audit also warns if baseline accuracy lies outside `[0.10, 0.95]`, because
an almost-always-wrong or almost-always-correct baseline has too little paired
variance for a useful accuracy-drop necessity measure. This is a measurement
prerequisite, not a scientific-result threshold.

The final analysis additionally reports, for every layer, baseline-correct→wrong
and baseline-wrong→correct transition rates plus parser/truncation rates. It also
reports `rho(necessity, parser_failure)` and `rho(necessity, truncation)`. These
are interpretation diagnostics, not post-hoc filters: if the apparent
"necessity" curve is mostly a format/termination-collapse curve, we call it
broad generation fragility rather than overselling it as reasoning-specific.

## Predeclared result labels

These labels are deliberately fixed in code (`src/topic12/stats.py`).

### `STRONG_LAYER_LEVEL_ALIGNMENT`

`rho >= 0.50`, bootstrap 90% lower bound `>= 0.20`, and depth-residual
`rho >= 0.25`.

Interpretation: functionally necessary layers are also unusually high-leverage
adaptation targets at finer resolution than a generic middle-layer effect.

### `BROAD_DEPTH_ALIGNMENT_ONLY`

Strong raw positive relation, but depth-residual `rho < 0.25`.

Interpretation: both quantities share a broad depth profile, but we do not yet
have evidence that necessity predicts leverage among neighboring layers.
Scientifically weaker; usually stop unless the raw shape itself is exceptionally
sharp and independently replicates.

### `STRONG_NEGATIVE_RELATION`

`rho <= -0.50` with bootstrap 90% upper bound `<= -0.20`.

This would be especially interesting: indispensable computation is least
plastic under isolated RL adaptation.

### `CREDIBLE_DISSOCIATION`

`|rho| <= 0.20` and the 90% CI is contained in `[-0.35, 0.35]`.

This is an equivalence-style condition, not "p > .05 therefore no relation."

### `INCONCLUSIVE_DO_NOT_TUNE`

Everything else. **Stop.** Do not search new layer subsets, change task weights,
or try ten ablation definitions until one looks good.

## Fastest execution

Install into an existing CUDA/PyTorch environment:

```bash
cd 12_reasoning_necessity_vs_rl_leverage
pip install -r requirements.txt
pytest -q
```

Tiny end-to-end smoke test:

```bash
bash scripts/smoke_test.sh
```

Locked four-GPU G-0:

```bash
bash scripts/launch_g0_4gpu.sh
```

The launcher:

1. evaluates the unablated baseline once;
2. loads one 1.7B model per GPU and assigns layers by `layer % 4`;
3. resumes completed `layer_XX.jsonl` files automatically after interruption;
4. audits ledger/grader/truncation integrity;
5. writes `REPORT.md`, `relation_metrics.json`, `layer_relation.csv`, and
   `relation.png`.

No inter-node communication is required. Four GPUs are used as four independent
inference workers, which is exactly the right way to exploit a slow
cross-node/network environment.

## Confirmation, only after a strong G-0

```bash
bash scripts/launch_g1_confirmation_4gpu.sh
```

This keeps the model, tasks, ledger rule, statistic, and complete layer sweep
fixed, while changing only `alpha` from `0.0` to `0.5` and increasing the sample
to 256/task.

A serious paper claim should ultimately replicate on an independent model
family, but **do not pay for that before G-0 survives**.

## Output interpretation

The scientifically important possibilities are:

- **aligned**: the pretrained computation core is also the easiest place for RL
  to improve that computation;
- **dissociated**: computation and adaptation leverage are different structural
  properties of the network;
- **negative**: the most indispensable computation is comparatively resistant
  to isolated adaptation.

The result is not interesting if it reduces to "middle layers are generally
important" with no finer relation, or if the full 28-layer curve is noisy and
requires post-hoc subset selection.

## References / provenance

- Aadim Nepal et al., *Layer Importance for Mathematical Reasoning is Forged in
  Pre-Training and Invariant after Post-Training*, 2025. Their released code
  uses inference-time layer ablation on math tasks.
- Zijian Zhang et al., *Is One Layer Enough? Training A Single Transformer
  Layer Can Match Full-Parameter RL Training*, arXiv:2607.01232v2, 2026.
  Qwen3-1.7B-Base has a full 28-layer GRPO scan; Table 13 is transcribed under
  `data/`.
- Math-Verify is used for symbolic/numeric answer grading.

## Files

- `scripts/run_ablation.py` — resumable deterministic generation for baseline
  and arbitrary layer shards;
- `scripts/launch_g0_4gpu.sh` — full G-0 on four GPUs;
- `scripts/check_integrity.py` — fails closed on invalid measurements;
- `scripts/analyze_relation.py` — locked statistics, bootstrap, plot, verdict;
- `scripts/launch_g1_confirmation_4gpu.sh` — milder predeclared confirmation;
- `src/topic12/ablation.py` — cache-safe residual-block intervention;
- `src/topic12/benchmarks.py` — deterministic task ledgers + Math-Verify grader;
- `src/topic12/stats.py` — correlation, depth-control, bootstrap, gate logic;
- `data/qwen3_1p7b_table13_math.csv` — immutable published RL curve;
- `tests/` — intervention and statistics unit tests.
