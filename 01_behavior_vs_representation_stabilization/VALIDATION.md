# Validation protocol — Topic 01

This document is the **technical contract** for the G0 experiment. The goal is to decide topic viability quickly without letting an attractive plot survive a bad measurement.

---

## 1. Scientific hypothesis

The broad hypothesis is not sufficient:

> representations change after behavior stabilizes.

That is too close to PolyPythias and existing training-dynamics work.

The hypothesis worth testing is:

> **Global output-distribution motion can enter a low regime while non-trivial residual / sparse-feature reorganization continues.**

G0 tests the residual-geometry premise. G1, if justified, tests sparse features.

---

## 2. Reference work and what is reused

### Kishino et al. — behavior geometry

Reference:

- *Establishing a Scale for KL Divergence in Language Models Across Various Settings*, Findings ACL 2026
- <https://aclanthology.org/2026.findings-acl.1163/>
- repository: <https://github.com/shimo-lab/modelmap/tree/main/kl-scale>

As checked in August 2026, the public `kl-scale` directory exposes figures / documentation but not a complete executable experiment pipeline. Therefore this repository implements the published equations directly.

For checkpoint `t`, let

```text
l_t = [log p_t(x_1), ..., log p_t(x_N)]
```

Stack all checkpoints into `L`, then double-center:

```text
Q = L - row_mean(L) - col_mean(L) + global_mean(L)
```

For nearby checkpoints:

```text
2 KL(p_i, p_j) ≈ ||q_i - q_j||² / N
```

The code reports the byte-normalized local proxy:

```text
||q_i - q_j||² / (2 N mean_bytes ln 2)
```

in bits/byte.

### Crosscoding Through Time — representation hook / later G1

Reference:

- <https://github.com/bayazitdeniz/crosscoding-through-time>

Their Pythia code uses an explicit residual-stream hook such as:

```text
blocks.9.hook_resid_pre
```

and trains crosscoders on selected checkpoint pairs/triplets rather than sweeping every checkpoint. We follow the same principle:

- G0 uses one explicit `resid_pre` block input;
- G1, only if needed, uses a few selected checkpoint snapshots.

### PolyPythias / Evolution of Concepts — novelty boundary

These works already establish that representations and learned features evolve through pretraining. Therefore the following outcomes are **not enough** for a paper:

- hidden states are numerically different late in training;
- CKA is not exactly 1;
- benchmark saturation occurs before some representation metric reaches its floor.

The paper-level target requires continued **feature-level** learning after low global output movement, ideally with local functional consequences.

---

## 3. Corpus construction

Script:

```text
src/prepare_corpus.py
```

Default source:

```text
NeelNanda/pile-10k
```

Procedure:

1. iterate source documents;
2. split raw UTF-8 bytes into 1,024-byte windows;
3. decode each window safely with invalid boundary bytes ignored;
4. drop windows shorter than 256 valid UTF-8 bytes;
5. reservoir-sample exactly 1,000 chunks with seed 42;
6. sort the selected chunks by source row / chunk index;
7. assign stable `example_id` values;
8. store exact UTF-8 byte length;
9. hash the full JSONL corpus.

The same corpus hash must be used at every checkpoint.

This is a **fast approximation** to the reference paper's Pile chunk sampling, not an exact corpus reproduction. The purpose of G0-A is to recover the qualitative fixed-horizon behavior trajectory, not reproduce a published decimal value.

### No silent token truncation

`extract_checkpoint.py` tokenizes with:

```text
truncation=False
```

and raises an error if an input exceeds `--max-tokens`.

If the corpus does not fit, fix the corpus. Do not silently change the effective text and then normalize by the wrong byte count.

---

## 4. Fixed-horizon checkpoint design

All comparisons use the same horizon:

```text
Δ = 1,000 Pythia training steps
```

Pairs:

```text
2k   → 3k
5k   → 6k
10k  → 11k
20k  → 21k
50k  → 51k
100k → 101k
142k → 143k
```

Reasons:

- avoids the learning-rate warmup region near the very beginning;
- covers early, transition, middle, late, and terminal training;
- keeps elapsed training identical across all pairwise movement measurements;
- costs only 14 checkpoint forwards per gate.

Do not replace these with logarithmically spaced adjacent entries unless the reported metric is explicitly divided by / modeled as a function of the interval length.

---

## 5. G0-A — behavior premise

### Extraction

Run:

```bash
GATE=behavior ./run_pilot.sh
```

For every checkpoint, save only:

- sequence log-likelihood per example;
- token length;
- exact corpus byte length / IDs.

No hidden states are extracted in G0-A.

### Raw metric

Compute the double-centered local KL proxy on all 1,000 chunks.

### Robust sensitivity metric

Squared movement is sensitive to pathological texts. The robust version therefore:

1. clips each checkpoint's lower 2% log-likelihood tail;
2. computes, for every example, its maximum absolute log-likelihood change over the seven fixed pairs;
3. removes the top 3% by this score;
4. recomputes double-centering and the KL proxy on retained examples.

We report **both raw and robust curves**. Robust preprocessing is a sensitivity analysis, not a way to hide an inconvenient raw result.

### Bootstrap

Bootstrap unit:

```text
text chunk / example_id
```

not token.

For each bootstrap sample:

1. resample examples with replacement;
2. rebuild the double-centered matrix on the sampled columns;
3. recompute every pair's KL proxy.

Default:

```text
500 bootstrap replicates
```

### G0-A pass condition

We do not hard-code a p-value threshold.

The premise is credible when:

- late fixed-horizon movement is clearly below early movement by a non-trivial effect size;
- raw and robust trajectories agree qualitatively;
- CIs are sufficiently narrow that the conclusion is not sample-noise dominated;
- outlier contribution inspection does not show a tiny number of chunks driving the curve.

If this does not hold, stop. Representation analysis cannot rescue a failed behavior premise.

---

## 6. G0-B — representation screen

Run only after G0-A:

```bash
GATE=representation ./run_pilot.sh
```

### Hook point

Default:

```text
middle GPT-NeoX transformer block input = resid_pre
```

Implementation:

```text
model.gpt_neox.layers[layer_idx].register_forward_pre_hook(...)
```

This makes the residual-stream location explicit and aligns with the hook style used by Crosscoding Through Time.

### Token positions

Default:

```text
4 deterministic interior quantile positions per text
```

Positions are recorded and must match exactly across checkpoints.

The statistical unit remains the text, not the individual token position.

### Metric 1 — matched cosine drift

For matched state `i`:

```text
d_i = 1 - cos(h_t,i, h_t+Δ,i)
```

Average positions within each example before bootstrap inference.

This metric answers:

> does the same residual state directly move in the same coordinate system?

### Metric 2 — pooled-standardized matched drift

Pool the pair to compute per-dimension mean / standard deviation, standardize both sides, then compute mean squared matched displacement.

This reduces domination by high-variance dimensions and gives a dimensionless movement measure.

### Metric 3 — projected linear CKA

CKA is retained as a control because it is tolerant to orthogonal rotations and isotropic rescaling.

To keep G0 cheap:

1. average hidden states per text;
2. apply one fixed Gaussian random projection to 128 dimensions;
3. compute linear CKA;
4. use only a small CKA bootstrap (20 replicates by default).

We do **not** make paper claims from projected CKA. It is diagnostic only.

### How to interpret disagreement between metrics

```text
cosine/std drift high + CKA near 1
```

suggests large coordinate/basis movement with preserved global geometry. This is weak evidence for the scientific question.

```text
cosine/std drift remain elevated while behavior KL is already low
```

is more interesting and justifies G1.

If all representation metrics fall with behavior, stop the topic.

---

## 7. Runtime design

The extraction code is optimized for a falsification run, not general-purpose logging.

Key choices:

- one selected residual hook rather than storing all hidden layers;
- `use_cache=False` because KV cache is unused;
- no allocator cache flushing inside the batch loop;
- default batch size 16, adjustable by environment variable;
- float16 storage for hidden vectors;
- uncompressed NPZ for small activation artifacts;
- behavior and representation gates are separate so a failed G0-A does not waste activation extraction compute.

For Pythia-410M on a large-memory GPU, batch size can usually be increased after a 100-example smoke run.

---

## 8. Required outputs

### G0-A

```text
artifacts/analysis/behavior_metrics.csv
artifacts/analysis/behavior_outliers.csv
artifacts/analysis/behavior_constant_horizon.png
artifacts/analysis/behavior_summary.json
```

### G0-B

```text
artifacts/analysis/representation_metrics.csv
artifacts/analysis/representation_constant_horizon.png
artifacts/analysis/behavior_vs_representation.csv
artifacts/analysis/behavior_vs_representation.png
artifacts/analysis/representation_summary.json
```

All plots are convenience views. CSVs and confidence intervals are the primary artifacts.

---

## 9. Decision matrix

| Observation | Interpretation | Action |
| --- | --- | --- |
| G0-A behavior premise fails | measurement/corpus mismatch or seed phenomenon not robust here | stop and debug |
| robust + raw behavior stabilize; all repr metrics stabilize similarly | no useful decoupling | kill topic |
| behavior stabilizes; only direct drift stays high; CKA remains near 1 | likely coordinate/basis drift | usually kill topic |
| behavior stabilizes; matched + standardized drift remain clearly elevated | candidate decoupling | run G1 |
| G1 crosscoder finds late sparse-feature turnover | real feature-level phenomenon | expand carefully |
| late features explain local/task-functional changes hidden by global KL | strong paper-level story | invest |

---

## 10. G1 — only after positive G0

Do not run a huge snapshot grid.

Select three checkpoints from the G0 trajectory:

```text
before stabilization
near stabilization
late stable regime
```

Then adapt the public Crosscoding Through Time pipeline:

- fixed residual hook;
- small checkpoint pair/triplet;
- sparse crosscoder;
- feature emergence / maintenance / discontinuation;
- RelIE / ablation only after a clear feature-level signal exists.

The key G1 question is:

> **Are there late features whose life cycle continues after global output movement is already low?**

The stronger G2 question is:

> **What local functional changes do those apparently behaviorally silent features encode?**

If G1 finds no such feature turnover, stop. Do not write a CKA paper.

---

## 11. Reproducibility checklist

Before trusting any run:

- [ ] same corpus SHA at every checkpoint;
- [ ] all seven pairs use exactly `Δ=1000`;
- [ ] no tokenizer truncation warning/error was bypassed;
- [ ] example IDs and byte lengths match across checkpoints;
- [ ] raw and robust KL both inspected;
- [ ] `behavior_outliers.csv` inspected for concentration;
- [ ] representation token positions match exactly across checkpoints;
- [ ] bootstrap unit is text, not token;
- [ ] CKA is treated as a control, not the only falsifier;
- [ ] G0-A was passed before interpreting G0-B;
- [ ] no full crosscoder sweep is started before G0-B gives a meaningful effect-size separation.
