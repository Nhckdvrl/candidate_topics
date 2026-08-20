# Topic 04 — Confidence and Error Correction

**Status: G-1v1 MEASUREMENT FAILURE → ONE LOCKED G-1v2 REPAIR**

> **If two learners are equally far from the correct answer, does being strongly committed to one wrong answer make corrective learning easier or harder?**

The natural scientific question remains alive, but **G0 has not been run**. The first measurement design failed before corrective training.

## What G-1v1 actually found

On `Qwen/Qwen2.5-1.5B-Instruct` × MMLU-Pro exact-K=10:

```text
scored items                 9981
v1 stable initially-wrong     716
high pool                     215
low pool                      215
matched pairs                  61
mean |Δ p(correct)|          0.00547
mean commitment separation   0.2529
```

The surviving 61 pairs were well matched, so the main problem was **not** the Hungarian matcher. The bottleneck was the v1 operationalization itself.

Two structural problems were identified before any correction outcome existed:

1. **The 8/10 top-wrong stability gate was treatment-dependent.**  
   Truly diffuse wrong distributions should naturally have unstable argmax identities, so requiring a stable top-wrong mechanically removed the low-commitment end of the construct.

2. **Arithmetic averaging over option permutations confounded semantic diffuseness with position susceptibility.**  
   A model can be extremely confident on every permutation yet switch *which semantic option* receives that confidence when labels move. Arithmetic averaging makes this look like a diffuse semantic belief.

Therefore the current result is:

> **G-1v1 failed to measure semantic wrong commitment cleanly enough for a paired correction experiment.**

It is **not** evidence that wrong commitment has no effect on correction.

## The one allowed repair: G-1v2

G-1v2 changes the measurement for a specific mathematical reason, not because a G0 effect was disappointing.

For each semantic option `j` and balanced permutation `r`, we already have mapped conditional probability `p[r,j]`.

The primary semantic score is now

\[
s_j = \frac{1}{R}\sum_r \log(p_{r,j}+\epsilon)
\]

followed by

\[
p^{debias} = softmax(s).
\]

Equivalently this is a normalized geometric mean.

Under the additive nuisance model

\[
z_{r,j} = \alpha_j + \beta_{\text{position}(r,j)},
\]

a complete balanced permutation family gives every semantic option every position exactly once. The mean position term is then constant across semantic choices and cancels after the final softmax.

The old arithmetic mean does **not** have this cancellation property because softmax is nonlinear.

## Commitment and position susceptibility are now separate axes

Primary semantic commitment remains

\[
a = p^{debias}(y^*)
\]

\[
q_j=\frac{p^{debias}(y_j)}{1-a},\quad j\ne y^*
\]

\[
c_{\max}=\max_j q_j.
\]

But permutation sensitivity is measured separately as

\[
S_{pos}=\frac1R\sum_r JS(p_r\;\|\;p^{debias}).
\]

`top_wrong_stability` is retained only as a diagnostic. It is **not** an inclusion criterion.

This distinction matters:

- low `c_max`, low `S_pos`: genuinely diffuse semantic uncertainty;
- high `c_max`, low `S_pos`: stable concentrated misconception;
- high/variable per-permutation confidence, high `S_pos`: position-driven response geometry rather than clean semantic commitment.

## Zero-GPU first step

The v1 repository already stores all 10 mapped permutation distributions per item. Therefore the first v2 test does not require new inference:

```bash
python code/reaggregate_g1v2.py \
  --input results/g1/base_scores.jsonl \
  --output results/g1v2/base_scores_reaggregated.jsonl

python code/build_matched_pairs.py \
  --input results/g1v2/base_scores_reaggregated.jsonl \
  --pairs-output results/g1v2/matched_pairs.jsonl \
  --eligible-output results/g1v2/eligible_wrong.jsonl \
  --report-output results/g1v2/matching_report.json \
  --require-k 10 \
  --p-caliper 0.02 \
  --high-quantile 0.70 \
  --low-quantile 0.30 \
  --discovery-fraction 0.70 \
  --seed 20260821
```

Do **not** run G0 from these pairs yet. This only tests common support and dynamic range.

## Reliability audit before G0

If the zero-GPU reaggregation yields enough pairs, run a deterministic 20% audit subset under:

1. primary prompt + `cyclic` balanced family A;
2. primary prompt + `hashed_cyclic` balanced family B;
3. alternate prompt + `cyclic`.

For both family and prompt audits, predeclared pass criteria are:

```text
Spearman c_max       >= 0.70
Spearman p(correct)  >= 0.90
median semantic JS   <= 0.05
```

Exact top-wrong agreement is diagnostic only.

The v2 scorer also records:

```text
mean_label_mass
min_label_mass
greedy_is_allowed_label_rate
```

so a sharp conditional A-J distribution is not silently treated as meaningful when the model is not actually using the answer-letter response channel.

## Hard decision rule

This is the **only** repair allowed for Topic 04.

After G-1v2:

- if `<200` matched pairs remain, **KILL Topic 04**;
- if balanced-family or prompt reliability fails, **KILL Topic 04**;
- if response-channel diagnostics show the label probabilities are not a valid answer distribution, **KILL / redesign as a new topic**;
- if 1.5B is ambiguous, the already-predeclared `Qwen2.5-3B-Instruct` replication may be run once under the same v2 protocol;
- do not move to 7B/14B, free response, hidden states, a new confidence metric, or a looser stability threshold as a rescue.

Only after v2 measurement passes do we return to the already-implemented G0 correction experiment.

## Why the underlying question remains worth one repair

Human hypercorrection research asks whether confidently held errors are easier to correct because feedback is surprising, or whether apparent confidence effects are actually driven by prior/partial knowledge.

The proposed AI experiment still offers a useful controlled decomposition:

- correct-target accessibility;
- concentration on a particular wrong hypothesis;
- corrective-learning dynamics.

But that decomposition is only scientifically meaningful if the first two can be measured separately and reliably.

## Repository layout

```text
04_confidence_error_correction/
├── README.md
├── VALIDATION.md
├── SERVER_RUNBOOK.md
├── MEASUREMENT_REPAIR.md
├── requirements.txt
├── tests/
│   └── test_g1v2_math.py
└── code/
    ├── prepare_candidates.py
    ├── mcq_utils.py
    ├── score_mcq.py
    ├── reaggregate_g1v2.py
    ├── merge_jsonl.py
    ├── audit_prompt_robustness.py
    ├── build_matched_pairs.py
    ├── build_sft_data.py
    ├── train_correction.py
    ├── evaluate_checkpoints.py
    └── analyze_correction.py
```
