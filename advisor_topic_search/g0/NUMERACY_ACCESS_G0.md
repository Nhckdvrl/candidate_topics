# Numeracy Access — Frozen Same-Computation G0

## Why this G0 exists

The EACL 2026 seed paper establishes a large gap between linearly decodable numerical-ranking information and verbal comparison performance. However, its headline Table 1 compares:

- a **zero-shot** prompt for probing; and
- a **one-shot** prompt for verbalization.

The paper also shows that some models are sensitive to the answer position in the one-shot demonstration and that few-shot prompting reduces several of those positional biases.

Therefore we must **not** treat the published cross-condition gap itself as proof of an access bottleneck.

The project survives only if the following event exists at useful density in the **same model, same prompt, same decision regime**:

```text
ranking is correctly decodable from the prompt state
BUT
that very prompt's greedy generation answers incorrectly
```

This is the mechanism-level prerequisite.

---

## Seed-exact scope correction

A closer audit of Appendix B/C matters for feasibility:

- the paper constructs both `int-sci` and `dec-sci` variants;
- it states that **dec-sci is used only for Figs. 3, 10, 11, 13 and 14**;
- **int-sci is used for all other cross-notation experiments**, including the headline Table 1 and the finetuning experiment;
- the paper's explicit `k=1..5` few-shot experiment is also run on **int-sci**.

Therefore the frozen project G0 must not require a second, less directly seeded setting to pass.

```text
PRIMARY G0      = int_sci_compare only
CONFIRMATION    = dec_sci_compare only after primary G0 passes
```

This reduces stacked experimental risk while keeping a clean independent extension available later.

---

## Frozen primary model

```text
Qwen/Qwen3-8B
```

Reason: the seed already reports a large gap on this exact accessible model in the primary int-sci experiment:

```text
one-shot verbalization = 70.00%
zero-shot classifier probe = 98.88%
```

so we are not model-fishing.

If G0 passes, use `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` as the first independent model confirmation before a broad mechanism story.

---

## Frozen primary data

Use the official generator and published seed:

```text
seed = 0
int_sci_compare:
  8,000 train
  1,600 validation
  1,600 test
```

The independent static audit in `numeracy_data_audit.py` found no displayed ties or ordering changes for the published seed despite 5-significant-digit scientific formatting.

Do not regenerate the primary test set with another seed.

`dec_sci_compare` is reserved for post-G0 confirmation, not part of the project-survival gate.

For any later independently generated confirmation set, reject exact `a == b`; the released dec-sci generator can produce ties under some seeds even though published seed 0 is clean.

---

## Frozen prompt regime

Primary behavioral regime:

```text
n_few_shot = 5
operator = larger
use_alt_prompt = false
```

Use the **exact five int-sci examples from the seed paper / official `src/verbalization.py`**:

```text
9.9 × 10^2  vs 100
161230       vs 7.182 × 10^5
713          vs 4.78 × 10^2
1.354 × 10^6 vs 4906723
20834        vs 6.5 × 10^3
```

with correct-answer positions alternating `A, B, A, B, A`.

Why five-shot rather than the headline one-shot setting:

- the seed's appendix shows that additional demonstrations reduce several positional artifacts;
- performance largely plateaus around five examples;
- importantly, 7B–8B models still struggle when the two numbers are close (`|log2(a/b)| < 0.1`) under few-shot prompting.

Thus G0 asks whether a representation/behavior dissociation survives after removing the easiest prompt-position explanation.

Do **not** search over prompt templates in discovery.

---

## Frozen hard regime

Primary critical subset:

```text
|log2(a / b)| < 0.1
```

This threshold is inherited from the seed paper, not selected from our outcome.

Published-seed static count:

```text
int-sci test: 129 / 1,600 = 8.0625%
answer-A rate: 46.51%
```

This is large enough to require a real instance-level critical cell rather than a handful of cherry-picked cases.

Report the full 1,600-item test set as secondary context, but do not replace the hard regime if the full-set result looks prettier.

---

## Same-prompt hidden-state extraction

For every train/validation/test item, construct the exact five-shot verbalization prompt and run the model on the complete prompt before generation.

Save the residual-stream hidden state at:

```text
last input token immediately before answer generation
```

for every transformer layer.

No token-position search is allowed in G0.

The greedy generation uses the **same prompt string and same model** and follows the seed's deterministic decoding/scoring convention.

The forward pass used for hidden-state extraction and the generation call may be executed separately for engineering convenience; scientific equivalence requires identical model weights, prompt tokens and deterministic inference configuration.

---

## Probe

Train the same simple logistic ranking probe used by the seed:

```text
y = 1[a > b]
```

on the five-shot prompt states.

Layer selection:

1. train one logistic probe per layer on train;
2. choose the layer with highest validation accuracy;
3. break exact ties by choosing the **earliest** layer;
4. freeze that layer before inspecting test labels;
5. evaluate once on test.

No nonlinear probe, SAE, token sweep, or post-hoc layer rescue in G0.

---

## Primary measurements

Report on the full test set and the frozen hard subset:

```text
A_gen   = greedy generation accuracy
A_probe = frozen-layer probe accuracy
Gap     = A_probe - A_gen

N_critical = count(probe correct AND generation wrong)
R_critical = N_critical / N_hard
E_covered  = N_critical / count(generation wrong)
```

Also report the 2×2 table:

| | generation correct | generation wrong |
|---|---:|---:|
| probe correct | n11 | **n10 critical** |
| probe wrong | n01 | n00 |

The project-level object is `n10`, not merely an aggregate accuracy difference.

---

## Frozen survival gate

### GO

Proceed to causal intervention only if **all** hold on the locked int-sci test set:

1. full-test probe accuracy `>= 0.90`;
2. hard-subset probe accuracy `>= 0.80`;
3. hard-subset `A_probe - A_gen >= 0.15`;
4. hard subset has `N_critical >= 30`;
5. hard-subset gap is positive;
6. invalid/unparseable generation is `< 5%` of the hard subset.

These are minimum-worthy bars rather than significance-only thresholds. With 129 hard items, fewer than 30 critical cases would make a first causal analysis too dependent on a small tail.

### KILL / DOWNGRADE

Stop the access-mechanism project if any of the following occurs:

- balanced five-shot prompting mostly closes the gap;
- the same-prompt probe loses the strong ranking signal;
- `probe-correct / generation-wrong` cases are too sparse;
- invalid output drives the apparent gap;
- a useful effect appears only after changing prompt, layer rule, threshold or model.

A failure here does **not** refute the EACL seed. It means the stronger same-computation causal-access project lacks a sufficiently dense object.

---

## Post-G0 confirmation, not part of survival

Only if primary int-sci G0 passes:

1. reproduce the same same-prompt dissociation on `dec_sci_compare`;
2. confirm on `DeepSeek-R1-Distill-Qwen-7B` or another seed-supported 7B–8B open model;
3. only then invest in broad causal intervention.

The purpose is to avoid making G0 itself a multi-setting obstacle while still preventing a one-dataset mechanism paper.

---

## Positive control

The selected probe must retain high accuracy on the full test distribution. If it fails generally, a null on the hard subset is not interpretable.

---

## What G0 does *not* claim

Even a strong G0 only establishes:

> the exact prompt condition contains easily decodable correct ranking information while the model's own output is wrong.

It does **not** establish that the linear probe direction is the model's native causal channel.

That distinction is reserved for G1.

---

## G1 design constraints if G0 passes

G1 must avoid a free `layer × strength × token` steering sweep.

Preferred causal tests should:

- use the G0-frozen layer/token rule or select one operating point on validation only;
- manipulate the ranking subspace with residual-norm-matched interventions;
- include shuffled-label/random-direction nulls;
- evaluate once on locked test critical cells;
- distinguish **information available** from **information causally used**.

A clean null is scientifically meaningful: if ranking is strongly decodable but calibrated intervention does not affect the decision, the readable representation may be epiphenomenal or not lie on the generation path.

---

## Estimated cost

The seed reports, per model on a single A100:

- hidden-state extraction on the full dataset: a few hours;
- probe training/evaluation: about 40 minutes;
- verbalization on 1,600 examples: about 25 minutes.

Primary G0 now runs only int-sci, so it is cheaper than the previous two-setting draft and comfortably within local GPU resources:

```text
paid API: 0
new annotation: 0
foundation-model training: 0
```

This is currently the strongest feasibility profile in the advisor topic pool.
