# Numeracy Access — Frozen Same-Computation G0

## Why this G0 exists

The EACL 2026 seed paper establishes a large gap between linearly decodable numerical-ranking information and verbal comparison performance. However, its headline Table 1 compares:

- a **zero-shot** prompt for probing; and
- a **one-shot** prompt for verbalization.

The paper also shows that some models are sensitive to the answer position in the one-shot demonstration and that few-shot prompting reduces several of those positional biases.

Therefore we must **not** treat the published cross-condition gap itself as proof of an access bottleneck.

The project survives only if the following event exists at useful density in the **same model, same prompt, same forward computation**:

```text
ranking is correctly decodable from the prompt state
BUT
that very generation answers the comparison incorrectly
```

This is the mechanism-level prerequisite.

---

## Frozen primary model

```text
Qwen/Qwen3-8B
```

Reason: the seed already reports a large gap on this exact accessible model (70.00% one-shot verbalization vs 98.88% classifier probing), so we are not model-fishing.

If G0 passes, use `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` as the first independent model-family confirmation before a broad mechanism story.

---

## Frozen data

Use the official generator and published seed:

```text
seed = 0
int_sci_compare: 8,000 train / 1,600 val / 1,600 test
dec_sci_compare: 8,000 train / 1,600 val / 1,600 test
```

The independent static audit in `numeracy_data_audit.py` found no displayed ties or ordering changes for the published seed despite 5-significant-digit scientific formatting.

Do not regenerate the primary test set with another seed.

For any later confirmation seed, explicitly reject exact `a == b` because the released dec-sci generator can produce ties under some seeds.

---

## Frozen prompt regime

Primary behavioral regime:

```text
n_few_shot = 5
operator = larger
use_alt_prompt = false
```

Use the **exact five examples already implemented in the official `src/verbalization.py`**.

Why five-shot rather than the headline one-shot setting:

- the seed's appendix shows that additional demonstrations reduce several positional artifacts;
- performance plateaus around five examples;
- crucially, the paper reports that 7B–8B models still struggle around chance when the two numbers are close (`|log2(a/b)| < 0.1`) even under few-shot prompting.

Thus the primary G0 deliberately asks whether a representation/behavior dissociation survives after removing the easiest prompt-bias explanation.

Do **not** search over prompt templates in discovery.

---

## Frozen hard regime

Primary critical subset:

```text
|log2(a / b)| < 0.1
```

This threshold is not selected from our results; it is the hard regime explicitly analyzed by the seed paper.

Published-seed static counts:

```text
int-sci test: 129 / 1,600
  answer A rate: 46.51%

dec-sci test: 137 / 1,600
  answer A rate: 44.53%

combined hard test: 266 items
```

This is large enough for an instance-level mechanism prerequisite rather than a handful of cherry-picked errors.

Report the full test set as secondary context, but do not replace the hard regime if the full-set result looks prettier.

---

## Same-prompt hidden-state extraction

For every train/val/test example, construct the exact five-shot verbalization prompt and perform one forward pass on the complete prompt **before generation**.

Save the residual-stream hidden state at:

```text
last input token immediately before answer generation
```

for every transformer layer.

No token-position search is allowed in G0.

The subsequent greedy generation must start from the same prompt and use the official verbalization decoding/scoring rule.

---

## Probe

Train the same simple logistic ranking probe used by the seed:

```text
y = 1[a > b]
```

on the same five-shot prompt states.

Layer selection:

1. train one logistic probe per layer on train;
2. choose the layer with highest validation accuracy;
3. break exact ties by choosing the **earliest** layer;
4. freeze that one layer before inspecting test labels;
5. evaluate once on test.

No nonlinear probe, SAE, token sweep, or post-hoc layer rescue in G0.

---

## Primary measurements

For each setting separately (`int-sci`, `dec-sci`) and for their pooled hard subset, report:

```text
A_gen   = greedy generation accuracy
A_probe = frozen-layer probe accuracy
Gap     = A_probe - A_gen

N_critical = count(probe correct AND generation wrong)
R_critical = N_critical / N_hard
E_covered  = N_critical / count(generation wrong)
```

Also report a 2×2 table:

| | generation correct | generation wrong |
|---|---:|---:|
| probe correct | n11 | **n10 critical** |
| probe wrong | n01 | n00 |

The project-level object is `n10`, not merely a difference between aggregate accuracies.

---

## Frozen survival gate

### GO

Proceed to causal intervention only if **all** hold on the locked test set:

1. pooled hard-subset probe accuracy `>= 0.80`;
2. pooled hard-subset `A_probe - A_gen >= 0.15`;
3. pooled hard subset has `N_critical >= 60`;
4. each of int-sci and dec-sci independently has `N_critical >= 20`;
5. the sign of the gap is positive in both settings;
6. invalid/unparseable generation is `< 5%` of the hard subset.

These bars are intentionally minimum-worthy rather than significance-only thresholds. With 266 hard test items, a much smaller critical cell would make instance-level causal work fragile.

### KILL / DOWNGRADE

Stop the access-mechanism project if any of the following occurs:

- balanced five-shot prompting mostly closes the gap;
- the same-prompt probe loses the high ranking accuracy seen by the zero-shot seed probe;
- `probe-correct / generation-wrong` cases are too sparse;
- the result exists only in one notation setting;
- a useful effect appears only after changing prompt, layer rule, threshold, or model.

A failure here does **not** refute the seed paper. It means the stronger causal-access project lacks a sufficiently dense same-computation object.

---

## Positive control

The selected probe must also retain high accuracy on the full validation/test distribution. If it fails generally, a null on the hard subset is not interpretable.

---

## What G0 does *not* claim

Even a strong G0 only establishes:

> the exact computation contains easily decodable correct ranking information while the model's own output is wrong.

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

A clean null is scientifically meaningful: if ranking is strongly decodable but targeted intervention does not affect the decision under calibrated controls, that argues the readable representation may be epiphenomenal or not on the generation path.

---

## Estimated cost

The seed paper reports, per model on a single A100:

- hidden-state extraction on the full dataset: a few hours;
- probe training/evaluation: about 40 minutes;
- verbalization on 1,600 examples: about 25 minutes.

So this G0 is comfortably within local GPU resources and requires:

```text
paid API: 0
new annotation: 0
foundation-model training: 0
```

This is currently the strongest feasibility profile in the advisor topic pool.
