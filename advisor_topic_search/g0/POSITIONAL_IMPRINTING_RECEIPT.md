# Positional Imprinting — Reproduction Receipt Card

Status: `ARTIFACT_VERIFIED / RECEIPT_PENDING`

This file freezes the prerequisite for the Round-11 candidate **Positional Imprinting of Parametric Knowledge**.

Do not implement the new history manipulation until this receipt passes.

---

## 1. Seed paper cell

```text
paper: Where is the answer? An empirical study of positional bias for parametric knowledge extraction in language model
venue/year: NAACL 2025 Main, Long Paper
section: 4.3 Empirical Study on the Positional Bias
primary controlled object: Wiki2023+ D_k position modulation, Fig. 3 / Fig. 4
model: Llama-2-7B-chat
training method: vanilla autoregressive (AR)
optimizer: Adam
initial learning rate: 1e-5, linear decay
steps: 3000
batch: 256 total, mixed QA + document samples
metric: closed-book QA exact match; F1 secondary
manipulation: same original first sentence s1 moved to different sentence positions k
```

Before running locally, transcribe and freeze the exact numeric Fig. 4 target values from the publication/source figure. No tolerance may be chosen after seeing the local result.

Secondary exact aggregate check from Table 1, unmodulated Wiki2023+:

```text
Llama-2-7B AR
EM1 40.9
EM2  6.3
EM3  8.1
EM4 11.7
EM5 11.6
EM6 10.7
Avg 14.9

F1_1 54.0
F1_2 20.5
F1_3 29.8
F1_4 35.7
F1_5 37.8
F1_6 36.4
Avg 35.7
```

The secondary check cannot replace the controlled D_k receipt because natural document content varies by position.

---

## 2. Artifact cell

```text
official repository: https://github.com/omron-sinicx/WhereIsTheAnswer
frozen upstream commit: 910fcddec93f7400b58257d70abf1dab31f1e179
training entrypoint: train.py, invoked by run_scripts/train.sh
base training config: train_configs/base.yaml
position transformation: make_dataset/change_order_film.py
dataset pointers: Wiki2023+ and Synthetic Language in official README
evaluation: official released evaluation path / scorer
paid external API required for seed run: no
```

Artifact completeness was checked on 2026-08-24: train/eval/config/data-building code is present.

---

## 3. Local receipt fields — fill only after execution

```text
local candidate_topics commit:
host / GPU:
python:
torch:
transformers:
deepseed:
model revision actually loaded:
data revision / file hashes:
upstream repo commit actually checked out:
exact command:
exact config overrides:
observed controlled-position metrics:
comparison with frozen paper values:
secondary Table-1 sanity check:
engineering anomalies:
receipt verdict:
```

Allowed engineering repair is limited to an objectively incorrect dependency/config/path/evaluator implementation. Do not alter model family, prompt, subset, seed, metric, or scientific gate to obtain the expected effect.

---

## 4. Receipt verdict rule

Use one of:

- `REPRODUCED` — seed positional-bias cell is reproduced under the frozen official contract;
- `STOP_REPRODUCTION` — it does not reproduce after at most one objectively justified engineering repair;
- `BLOCKED` — exact official artifact cannot execute for a non-scientific infrastructure reason that has not yet been repaired.

No history/mechanism experiment proceeds under `STOP_REPRODUCTION` or unresolved `BLOCKED`.

---

# 5. Post-receipt mother G0 — frozen design only, not yet implementation

Scientific question:

> If the final fact-position exposure multiset and the recent training tail are identical, does early position-of-acquisition still affect final closed-book accessibility?

Matched schedules:

```text
EARLY-FIRST: 1,1,5,5 + 1,5,1,5
LATE-FIRST:  5,5,1,1 + 1,5,1,5
INTERLEAVED: 1,5,1,5 + 1,5,1,5
```

Every condition must have the same:

- facts;
- P1/P5 document variants;
- number of P1/P5 exposures;
- QA samples;
- total token/update budget;
- final four positional exposures;
- optimizer state trajectory except for the intended sample-order difference.

### Critical engineering constraint

The official path inherits Hugging Face `Trainer` sampling behavior. Dataset-file order alone is not a valid training-history manipulation.

The post-receipt implementation must explicitly guarantee the optimizer's sample schedule, e.g. by a frozen sequential/distributed sampler or an equivalent phase-controlled implementation that preserves optimizer state.

Do not write or tune that implementation before the seed receipt passes.

### Planned primary statistic

```text
Delta_history = paired final QA accuracy(EARLY-FIRST) - paired final QA accuracy(LATE-FIRST)
```

A practical positive/equivalence region must be frozen from seed-independent power considerations before running the history G0. Round 11 proposes `>=5 pp` as paper-interest positive and `+/-3 pp` as an equivalence region, but these values are not final until the preregistration/power card is explicitly frozen before execution.

---

## 6. Promotion gate

```text
ARTIFACT_VERIFIED
    -> exact seed receipt
    -> REPRODUCED
    -> matched history G0
    -> mother phenomenon survives
    -> branch-map audit
    -> only then REGISTER / Topic 25
```
