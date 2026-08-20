# Second Stage Plan — Confirming DLM Trajectory Fate

## Status and purpose

G0 on `GSAI-ML/LLaDA-8B-Instruct` / GSM8K passed the preregistered feasibility gate, but the strongest effects were selected from a grid over denoising steps, layers, and lead thresholds. The next stage is therefore **confirmation, not expansion**.

The scientific goal is to determine whether the two G0 discoveries survive when the selected cells are frozen, first on untouched/in-distribution examples, then on genuinely new but distribution-matched math problems, and finally on a second DLM family.

The second stage deliberately does **not** search for a new best step, layer, lead, parser, probe, or decoding geometry.

## G0 facts that are now frozen

G0 used:

```text
model             GSAI-ML/LLaDA-8B-Instruct
benchmark         openai/gsm8k test
examples          ids 0..999
prompt            MidTruth boxed-answer prompt
steps             64
generation        128 tokens
block length      32
temperature       0
surface state     complete x0 before token transfer
parser            strict numeric \\boxed{...}
hidden pooling    global mean over generated positions
probe             StandardScaler -> PCA(max 64) -> LogisticRegression
controls          surface features + same-layer step-0 hidden state
```

Two discovery cells survived the G0 gate:

| hypothesis | final-outcome-controlled task | frozen step | frozen LLaDA hidden tuple index | frozen minimum lead |
|---|---|---:|---:|---:|
| H1 | `transient_recovery` | 16 | 25 | 4 |
| H2 | `transient_overwrite` | 4 | 28 | 16 |

These values are **locked before Stage 2**. No neighboring-cell rescue is permitted if they fail.

## Why these datasets and models

### Why GSM1K is the primary confirmation dataset

The cleanest confirmation changes the questions while changing as little else as possible. GSM1K was explicitly commissioned to mirror GSM8K in style and complexity while using new human-written grade-school arithmetic problems. The released `ScaleAI/gsm1k` test split contains 1,205 examples with `question` and `answer` fields.

That makes GSM1K preferable to MATH500 for G1: MATH500 changes mathematical domain, difficulty, notation, and answer format at the same time. A failure there would not tell us whether G0 was a selection artifact or whether the phenomenon simply does not transfer to competition mathematics.

References:

- GSM1K paper, NeurIPS 2024 Datasets & Benchmarks: https://proceedings.neurips.cc/paper_files/paper/2024/hash/53384f2090c6a5cac952c598fd67992f-Abstract-Datasets_and_Benchmarks_Track.html
- Dataset: https://huggingface.co/datasets/ScaleAI/gsm1k

### Why Dream is the second model

`Dream-org/Dream-v0-Instruct-7B` is the strongest immediate model replication because the ACL 2026 probing seed paper already evaluates both LLaDA-8B-Instruct and Dream-v0-Instruct-7B with the same linear hidden-state probing paradigm. The paper reports that correctness information is decodable in both models, but the layer dynamics differ, so a Dream replication tests whether trajectory-fate information is a DLM-level phenomenon rather than a LLaDA-only representation.

References:

- Probing Functional Correctness in Diffusion Language Models: https://aclanthology.org/2026.acl-srw.15/
- Reference probing code: https://github.com/guan404ming/dlm-probing
- Dream model: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B

### Why Dream uses deterministic `maskgit_plus`

Dream's official generation implementation exposes both the default `origin` update and a confidence-ranked `maskgit_plus` update. Under `origin`, token transfer includes Bernoulli randomness even when token temperature is zero. That is a poor primary falsification geometry for our question: future random transfer decisions are not information that the current hidden state can encode.

Therefore G1-C uses the official deterministic `maskgit_plus` path with `temperature=0`, 64 steps, and 128 generated tokens. This keeps the trajectory deterministic given the current model state, matching the causal interpretation used in the LLaDA G0.

This is a **cross-model replication geometry**, not an attempt to claim exact step-wise equivalence between the two samplers.

## Literature boundary / collision check

The closest adjacent 2026 work still leaves this question open:

- **Time Is a Feature** establishes non-monotonic surface trajectories and temporal oscillation.
- **Probing Functional Correctness in DLMs** predicts eventual final correctness from hidden states, but does not condition on current surface correctness and final outcome to predict a future transient flip.
- **TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding** (arXiv:2607.03236) uses output/logit history and proposal persistence to improve commit timing on LLaDA, Dream, and LLaDA2-Mini. It is relevant evidence that trajectory stability matters, but it does not probe whether a *single current hidden state*, under fixed current and final correctness, predicts future recovery/overwrite.

The surviving claim remains narrow:

> At the same current surface state and the same final outcome, does the current DLM hidden representation contain information about a future transient correctness transition before that transition becomes visible?

## Stage 2 protocol

### G1-A — untouched GSM8K selection audit

**Purpose:** cheaply test whether the two selected G0 cells have the same direction on examples that were never used to choose them.

G0 used GSM8K test ids `0..999`. G1-A uses the untouched tail:

```text
dataset       openai/gsm8k test
ids           1000..1318
n             319
model         LLaDA-8B-Instruct
geometry      exactly the G0 64-step / 128-token deterministic geometry
capture       steps 0,4,16,63 only
layers        25,28 only
```

The exact hypotheses remain H1/H2 above. Because 319 examples are likely to yield only ~10–20 positive transient events per task, G1-A is a **directional audit**, not the decisive confirmatory test.

For a supported task, the direction is counted as preserved only when all three point estimates are positive in the preregistered direction:

```text
AUC > 0.50
AUC - surface AUC > 0
AUC - step0-hidden AUC > 0
```

A noisy G1-A failure does not kill the topic; it is reported and the decisive test is G1-B.

### G1-B — decisive independent-data confirmation on GSM1K

This is the main Stage-2 experiment.

#### G1-B0: 200-example surface-only gate

Before extracting any hidden states, run the first 200 GSM1K examples using the exact LLaDA G0 geometry and count only the two **locked** tasks at their locked step/lead.

Default preflight support:

```text
positive >= 6
negative >= 20
```

for each task. `GO_ONE` is sufficient to run the full confirmation; `STOP_LOW_LOCKED_SUPPORT` stops before the 1,205-example hidden-state run.

The gate is about whether the natural phenomenon exists on GSM1K. It must not be rescued by moving the step or reducing the lead threshold.

#### G1-B1: full 1,205-example confirmation

```text
dataset             ScaleAI/gsm1k test, all 1,205 examples
model               GSAI-ML/LLaDA-8B-Instruct
steps               64
generation          128 tokens
block length        32
temperature         0
parser              strict boxed numeric answer
capture steps       [0,4,16,63]
hidden indices      [25,28]
probe               same linear G0 pipeline
CV                  5-fold stratified, fixed random state
bootstrap           2,000 by default
```

No layer/step/lead grid is produced. The script evaluates exactly two rows.

#### Positive control

At step 63, on locked layers 25 and 28, final-correctness probing must still work:

```text
AUC >= 0.65
AUC - step0 AUC >= 0.03
```

If this fails, the confirmation is `GEOMETRY_NOT_VALIDATED`; a negative trajectory-fate result is not interpreted as scientific evidence.

#### Confirmatory statistic

For each locked task define:

```text
M = min(
    AUC_hidden - 0.55,
    (AUC_hidden - AUC_surface) - 0.03,
    (AUC_hidden - AUC_step0) - 0.03
)
```

The task is confirmed only when:

```text
one-sided 97.5% bootstrap lower bound of M > 0
and min(class counts) >= 25
```

The 97.5% lower bound is a conservative Bonferroni correction for the two predeclared hypotheses (family-wise alpha 0.05; alpha 0.025 per task). This is intentionally stricter than the exploratory G0 gate.

Possible outcomes:

```text
CONFIRM_BOTH
CONFIRM_ONE
FAIL_BOTH
LOW_SUPPORT_ONE_TASK
LOW_SUPPORT_BOTH
GEOMETRY_NOT_VALIDATED
```

Interpretation:

- `CONFIRM_BOTH`: strong evidence that G0 was not winner's curse; proceed to cross-model replication.
- `CONFIRM_ONE`: the surviving asymmetry is scientifically useful; proceed, but narrow the paper claim to that direction.
- `FAIL_BOTH`: kill or sharply demote the topic. Do not search new cells.
- low support: the phenomenon is not frequent enough on distribution-matched new problems to sustain the broad claim.
- invalid geometry: debug the positive control first.

### G1-C — cross-model replication on Dream

Run only if G1-B returns `CONFIRM_BOTH` or `CONFIRM_ONE`.

To isolate the model change, G1-C keeps GSM1K fixed and changes only the DLM family.

```text
dataset             ScaleAI/gsm1k, all 1,205
model               Dream-org/Dream-v0-Instruct-7B
sampler             official deterministic maskgit_plus
steps               64
generation          128 tokens
temperature         0
capture steps       [0,4,16,63]
parser              same strict boxed numeric answer
```

#### Frozen Dream layer mapping

We cannot reuse LLaDA tuple indices literally because Dream has a different depth. The mapping is fixed **before seeing Dream fate labels** by relative block depth:

```text
LLaDA recovery layer:   25 / 32 = 0.781 -> Dream round(0.781 * 28) = tuple index 22
LLaDA overwrite layer:  28 / 32 = 0.875 -> Dream round(0.875 * 28) = tuple index 25
```

Thus Dream tests exactly:

| task | step | Dream hidden tuple index | lead |
|---|---:|---:|---:|
| transient recovery | 16 | 22 | >=4 |
| transient overwrite | 4 | 25 | >=16 |

This mapping is also broadly consistent with the ACL probing paper's finding that Dream correctness signal is concentrated in roughly the middle/upper portion of its hidden-state stack on reasoning tasks. We do not inspect a Dream layer heatmap before the confirmatory result.

As in G1-B, first run a 200-example surface-only support gate. If the locked transient event does not occur often enough under deterministic Dream decoding, record **limited model generality** and stop; do not switch to `origin`, change step, or search layers to rescue it.

The same positive-control and confirmation statistic are used for the full 1,205-example Dream run.

A Dream failure with good class support and a valid positive control limits the cross-model claim, but it does not erase a confirmed LLaDA/GSM1K phenomenon.

## What is deliberately postponed to G2

Do not add these before the confirmation stage is decided:

- MATH500 or competition mathematics;
- ARC-Challenge / MBPP / HumanEval;
- LLaDA-1.5 / LLaDA2-Mini or other new DLM families;
- nonlinear probes or MLPs;
- layer sweeps on GSM1K or Dream;
- alternative lead thresholds;
- fallback-parser primary analyses;
- causal activation steering;
- early-exit / commit-gating applications.

If G1 survives, these become useful **generalization/mechanism** experiments rather than tools for rescuing a fragile discovery.

## Efficiency design

Stage 2 is intentionally cheaper than repeating G0 grids:

- capture only four denoising steps: `0,4,16,63`;
- capture only two layers per model;
- surface-only 200-example gates before each expensive full run;
- four independent GPU shards, no inter-GPU communication;
- no full hidden-state trajectory storage;
- no grid search and no second probe family.

The one-command runner stops automatically before Dream if same-model independent-data confirmation fails.

## Commands

From `02_dlm_trajectory_fate/`:

```bash
pytest -q tests/test_stage2.py
bash -n run_stage2_4gpu.sh
GPUS="0 1 2 3" ./run_stage2_4gpu.sh
```

The runner executes, in order:

```text
G1-A  LLaDA / untouched GSM8K ids 1000..1318
G1-B0 LLaDA / GSM1K 200-example surface gate
G1-B1 LLaDA / GSM1K 1,205-example locked confirmation
G1-C0 Dream / GSM1K 200-example surface gate
G1-C1 Dream / GSM1K 1,205-example locked model replication
```

## Expected outputs

```text
artifacts/g1a_gsm8k_holdout/confirm/locked_confirmation.{csv,json}
artifacts/g1b_gsm1k_preflight/locked_surface_{support.csv,gate.json}
artifacts/g1b_gsm1k_confirm/confirm/locked_confirmation.{csv,json}
artifacts/g1c_dream_preflight/locked_surface_{support.csv,gate.json}
artifacts/g1c_dream_gsm1k/confirm/locked_confirmation.{csv,json}
```

Raw hidden-state NPZ files should remain server-local and should not be committed.

## Final Stage-2 decision

The project graduates from “positive candidate” to “validated research topic” only if the same-model independent-data confirmation survives without moving the frozen cells.

A useful decision matrix is:

| G1-B GSM1K LLaDA | G1-C Dream | conclusion |
|---|---|---|
| both/one confirmed | both/one confirmed | strong topic; proceed to G2 mechanism/generalization |
| both/one confirmed | valid negative | real LLaDA phenomenon, model generality limited; still potentially publishable with narrower claim |
| fail both | not run | G0 likely selection-driven; stop |
| low support | not run | phenomenon too sparse off GSM8K for broad claim; strongly demote |
| geometry invalid | not interpretable | fix/reproduce correctness probe before deciding |

The main discipline is simple: **Stage 2 confirms the G0 claim; it does not search for another one.**
