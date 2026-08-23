# Advisor Topic Search — Binding Resource Constraints

These constraints are **promotion gates**, not optional preferences.

Our resource profile must be stated accurately:

> **We are cash-constrained and annotation-constrained, but comparatively compute-rich.**

That means the wrong project for us is one whose first scientific claim requires tens of thousands of paid frontier-model API calls or a new human annotation campaign. The right project can be quite GPU-intensive if it uses open models, public checkpoints, existing labels, and local mechanistic analysis.

## R1. Low cash cost

A candidate should **not require large-scale paid closed-model API calls** to establish that the phenomenon exists.

Preferred resources, in order:

1. already released response traces / logits / predictions / human judgments;
2. public datasets with existing gold labels;
3. open-weight models runnable locally;
4. public intermediate checkpoints;
5. programmatically generated examples with exact labels;
6. only then, a small number of paid API calls for spot checks or external generalization.

A project whose G0 requires tens or hundreds of thousands of GPT / Claude / Gemini calls is downgraded even if the scientific question is attractive.

## R2. Low annotation dependence

A candidate should **not require hiring annotators or constructing thousands of new semantic labels** before the main phenomenon can be tested.

Strongly preferred labels are:

- existing benchmark gold answers;
- exact-match / executable / symbolic / numeric labels;
- public human-response logs;
- public revision history or structured metadata;
- deterministic transformations with known answers;
- already released expert annotations.

A small manual audit is acceptable **after** a signal exists, for example 50–200 high-value cases to estimate false-positive rate. Large annotation is not acceptable as a prerequisite for proving the object exists.

## R3. Compute-rich preference

Local GPU computation is an **advantage**, not a bottleneck to be minimized away.

Once a cheap behavioral / phenomenon G0 is established, we should actively exploit available compute for:

- hidden-state extraction across layers;
- linear / nonlinear probes;
- layer-wise trajectory analysis;
- activation patching and causal tracing;
- attention / MLP ablation;
- steering and representation engineering;
- SAE / crosscoder analysis when justified by a stable phenomenon;
- dense checkpoint analysis;
- lightweight SFT / LoRA / RL or controlled fine-tuning for causal tests;
- quantization / compression interventions across open checkpoints.

A candidate that is slightly more GPU-expensive but has a clean mechanism opening can be preferable to a nearly free dataset-only project.

## R4. Existing-resource G0

The best candidate should permit the first decisive experiment using almost entirely existing artifacts:

```text
released dataset
+ released model/checkpoints or released responses
+ automatic scoring
= phenomenon-existence G0
```

Then, if G0 passes:

```text
same object
+ local GPU access to hidden states / interventions
= mechanism phase
```

If we have to create the scientific object ourselves before measuring it, treat the candidate skeptically.

## R5. Compute is abundant, but search is still not free

GPU abundance does **not** justify unconstrained fishing.

Before calling an experiment feasible, record:

- model size;
- model family;
- number of checkpoints;
- number of examples / rollouts;
- expected GPU-hours;
- storage for activations;
- whether multi-node communication is needed;
- number of layer / threshold / prompt choices exposed to researcher degrees of freedom.

Prefer a hypothesis-led mechanism experiment over a huge search across:

```text
model × checkpoint × layer × token × threshold × prompt × dataset
```

The latter creates winner's-curse risk even when GPU cost is affordable.

## R6. Resource kill rules

Kill or sharply downgrade a candidate if any of the following is true:

1. phenomenon existence depends on large paid API usage;
2. the only useful models are closed and no released outputs exist;
3. the first useful dataset requires substantial new annotation;
4. the label requires an expensive LLM judge for every example and no reliable automatic proxy exists;
5. the only way to instantiate the phenomenon is to train a large foundation model from scratch;
6. a supposedly cheap G0 actually requires a large model × checkpoint × prompt × sample sweep;
7. existing public artifacts cover only a toy regime and reaching a meaningful regime requires costly model/data/configuration fishing;
8. mechanism work would require so many flexible layer/feature choices that a positive result is not independently interpretable.

## R7. Candidate-card additions

Every promoted candidate must explicitly include:

```text
Existing resources:
Paid API requirement for G0:
New annotation requirement for G0:
Open-weight model availability:
Local compute estimate:
Activation / checkpoint storage estimate if relevant:
Can G0 be run entirely from released artifacts?:
Mechanism-ready after G0?:
```

A candidate that cannot answer these lines concretely is not ready for coding.

## R8. Ideal project profile

The resource sweet spot is:

```text
natural NLP question
+ published anomaly or old scientific problem
+ existing gold data
+ open model / public checkpoint
+ automatic scoring
+ local GPU-heavy mechanism analysis
+ little or no paid API
+ little or no new annotation
```

That is the default target for future advisor-topic search.
