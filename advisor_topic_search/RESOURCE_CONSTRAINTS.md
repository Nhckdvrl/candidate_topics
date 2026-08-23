# Advisor Topic Search — Binding Resource Constraints

These constraints are **promotion gates**, not optional preferences.

The goal is to avoid selecting a scientifically attractive question whose phenomenon-existence test is financially or operationally infeasible for us.

## R1. Low cash cost

A candidate should **not require large-scale paid closed-model API calls** to establish that the phenomenon exists.

Preferred experimental resources, in order:

1. already released response traces / logits / predictions / human judgments;
2. public datasets with existing gold labels;
3. open-weight models runnable locally;
4. public intermediate checkpoints;
5. programmatically generated examples with exact labels;
6. only then, a small number of paid API calls for spot checks or external generalization.

A project whose G0 requires tens or hundreds of thousands of GPT/Claude/Gemini calls is downgraded even if the scientific question is good.

## R2. Low annotation dependence

A candidate should **not require hiring annotators or constructing thousands of new semantic labels** before the main phenomenon can be tested.

Strongly preferred labels are:

- existing benchmark gold answers;
- exact-match / executable / symbolic / numeric labels;
- public human-response logs;
- public revision history or structured metadata;
- deterministic transformations with known answers;
- already released expert annotations.

A small manual audit is acceptable **after** a signal exists, for example to verify 50–200 high-value cases or estimate false-positive rate. Large annotation is not acceptable as a prerequisite for proving the object exists.

## R3. Existing-resource G0

The best candidate should permit the first decisive experiment using **almost entirely existing artifacts**:

```text
released dataset
+ released model/checkpoints or released responses
+ automatic scoring
= phenomenon-existence G0
```

If we have to create the scientific object ourselves before measuring it, the candidate should be treated skeptically.

## R4. Compute is cheaper than cash, but still must be estimated

Local GPU computation is acceptable when the experiment fits existing hardware. We should still estimate:

- model size;
- number of checkpoints;
- number of examples / rollouts;
- expected GPU-hours;
- storage requirements;
- whether multi-node communication is required.

Prefer inference / probing / lightweight fine-tuning over training a foundation model from scratch unless the latter is uniquely necessary.

## R5. Resource kill rules

Kill or sharply downgrade a candidate if any of the following is true:

1. phenomenon existence depends on large paid API usage;
2. the only useful models are closed and no released outputs exist;
3. the first useful dataset requires substantial new annotation;
4. the label requires an expensive LLM judge for every example and no reliable automatic proxy exists;
5. the only way to instantiate the phenomenon is to train a large model from scratch;
6. a supposedly cheap G0 actually requires a large model × checkpoint × prompt × sample sweep;
7. existing public artifacts cover only a toy regime and reaching a meaningful regime requires costly data/model search.

## R6. Candidate-card additions

Every promoted candidate must now explicitly include:

```text
Existing resources:
Paid API requirement for G0:
New annotation requirement for G0:
Local compute estimate:
Can G0 be run entirely from released artifacts?:
```

A candidate that cannot answer these lines concretely is not ready for coding.
