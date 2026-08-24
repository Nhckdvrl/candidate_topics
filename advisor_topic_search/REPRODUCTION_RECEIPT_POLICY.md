# Reproduction Receipt Policy

Date: 2026-08-24

This file is a **hard promotion rule** for `advisor_topic_search/`.

A topic may look scientifically excellent and may have public code, but it **must not be formally registered as a numbered candidate** until its motivating experimental object has an exact reproduction receipt.

## Why this policy exists

Recent candidates exposed a repeated failure mode:

```text
paper reports an interesting phenomenon
→ we infer that an accessible open model must instantiate it
→ we design a new G0 / mechanism
→ only after registration do we discover that the exact seed cell was not pinned down,
  the reproduction regime is not usable, or our evaluator is not seed-faithful
```

That ordering is now forbidden.

The new ordering is:

```text
paper / phenomenon
→ exact seed cell
→ exact released artifact
→ local reproduction receipt
→ critical-cell screen
→ scientific next question
→ only then numbered registration
```

## Required receipt fields

Before promotion, write down all of the following.

### Paper cell

```text
paper / venue / year:
exact table / figure / section:
model:
model checkpoint or revision:
dataset:
split:
task / condition:
prompt or prompt file:
decoding / inference:
metric:
n:
seed(s):
reported value(s):
```

“Model X also shows the trend” is not enough. The value must come from the exact experimental cell we intend to reproduce.

### Artifact cell

```text
official repository:
repository commit:
exact entrypoint / script:
exact config:
data revision / file:
scorer / parser:
required external service or API:
```

“Official code exists” is not enough. We should be able to identify the command / config that produces the seed result.

### Local receipt

Record:

```text
local git commit:
host / GPU:
python / torch / transformers:
model revision actually loaded:
data revision actually loaded:
exact command:
observed metric(s):
comparison to paper cell:
engineering anomalies:
```

No invented absolute threshold is allowed. The reproduction decision must be tied to the paper's exact reported cell, official released outputs, or a clearly justified tolerance derived **before** inspecting the new result.

## Preferred objects

Prefer, in this order:

1. released instance-level logits / predictions / traces;
2. released deterministic scorer + open model;
3. exact executable / symbolic labels;
4. public checkpoints and frozen outputs;
5. local deterministic inference;
6. free-form generation only when unavoidable.

A deterministic next-token probability experiment is safer than a long free-form generation experiment because it removes decoding, truncation and parsing as major failure surfaces.

## Critical-cell receipt

Mechanism work requires more than aggregate reproduction. Before hidden-state work, measure the exact instance-level event the mechanism will explain.

Example:

```text
condition A behaves correctly
condition B exhibits the target failure
a matched control variable remains intact
```

If the event is sparse, the topic does not get rescued by model / prompt / layer / subset shopping.

## Allowed repair vs forbidden rescue

Allowed once:

- an objectively wrong script / config / model revision was used;
- the published inference contract was implemented incorrectly;
- a parser or metric demonstrably disagrees with the official evaluator;
- an environment bug prevents the exact experiment from executing.

Forbidden as a way to obtain a positive result:

- changing model family;
- changing prompt after observing the result;
- changing data subset;
- changing seed until positive;
- inventing a new metric;
- relaxing a scientific gate;
- broad model / layer / threshold sweeps.

## Promotion statuses

Use these labels in Round logs:

- `DISCOVERED` — interesting external observation only.
- `ARTIFACT_VERIFIED` — exact public implementation / data path found.
- `RECEIPT_PENDING` — exact seed cell identified; local reproduction not yet complete.
- `REPRODUCED` — receipt completed.
- `CRITICAL_CELL_READY` — exact instance-level event is dense enough.
- `REGISTER` — may become a numbered candidate.
- `STOP_REPRODUCTION` — exact seed object did not reproduce after one justified implementation repair.

## Final principle

> **Do not spend mechanism creativity on an experimental object we have not personally verified exists in the exact model / artifact regime we intend to study.**
