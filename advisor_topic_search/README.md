# Advisor Topic Search

This is the **advisor-facing topic-search track** for projects that must plausibly fit the actual Sasano-lab research style.

It is intentionally separated from `user_interest_topic_search/`. Agentic RL, VLA, generic post-training, and robotics are **not** assumed to be acceptable here merely because they are fashionable or personally interesting.

## 1. What the advisor/lab actually appears to value

Recent Sasano-lab work and presentations (2025–2026) repeatedly use **specific, well-defined language/information objects** and ask a crisp empirical question about them. Representative neighboring objects include:

- character-level information acquisition in language models;
- semantic frames / frame induction / frame definitions;
- prompt-based text embedding structure, redundancy, intrinsic dimension, compression;
- Japanese quiz difficulty and early answering, including human-vs-LLM comparison;
- citation importance / related-work citation organization;
- scientific-document extraction and domain-specific information extraction;
- specialized structured generation such as patent-claim translation;
- LLM-based social-survey replication / personality inference;
- multilingual performance factors;
- model-internal grammatical/computational circuits;
- detailed Kanji description generation.

The common denominator is **not simply “NLP.”** It is:

> concrete object + naturally understandable question + existing data path + measurable outcome + moderate project scale.

## 2. Advisor-line constraints

A lead should normally satisfy all of the following before serious coding:

1. The object can be named without saying “Agent,” “RL,” “SAE,” or “VLA.”
2. The question would still sound legitimate in an NLP / language-science / information-science seminar.
3. Data can be obtained or constructed in days/weeks rather than months.
4. The primary observable is simple and auditable.
5. A successful paper can plausibly target ACL/EMNLP/NAACL/EACL/Findings/TACL or a closely related venue.
6. The topic does not simply copy a current labmate's exact object.
7. LLMs may be the object or the instrument, but the paper must not reduce to “try a newer model.”
8. Prefer large-scale verification of a concrete phenomenon or old hypothesis over invented latent mechanisms.

## 3. Priority search lanes

### S — Acquisition / development of concrete linguistic or symbolic information

Examples of acceptable *shapes*:

- when/how a directly testable property becomes available across open checkpoints;
- non-monotonic acquisition of a concrete skill;
- interference/order effects for a property with mechanically generated tests;
- capability appearing in behavior before/after a directly related representation measure, only when the behavioral phenomenon is already real.

Avoid duplicating current character-level / typoglycemia / onomatopoeia work.

### S — Quiz / question science

Objects are unusually clean: clue order, answerability, human difficulty, distractors, early-answer threshold, ambiguity, information gain.

Good questions should concern the **structure of questions and answering**, not just benchmark LLM accuracy.

### S — Text embedding structure and compression

The lab has active work on redundancy, isotropy, intrinsic dimension and compression. Adjacent work should identify a concrete failure/invariance/geometry phenomenon, not generic embedding probing.

### A — Scientific-document / citation analysis

Strong fit when the label is structurally observable or cheaply auditable. Topic16/17 show a hard warning: avoid projects whose core y-axis requires a complicated semantic ontology and exhaustive evidence provenance.

### A — Semantic resources / frames

Good fit when the question is about a concrete resource property, induction error, coverage, definition structure, or cross-resource consistency. Avoid simply building another frame benchmark.

### A — Specialized generation / translation

Good fit when there is a domain-specific structural requirement whose violation is measurable (patent claims, formal descriptions, constrained documents). Avoid generic BLEU/model comparison.

### B — LLM as instrument for old labor-intensive science

Potentially advisor-friendly if LLMs allow a previously impractical large-scale analysis. The scientific question must pre-exist the LLM and the measurement must be validated cheaply.

## 4. Explicit exclusions

Do not put these into the advisor track without direct evidence that the advisor wants them:

- generic Agentic RL;
- robot/VLA control;
- world models;
- generic RL trace analysis;
- generic continual learning;
- broad mechanistic interpretability without a very concrete language object;
- topics whose main novelty is an optimization algorithm unrelated to a concrete NLP phenomenon.

## 5. Promotion rule

A lead first enters a round log, not a numbered `topicXX` folder.

Before promotion record:

```text
Concrete object:
Observed phenomenon / historical question:
Why advisor-fit:
Exact open question:
Available data:
Primary observable:
Closest lab collision:
2025–2026 literature collision:
One decisive first experiment:
Positive-result significance:
```

Only leads that survive both **advisor-fit** and **scientific-interest** screening should be promoted.

See `ROUND_01_2026-08-23.md` for the first advisor-specific reset.
