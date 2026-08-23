# Advisor Topic Search

This directory is the **advisor-facing research-topic search track**.

Its job is different from [`user_interest_topic_search`](../user_interest_topic_search/): here the first constraint is not "what frontier topic is personally exciting?" but rather:

> **What specific research question would plausibly fit the actual Sasano-lab style, survive advisor scrutiny, use accessible data, and mature into a defensible NLP / language-science / information-science paper?**

This track must not import Agentic RL, VLA, robotics, generic post-training, or generic mechanism work merely because those areas are fashionable or personally interesting.

The goal is to search **inside and adjacent to the lab's demonstrated research neighborhood**, while avoiding direct duplication of current labmates.

---

# 1. What this track is for

This directory searches for projects with the following shape:

```text
specific linguistic / informational / scientific object
        ↓
crisp empirical question
        ↓
existing or cheaply constructible data
        ↓
simple auditable measurement
        ↓
clear first experiment
        ↓
ACL/EMNLP/NAACL/EACL/TACL-class paper shape
```

The advisor-facing standard emphasizes:

- concrete objects;
- natural questions;
- strong data paths;
- interpretable empirical analysis;
- moderate solo-project scale;
- clear novelty beyond "new model on old benchmark";
- results that can be explained without a long chain of latent assumptions.

This does **not** mean every topic must be traditional linguistics. LLMs, open checkpoints, embeddings, scientific documents, structured generation, and large-scale model behavior are all acceptable when attached to a concrete object and a clean question.

---

# 2. What the lab/advisor research style actually looks like

The lab neighborhood should be inferred from real work, not from generic "NLP" labels.

Representative recent / neighboring objects include:

- character-level information acquisition in LMs;
- semantic frames / frame induction / frame definitions;
- text embedding geometry, redundancy, intrinsic dimension and compression;
- quiz answering, early answering, clue structure and difficulty;
- human-vs-LLM difficulty comparisons;
- citation importance and related-work organization;
- scientific-document information extraction;
- specialized structured generation, including patent-style translation;
- multilingual performance factors;
- LLM-based survey replication / personality inference;
- model-internal grammatical/computational structure;
- detailed Kanji-description generation;
- concrete linguistic acquisition trajectories across open checkpoints.

The common denominator is:

> **a named object that is already scientifically meaningful before the model is introduced.**

Examples:

- clue order is a real property of quiz questions;
- semantic frames are a real linguistic resource object;
- patent-claim structure is a real domain constraint;
- citation organization is a real scientific-writing object;
- embedding compression is meaningful because downstream semantics must survive it;
- acquisition order is meaningful because the target linguistic property can be tested directly.

The model is a tool or experimental system, not the sole reason the problem exists.

---

# 3. Search theme range

## 3.1 S-tier: quiz / question science

This is one of the cleanest advisor-fit areas because the objects and observables are concrete.

Search around:

- clue ordering;
- earliest answerable clue;
- information gain across clues;
- clue redundancy;
- clue ambiguity;
- distractor structure;
- human-vs-LLM difficulty reversals;
- question construction;
- answerability under partial information;
- easy-for-human / hard-for-model and reverse subsets;
- whether generated questions preserve real difficulty structure;
- changes in answerability when clue structure is minimally manipulated.

Prefer questions about **how questions work**, not simply model accuracy on quiz datasets.

Strong first experiments often look like:

```text
same question
one clue property changed
answerability / difficulty measured directly
```

Avoid generic QA benchmarking.

## 3.2 S-tier: acquisition / development of concrete language or symbolic properties

The lab already has strong affinity for checkpoint-rich acquisition studies.

Search for properties that are:

- clearly definable;
- automatically testable at scale;
- not already being studied by current labmates;
- interesting as developmental trajectories rather than one final-model score.

Possible object classes:

- orthographic regularities;
- morphological behavior;
- lexical relations;
- compositional patterns;
- syntactic constraints;
- conventionalized symbolic patterns;
- domain-specific structural conventions;
- factual/semantic distinctions with mechanically generated probes.

Valuable phenomenon shapes:

- abrupt acquisition;
- non-monotonic learning;
- temporary regression;
- ordering differences across model scales;
- stable acquisition sequence across independent model families;
- one property learned only after another prerequisite property appears;
- surprising late emergence despite early exposure.

Hard warning: do not duplicate current character-level / typoglycemia / onomatopoeia themes.

## 3.3 S/A-tier: text embedding structure, redundancy and compression

The lab has active expertise here, so adjacent questions can be highly advisor-compatible.

Search around:

- how much dimension can be removed before a **specific semantic capability** fails;
- tasks that are unusually fragile or unusually invariant to compression;
- whether different semantic relations occupy distinct effective dimensions;
- geometry changes caused by prompt templates;
- reconstructability vs task utility;
- anisotropy / redundancy only when tied to a concrete downstream phenomenon;
- cross-lingual compression behavior;
- whether embedding equivalences break under a precise semantic contrast.

Avoid:

- another generic CKA/cosine/analogy correlation paper;
- "we compressed embeddings and average score changed slightly";
- arbitrary hidden geometry with no concrete task consequence.

The healthy shape is:

```text
concrete semantic behavior
-> controlled embedding intervention
-> direct capability cliff / invariance
```

## 3.4 A-tier: scientific-document / citation / scholarly communication analysis

This is clearly lab-adjacent but must incorporate the failure of Topics 16 and 17.

Good objects:

- citation placement;
- citation purpose when structurally annotated;
- related-work organization;
- claim/reference alignment with directly visible textual evidence;
- scientific entities / methods / datasets;
- reproducible extraction from paper sections;
- document structure;
- citation chains only when the target label is auditable without exhaustive semantic provenance.

Prefer **structural or local labels**.

Avoid core measurements requiring:

```text
same semantic proposition
+ hidden evidence provenance
+ expert epistemic certainty
+ exhaustive literature audit
```

unless the measurement problem itself is the paper.

Topic16/17 show that a beautiful meta-science question can still be a terrible solo-project measurement object.

## 3.5 A-tier: semantic resources / frames

Search around:

- frame coverage;
- missing distinctions;
- consistency of frame definitions;
- induction errors;
- lexical-unit coverage;
- cross-resource disagreement;
- cross-lingual frame alignment;
- how generated definitions affect annotation or retrieval;
- whether modern LMs expose systematic blind spots in an existing resource.

Prefer a **specific resource defect or scientific property**, not "build a bigger FrameNet with an LLM."

## 3.6 A-tier: specialized translation / structured generation

The target is not generic MT quality.

Search for domains where target text obeys explicit structure:

- patent claims;
- legal / regulatory formulations;
- scientific abstracts or structured summaries;
- definitions;
- formal descriptions;
- technical instructions;
- controlled-language generation.

Strong questions concern:

- systematic structural violations;
- information omission/addition;
- ordering constraints;
- referential consistency;
- terminology consistency;
- preservation of logical dependency;
- mismatch between generic fluency and domain validity.

Metrics should preferably be programmatic or expert-light.

## 3.7 A/B-tier: LLMs as instruments for older labor-intensive science

This is potentially very advisor-friendly when done carefully.

Search for pre-LLM scientific questions historically limited by:

- expensive annotation;
- manual taxonomy construction;
- reading thousands of documents;
- hand-writing controlled stimuli;
- large human panels;
- expensive survey coding;
- slow comparison across many resources/languages.

The strong project shape is:

```text
old scientific question
+ historically expensive measurement
+ validated LLM-assisted scaling
-> previously impossible large-scale empirical test
```

The weak shape is:

```text
LLM replaces annotator and gets 92% accuracy
```

The scientific result, not the annotator replacement, must be the point.

## 3.8 B-tier: multilingual / figurative / lexical phenomena

This can fit the lab but should be used selectively because generic multilingual benchmark work is crowded.

Search only when there is a concrete phenomenon such as:

- strong cross-language reversal;
- resource/frequency factor explaining a specific gap;
- literal/figurative dissociation;
- systematic failure tied to a linguistic structure;
- acquisition-order differences across languages.

Avoid broad "evaluate idioms in 20 languages" work unless the result answers a sharper scientific question.

---

# 4. Explicitly out of scope for this track

Unless there is direct advisor interest or a very concrete NLP object, do **not** put the following here:

- generic Agentic RL;
- tool-learning agents;
- web agents;
- generic reasoning RL;
- generic RL trace analysis;
- robot/VLA control;
- world models / WAM;
- embodied policy mechanisms;
- generic continual learning;
- generic SAE / activation-patching studies;
- broad mechanistic interpretability detached from a language object;
- generic RAG;
- benchmark-only model comparisons;
- optimization-algorithm papers with no concrete language/information object.

These belong in [`user_interest_topic_search`](../user_interest_topic_search/) unless they independently acquire a strong advisor-facing object.

---

# 5. How to search

The search should be **object-first**, not keyword-first.

Bad process:

```text
search "latest NLP research gaps"
-> pick fashionable keyword
-> invent hypothesis
```

Good process:

```text
identify concrete object family
-> inspect recent lab / neighboring papers
-> inspect older literature on that object
-> collect unresolved phenomena / contradictions / limitations
-> search exact 2025–2026 collisions
-> design one decisive empirical contrast
```

## 5.1 Start from lab neighborhood

For each lane, inspect:

- Sasano-lab publications;
- recent M2 / advanced student themes;
- papers cited heavily by lab work;
- neighboring groups working on the same object;
- datasets/resources already familiar to the lab.

Do not copy a labmate's exact topic. Search one level outward:

```text
current lab object
-> adjacent property / failure / dataset / scientific question
```

## 5.2 Search older literature

This matters especially for:

- quiz science;
- psycholinguistics;
- semantic resources;
- scientometrics;
- translation conventions;
- educational measurement;
- information extraction;
- embedding semantics.

Older papers often contain real questions that were expensive to test at scale. Modern LLMs/checkpoints may make them tractable.

## 5.3 Search 2025–2026 exact collisions

For every surviving lead, search:

- exact scientific question;
- exact object;
- exact experimental contrast;
- obvious method opening.

A topic is not novel because the exact title string is new.

## 5.4 Inspect appendices and error analyses

Useful topic sources:

- unexplained failure categories;
- large subgroup reversals;
- annotation disagreement;
- surprising model-size effects;
- data/resource limitations;
- negative ablations;
- human/model disagreement tables;
- non-monotonic checkpoint plots;
- properties that current evaluation averages hide.

---

# 6. Advisor-track workflow

## Stage A0 — object inventory

Record:

```text
Concrete object:
Why this object is scientifically meaningful:
Existing dataset/resource:
Closest lab connection:
Current labmate collision:
```

No hypothesis required yet.

## Stage A1 — phenomenon / historical-question inventory

For each object, collect only questions with an external anchor:

- known anomaly;
- published contradiction;
- repeated failure;
- old scientific question;
- large human/model difference;
- untested consequence explicitly exposed by an existing result.

Do not create a project from an empty cell alone.

## Stage A2 — advisor-fit screen

Ask:

```text
Can the question be understood immediately?
Would it sound natural in an NLP/language-science seminar?
Is the object concrete?
Can the main variable be measured simply?
Is data already accessible?
Could one student execute this in ~half a year?
```

If not, kill before method design.

## Stage A3 — collision audit

Search:

- labmate overlap;
- exact 2025–2026 literature;
- same question in older literature;
- obvious method already saturated.

## Stage A4 — first decisive experiment

The first experiment should ideally be:

```text
same object
one controlled property changed
one direct outcome
```

or:

```text
open checkpoints
one concrete property
trajectory across training
```

or:

```text
large corpus/resource
one structurally defined phenomenon
frequency / distribution / failure rate
```

Avoid protocols that require many semantic judgments merely to define the sample.

## Stage A5 — paper-shape audit

Before promotion answer:

```text
Positive headline:
Why would the advisor care?
Why would ACL/EMNLP readers care?
What exactly is new scientifically?
What data already exists?
What is the first main table/figure?
If positive, what follow-up method or analysis becomes natural?
```

Only then create a numbered candidate.

---

# 7. Evidence priority

For advisor search, evidence should be weighted roughly as follows.

## Tier S

- exact phenomenon already observed in published work;
- strong human/model disagreement on a concrete object;
- explicit unresolved old question;
- released dataset/resource containing the phenomenon;
- repeatable checkpoint trajectory.

## Tier A

- one strong published observation plus independent reason the extension matters;
- explicit limitation/future-work question with accessible data;
- neighboring resource contradiction.

## Tier B

- plausible adjacent gap;
- one paper plus analogy to another domain.

Use only for cheap exploration, not full project registration.

## Tier C

- pure cross-paper bridge hypothesis;
- latent mechanism guessed from aggregate results.

Normally reject.

---

# 8. Hard promotion requirements

A lead should not leave the search log unless all are true:

1. **Concrete object** is clearly named.
2. **External anchor** exists.
3. **Advisor fit** is explicit.
4. **Labmate collision** has been checked.
5. **2025–2026 collision** has been checked.
6. **Data path** already exists.
7. **Primary measurement** is simple and auditable.
8. **One decisive first experiment** can substantially raise or kill the idea.
9. **Positive result is interesting**, not merely clean.
10. **Project fits solo scale** and does not begin with months of annotation/data construction.
11. **Null result is informative** rather than "try another model/threshold."
12. **No complexity smell**: interpretation does not require an expanding control stack.

---

# 9. Lessons from the existing failed topics that matter especially here

## Topic16 / 17 lesson — meta-science can fail at measurement

A naturally interesting citation/reproducibility question is still unsuitable if the core label requires exhaustive semantic and provenance auditing.

Therefore scientific-document projects should strongly prefer locally observable structure.

## Topic10 lesson — toy phenomenon is not enough

If the meaningful regime cannot be instantiated, do not promote a cute small-scale effect.

## Topic12 lesson — profile correlation is not a scientific law

Embedding/checkpoint profiles can share broad geometry without local correspondence.

## Topic05 lesson — semantic constructs can become unnatural

If defining the thing requires many exclusions, the object may not be stable enough for a good project.

## Topic14/15 lesson — strong seed phenomenon does not validate our explanation

Advisor-track projects should prefer **describing or testing a concrete phenomenon directly** before adding a mechanistic story.

---

# 10. Boundary with the user-interest track

This track answers:

> **What project is most defensible and natural inside the advisor/lab research environment?**

The user-interest track answers:

> **What frontier-AI problem is personally worth pursuing even if it does not resemble current lab work?**

Do not blur them.

Examples:

| Topic shape | Default home |
|---|---|
| Agentic RL recovery across checkpoints | user-interest |
| VLA action-chunk control staleness | user-interest |
| reasoning-RL strategy collapse | user-interest |
| quiz clue information structure | advisor |
| semantic-frame coverage inconsistency | advisor |
| embedding capability under compression | advisor |
| acquisition trajectory of a concrete linguistic property | advisor |
| patent-structure preservation | advisor |
| structurally measurable scientific-document phenomenon | advisor |

A lead may appear in both only if it independently satisfies both sets of requirements. Rebranding alone is not enough.

---

# 11. Required format for every future advisor search round

Every round should contain:

```text
Search scope
Objects inspected
Sources inspected
Observed phenomena / historical questions
Exact labmate collisions
Exact literature collisions
Candidates retained
Candidates killed and why
Next search branches
```

Each retained lead must record:

```text
Concrete object:
External anchor:
Natural question:
Why advisor-fit:
Why scientifically interesting:
Available data:
Primary observable:
Closest labmate:
Exact literature collision:
Cheapest decisive experiment:
Positive-result headline:
Method/analysis opening:
Status:
```

The search log should preserve killed leads too, because avoiding repeated rediscovery is part of the purpose of this directory.
