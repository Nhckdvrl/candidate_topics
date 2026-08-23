# Master Topic Search

This directory is the **general research-topic search log** for `candidate_topics`.

It is intentionally broader than [`embodied_topic_search`](../embodied_topic_search/): this folder is not tied to one hot subfield. Its job is to search across the lab's neighboring research areas and adjacent scientific literatures for **specific, real, valuable research questions** that can survive the repository's increasingly strict topic-selection bar.

The main lesson from Topics 01–12 is that a clean experiment does not make a good research question. We repeatedly started from two real observations, invented a plausible relation between them, and then built an elegant experiment to test that bridge. Many such bridges simply did not exist.

The search policy here therefore starts from **observed problems and anomalies**, not from clever hypotheses.

---

## 1. What kind of topic are we looking for?

A strong candidate should ideally satisfy all of the following.

### 1.1 Start from something that already exists

Prefer:

- a robust anomaly already observed in a real system;
- a repeated failure mode reported by multiple papers / practitioners;
- two published results that genuinely conflict on the **same scientific object**;
- a sharp reversal / phase transition / collapse / crossover in existing results;
- a long-standing scientific problem whose study was previously bottlenecked by expensive human/expert work and for which LLMs make a qualitatively new experimental regime possible.

Avoid starting from:

> Paper A finds X, Paper B finds Y, therefore perhaps X and Y obey a deeper law Z.

`X` and `Y` can both be real while `Z` is merely an attractive story.

### 1.2 The question should be interesting before the method

The scientific question should be explainable in one sentence without mentioning:

- probes;
- SAE;
- hidden states;
- a particular benchmark;
- a threshold / gate;
- an implementation trick.

The listener should understand why the answer matters before hearing how we measure it.

### 1.3 Positive results must be worth being excited about

Feasibility is not enough.

Before registering a topic, assume the hypothesis is completely true and ask:

> If the cleanest possible result came out exactly as hoped, would this actually change how we understand the system / phenomenon, or would it merely confirm something that already sounds obvious?

If the positive result is too unsurprising, do not do the project.

### 1.4 The first decisive experiment should be simple

Prefer one clean contrast whose interpretation is nearly forced by the question.

A good first experiment often looks like:

```text
same object
same data / state / task
one meaningful factor changed
one primary observable
```

If interpretation requires a growing chain of matching rules, auxiliary probes, several prerequisite gates and exclusions of many alternative explanations, treat that as evidence that the question or construct may be unnatural.

### 1.5 Meaningful regime must exist before we invest

A phenomenon on a toy object is not sufficient if the intended scientific claim requires a natural / realistic / non-trivial regime.

Before treating a candidate as healthy, ask:

- does the target model actually perform the task competently?
- does the relevant event occur often enough at instance level?
- is there an open model / dataset / checkpoint that instantiates the phenomenon without fishing?
- can the experiment be run in a regime large enough that a positive result matters?

### 1.6 A null result should answer the question

The ideal experiment is informative in both directions.

Bad null:

> We did not observe the proposed relation, but maybe another layer, model, prompt, metric, threshold or dataset would reveal it.

Good null:

> In the exact regime where the motivating phenomenon is robust, changing the proposed causal factor produces essentially no meaningful change.

---

## 2. Two main search modes

### Mode A — Observed anomaly -> competing explanations -> decisive test

This is the default.

```text
robust observed anomaly
        ↓
why is this surprising / important?
        ↓
what explanations already exist?
        ↓
can one clean intervention distinguish them?
```

We should first build an **anomaly inventory** rather than immediately packaging every observation as a new project.

Useful anomaly shapes include:

- performance collapse and spontaneous recovery;
- non-monotonic scaling;
- stronger models behaving less human-like / less calibrated / less useful in a special task;
- train/test or capability/use dissociations that appear directly in behavior;
- phase transitions during training;
- sharp disagreement between methods that are supposed to measure the same object;
- a method helping strongly on one structurally related regime and hurting on another;
- aggregate success hiding a specific repeated failure event;
- human/model disagreement concentrated exactly where a confidence or evaluation signal is supposed to help.

### Mode B — Old problem -> LLM changes the research paradigm

This is a secondary but important search route inspired by the advisor's taste.

Look for problems that existed **before modern LLMs** and were scientifically interesting, but progress was bottlenecked by things such as:

- expensive expert annotation;
- large human panels / behavioral experiments;
- manual taxonomy construction;
- hand-built rules / lexicons / knowledge bases;
- laborious literature review;
- one-off domain-specific pipelines;
- inability to generate sufficiently diverse controlled stimuli;
- inability to simulate or test many plausible hypotheses at scale.

The key question is **not**:

> Can an LLM replace the old classifier / annotator and improve accuracy?

That is usually too incremental.

Instead ask:

> Does the availability of LLMs make a previously impractical *scientific question or experimental design* possible for the first time?

Good examples of the shape (not necessarily open topics):

- replacing expensive psychometric field tests with synthetic populations might enable rapid item calibration — but then the scientific question becomes whether synthetic populations preserve the population variation that calibration fundamentally relies on;
- expert elicitation / Delphi-style panels can be scaled cheaply with LLM personas — but only if many synthetic agents contain genuinely independent information rather than correlated samples from one model;
- constructing thousands of controlled natural-language stimuli was previously manual — LLMs may allow much larger causal behavioral experiments, provided generated stimuli preserve the intended intervention.

This route is especially attractive when the LLM is an **experimental instrument**, not merely the object being benchmarked.

---

## 3. Search scope

Do **not** restrict the search to linguistics or Japanese-specific phenomena.

The advisor / lab history suggests a broader preference for **specific tasks and specific scientific objects**, including but not limited to:

- unusual / constrained generation tasks;
- quiz / question answering and question construction;
- representation and embedding geometry;
- knowledge awareness / hallucination;
- model learning dynamics and grokking-like phenomena;
- translation and structured generation;
- information extraction / document understanding;
- citation / scientific-document analysis;
- annotation and meta-research;
- human–AI decision making;
- evaluation methodology;
- behavioral science where LLMs enable new experimental scale;
- older NLP / cognitive / information-science problems that may change qualitatively in the LLM era.

Hotness of the field is not a requirement. A narrow, underexplored problem with a strong phenomenon can be better than a crowded frontier topic.

---

## 4. Explicit anti-patterns from the archive

Before registering anything, check against these recurring failure modes.

### Bridge-hypothesis failure

Two stable observables do not imply that their fine-grained structures correspond.

### Mechanism without mechanism-level phenomenon

Do not explain a hidden failure event that is not actually occurring in the chosen system.

### Non-identifying intervention

A seemingly intuitive intervention may change the task itself and therefore fail to distinguish the intended explanations.

### Aggregate metric != mechanism

A full-sequence / aggregate score can strongly track a property even when the local causal signal required by the story is absent.

### Shared global geometry

Two profiles can correlate because both vary with depth / time / scale without corresponding locally.

### Toy-only effect

A clean effect is insufficient if a meaningful non-toy experimental object cannot be instantiated.

### Natural-crossover scarcity

If the experiment depends on per-instance winner reversals or disagreement, verify that enough such cases actually exist before designing the full study.

### Complexity smell

If every criticism adds another control or another gate, reconsider the question instead of continuing to patch the protocol.

### Obvious-positive problem

If the hypothesis being true would sound like “of course,” the project is unlikely to satisfy the significance bar even if the experiment works perfectly.

---

## 5. Candidate intake template

Do **not** create a numbered topic directory immediately.

First record candidates in search logs using this minimal schema:

```markdown
## Candidate / anomaly name

### Observed phenomenon
What has already been observed? Give exact papers / systems / numbers when available.

### Why it is surprising
Why is this not the obvious expected behavior?

### Why it matters
If we understood or overturned the current interpretation, what would change?

### Existing explanations
What explanations have the original authors or neighboring literature already proposed?

### Exact open question
One sentence.

### Cheapest decisive contrast
What is the first experiment that could materially answer the question?

### Meaningful-regime check
Do we already have a competent, sufficiently natural experimental object?

### Collision audit
Has the exact question or contrast already been done?

### Verdict
KEEP / DEEP-AUDIT / HOLD / KILL.
```

A candidate graduates to its own numbered project only after the phenomenon, novelty and experimental object survive this audit.

---

## 6. Evidence sources to search

Search broadly rather than treating conference papers as the only source of insight.

High-value sources include:

- ACL / EMNLP / NAACL / EACL / COLM / TACL / ICLR / ICML / NeurIPS / AAAI;
- older NLP, information science, HCI, cognitive science, psycholinguistics, education, scientometrics and IR literature;
- 2025–2026 arXiv papers when the topic is moving quickly;
- paper appendices and negative ablations;
- GitHub issues / discussions and released experiment logs;
- practitioner blogs and research-engineering posts;
- repeated anomalies visible in training curves / traces;
- authors' stated limitations and unresolved empirical contradictions.

For the old-problem route, explicitly search literature from **before the LLM era** (often 1990s–2021) to understand what was expensive or impossible, then audit whether LLMs genuinely remove that bottleneck.

---

## 7. Current search strategy

The current broad search should proceed in rounds.

### Round structure

1. **Phenomenon mining** — collect observations only; do not invent a project yet.
2. **Historical audit** — for promising old problems, identify the pre-LLM bottleneck.
3. **Collision audit** — search the exact new question aggressively, especially 2025–2026.
4. **Meaningful-regime audit** — verify an open/reproducible experimental object exists.
5. **Interestingness test** — assume the cleanest result is positive and ask whether it is genuinely exciting.
6. **Only then design G-0.**

The search log should preserve killed ideas as well as survivors so we do not repeatedly rediscover the same tempting but weak directions.

---

## 8. Current working principle

The central rule for this folder is:

> **Do not search for a hypothesis that might be true. Search for a real phenomenon that demands an explanation, or an old scientific bottleneck that LLMs genuinely make possible to remove.**

A method comes later.
