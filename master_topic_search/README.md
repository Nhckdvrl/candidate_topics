# Master Topic Search

This directory is the **general research-topic search log** for `candidate_topics`.

It is intentionally broader than [`embodied_topic_search`](../embodied_topic_search/). The goal is not to stay inside one fashionable subfield and not to stay inside Japanese linguistics. The goal is to search across the advisor's preferred problem style, the lab's actual neighboring research areas, older scientific literatures, and current AI work for **specific, real, valuable research questions** that can survive a strict early kill process.

The governing lesson from the archived topics is simple:

> **A clean experiment does not make a good research question.**

We repeatedly started from two real observations, invented a plausible relation between them, and then built an elegant experiment to test that bridge. Many such bridges simply did not exist. This folder therefore starts from **observed problems / anomalies / long-standing bottlenecks**, not from clever hypotheses.

---

# 1. Advisor bar: what kind of problem is worth doing?

The advisor's taste is not "linguistics" as such. The recurring preference is for a **concrete and distinctive scientific object** whose question is immediately understandable and whose answer would be genuinely worth knowing.

Examples from the lab history include FrameNet-like semantic resources, detailed Kanji description, and quiz / early-answering tasks. What matters about these examples is not that they are language problems. They are **specific objects with a natural task and a crisp question**.

## 1.1 Interestingness before feasibility

The first question is not:

> Can we make this work?

It is:

> **If the cleanest positive result came out exactly as hoped, how happy should we be?**

A feasible result that sounds inevitable after hearing the story is not enough.

Before registering a topic, assume the hypothesis is completely true and ask:

> Would this change how we understand the system / phenomenon, or would it merely confirm something that already sounds obvious?

If the positive result is not surprising enough to matter, do not do the project.

## 1.2 Natural question first

A good topic should be explainable in one sentence before mentioning:

- probes;
- SAE;
- hidden states;
- a benchmark name;
- a threshold / gate;
- a specific architecture trick.

The listener should understand why the answer matters before hearing how we measure it.

## 1.3 Specific / slightly unusual is often better than hot / crowded

Do not optimize for the hottest current keyword. A narrow, underexplored object with a strong phenomenon can be much healthier than a crowded Agent / RAG / generic post-training direction.

The target is not "small" in scientific significance; it is **specific in object and question**.

## 1.4 Data must exist early

A topic should have a realistic data path from the beginning.

Prefer:

- public datasets;
- lab-owned or already accessible data;
- data that can be automatically constructed with a clear validity check;
- human annotation that is small and controlled.

Avoid projects that first require months of data collection, difficult data-access applications, large expert re-annotation, or building a huge dataset before we even know whether the scientific question is alive.

A useful operational target is:

- first data / pilot within roughly **1–2 weeks**;
- feasibility judgment within roughly **one month**;
- main experiments on the order of **3–4 months**;
- a project that can plausibly mature into a submission on roughly a **half-year** horizon.

## 1.5 Solo-project scale, but not artificially zero-training

The project should be manageable by one researcher and should not require a giant new system or frontier-scale pretraining.

However, "cheap" does **not** mean:

- zero-shot only;
- no training under any circumstances;
- rules only.

Training / finetuning / controlled small models are allowed when they directly answer the scientific question. Compute should serve the question rather than become the project.

## 1.6 International-paper bar

A healthy candidate should have a path to an international conference paper:

- clear scientific question;
- strong baseline / comparison;
- reproducible experiment;
- decisive main result;
- appropriate ablation or robustness analysis **after** the phenomenon is established;
- error / failure analysis;
- claims that do not outrun the evidence.

"Nobody has done exactly this dataset + model combination" is not sufficient novelty.

---

# 2. What the lab actually covers

The lab should be used as a **neighborhood map**, not as a restriction to copy a labmate's topic.

Recent / current neighboring work spans substantially more than Japanese linguistics:

- **training dynamics with open checkpoints** — Pythia / other checkpoint-rich model families, including acquisition trajectories and non-monotonic behaviors;
- **sentence embedding structure** — analogy-like structure, compression, dimensionality reduction, and embedding inversion / reconstruction;
- **diffusion-language-model generation dynamics** — what becomes determined when during denoising, including compositional semantic information;
- **mechanistic interpretability / knowledge awareness** — internal representation of known vs unknown entities, SAE-style analysis, and related behavior;
- **multilingual / figurative understanding** — idioms, proverbs, literal-vs-figurative processing, cross-lingual differences;
- **specialized translation / structured generation** — e.g. patent claims whose target-side structural conventions matter;
- **scientific-document analysis** — citation intent and other document-level labels;
- **question / quiz science** — early answering, clue structure, question construction, distractors;
- **generation for special communication tasks** — cases where ordinary generic generation does not directly solve the real task;
- **annotation / meta-research** — using LLMs to reduce the cost of large-scale analysis of scientific or institutional information.

The stronger references for the advisor bar are the more mature M2 / advanced projects and established lab lines. Very early B4 exploratory themes are useful for breadth but should not be treated as evidence that a topic already meets the advisor's publication bar.

## 2.1 Do not collide with labmates

Lab-adjacent means **search around the neighborhood**, not reproduce the same research question.

In particular, avoid simply redoing current labmate themes such as:

- DLM compositional-semantic emergence itself;
- typoglycemia acquisition in Pythia;
- onomatopoeia acquisition in Pythia;
- generic sentence-embedding analogy / inversion;
- generic multilingual idiom MI;
- the exact current patent-translation setup.

A candidate should use these as clues about promising *types of objects / methods*, then move to a genuinely different scientific question.

---

# 3. The three filters every candidate must pass

A candidate is not a project until it passes all three.

## Filter A — Real object / phenomenon

Prefer:

- a robust anomaly already observed in a real system;
- a repeated failure mode reported by multiple papers / practitioners;
- two published results that genuinely conflict on the **same scientific object**;
- a sharp reversal / phase transition / collapse / crossover in existing results;
- a long-standing scientific problem whose study was historically blocked by cost / labor and for which LLMs make a qualitatively new experimental regime possible.

Avoid starting from:

> Paper A finds X, Paper B finds Y, therefore perhaps X and Y obey a deeper law Z.

`X` and `Y` can both be real while `Z` is merely an attractive bridge invented by us.

## Filter B — Meaningful regime

A phenomenon on a toy object is not sufficient if the intended scientific claim requires a natural / realistic / non-trivial regime.

Before treating a candidate as healthy, ask:

- does the target model actually perform the task competently?
- does the relevant event occur often enough at instance level?
- is there an open model / dataset / checkpoint that instantiates the phenomenon without fishing?
- can the experiment be run in a regime large enough that a positive result matters?

## Filter C — Decisive identification

Prefer one clean contrast whose interpretation is nearly forced by the question.

A good first experiment often looks like:

```text
same object
same data / state / task
one meaningful factor changed
one primary observable
```

If interpretation requires a growing chain of matching rules, auxiliary probes, several prerequisite gates and exclusions of many alternative explanations, treat that as evidence that the question or construct may be unnatural.

---

# 4. Two primary search modes

## Mode A — Observed anomaly -> existing explanations -> decisive test

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

We first build an **anomaly inventory** rather than immediately packaging every observation as a project.

High-value anomaly shapes include:

- performance collapse and spontaneous recovery;
- non-monotonic scaling;
- stronger models behaving less human-like / less calibrated / less useful in a special task;
- capability/use dissociations already visible directly in behavior;
- phase transitions during training;
- sharp disagreement between methods that are supposed to measure the same object;
- a method helping strongly in one structurally related regime and hurting in another;
- aggregate success hiding a repeated local failure event;
- confidence / uncertainty failing precisely on the cases where it is supposed to arbitrate decisions;
- stable phenomena that reverse under a simple natural intervention.

## Mode B — Old problem -> LLM changes the research paradigm

This is a special advisor-favored route, but it should not be forced.

Look for scientific problems that existed **before modern LLMs** and were interesting, but progress was bottlenecked by:

- expensive expert annotation;
- large human panels / behavioral experiments;
- manual taxonomy construction;
- hand-built rules / lexicons / knowledge bases;
- laborious paper-by-paper literature review;
- one-off domain-specific pipelines;
- inability to generate enough controlled stimuli;
- inability to test many hypotheses / counterfactuals at scale.

The weak version is:

> Can an LLM replace the old classifier / annotator and improve accuracy?

Usually reject this.

The stronger question is:

> **Does the availability of LLMs make a previously impractical scientific question or experimental design possible for the first time?**

This route is especially attractive when the LLM is an **experimental instrument**, not merely the benchmarked object.

Examples of the desired shape (not claims that these are still open):

- old psychometrics required expensive field testing; synthetic learners make large-scale pretesting possible, which creates new scientific questions about whether synthetic populations preserve human response geometry;
- old expert-panel methods were constrained by panel cost; synthetic panels make scale trivial, which exposes the deeper issue of whether many LLM personas contain independent information;
- old meta-science required humans to read thousands of papers; LLMs may make semantic tracing of claim evolution through an entire citation graph possible;
- old behavioral science used dozens of hand-written stimuli; LLM generation may allow thousands of validated controlled stimuli and therefore tests of generality that were previously impractical.

---

# 5. Explicit archive lessons: what repeatedly killed our topics

The archived topics are not merely failed experiments. They are training data for topic selection.

## 5.1 Bridge-hypothesis failure

Two stable observables do not imply that their fine-grained structures correspond.

This killed the temptation to infer a deep law merely because two curves / profiles both look structured.

## 5.2 Mechanism without mechanism-level phenomenon

Do not explain a hidden failure event that is not actually occurring in the chosen system.

Aggregate degradation is not evidence that the specific local failure required by the story exists.

## 5.3 Non-identifying intervention

A seemingly intuitive intervention may change the task itself and therefore fail to distinguish the intended explanations.

If a cue / prefix / intervention simplifies the problem, behavioral recovery cannot automatically be interpreted as recovery of an old internal route.

## 5.4 Aggregate metric != mechanism

A full-sequence or aggregate score can strongly track a property even when the local causal signal required by the mechanism story is absent.

## 5.5 Shared global geometry

Two layer / checkpoint / time profiles can correlate because both vary along depth / time / scale without corresponding locally.

Always remove / model the obvious global axis before interpreting profile correlation as a law.

## 5.6 Toy-only effect

A clean, replicated effect is insufficient if a scientifically meaningful non-toy experimental object cannot be instantiated without model / data / prompt / configuration fishing.

**Phenomenon existence** and **meaningful-regime existence** are separate gates.

## 5.7 Natural-crossover scarcity

If the experiment depends on per-instance winner reversals / disagreement / crossover, establish that these cases exist at useful density **before** designing the full experiment.

Aggregate model differences do not guarantee paired common support.

## 5.8 Complexity smell

If every criticism adds another control, gate, matching rule or auxiliary measurement, revisit the question instead of patching the protocol forever.

A good question should not need six exclusions merely to become interpretable.

## 5.9 Obvious-positive problem

If the hypothesis being true would sound like "of course," the project is unlikely to satisfy the significance bar even if the experiment works perfectly.

## 5.10 One spectacular seed is not a phenomenon

Randomize arbitrary identities / mappings when the intended claim is general. Do not let one large seed create a project when the rest do not replicate.

## 5.11 Strong prerequisite + null explanatory intervention is a real null

If the motivating base phenomenon is huge and stable but the proposed explanatory intervention is near zero, accept that the explanation is wrong. Do not blame the testbed and start searching for a favorable model / mapping.

## 5.12 Do not rescue by model / layer / threshold search

A weak frozen primary result should not be converted into a new discovery by searching:

- another layer;
- another threshold;
- another model family;
- another prompt;
- another metric;
- another arbitrary subset.

A new experiment is justified only by a **new external observation** that changes the scientific premise.

---

# 6. Practical registration bar

A candidate should normally remain only in the search log until the following are all true.

### 6.1 One-sentence natural question

No probe / SAE / benchmark name required to explain it.

### 6.2 Strong external anchor

At least one of:

- replicated anomaly;
- strong published tension;
- old well-defined scientific problem with a clearly documented pre-LLM bottleneck.

### 6.3 Exact collision audit

Search the **exact question and experiment**, not only the broad topic, especially in 2025–2026 work.

### 6.4 Meaningful experimental object already identified

Model / dataset / checkpoint / corpus exists and is competent enough.

### 6.5 One decisive first plot

The first plot should be able to substantially raise or kill the project.

### 6.6 Positive-result excitement test

Assume the hypothesis is true. If the result would not be genuinely exciting, reject before coding.

### 6.7 Null-result informativeness test

Bad null:

> Maybe another model / layer / threshold would work.

Good null:

> In the exact regime where the motivating phenomenon is alive, the proposed factor does not matter enough to support the claim.

### 6.8 Resource fit

Estimate real engineering / rollout / training cost before calling the experiment cheap.

---

# 7. Candidate intake template

Do **not** create a numbered topic directory immediately.

First record candidates in search logs using this schema:

```markdown
## Candidate / anomaly name

### Historical / empirical anchor
What already exists? Exact paper / old problem / system / numbers.

### Why this is surprising
Why is this not the obvious expected behavior?

### Why it matters
What scientific or practical belief changes if we understand it?

### Existing explanations / old bottleneck
What does the literature already think, or what historically prevented the problem from being studied well?

### Exact open question
One sentence.

### Cheapest decisive contrast
What is the first experiment that could materially answer the question?

### Meaningful-regime check
Do we already have a competent, natural experimental object?

### Collision audit
Has the exact question / contrast already been done?

### Positive-result excitement test
Assume the cleanest positive result. Is it actually worth a paper?

### Complexity-smell check
How many independent controls are already required just to interpret G0?

### Verdict
KEEP / DEEP-AUDIT / HOLD / KILL.
```

A candidate graduates to its own numbered project only after the phenomenon, novelty, meaningful regime and identification all survive this audit.

---

# 8. Evidence sources to search

Search broadly rather than treating conference papers as the only source of insight.

High-value sources include:

- ACL / EMNLP / NAACL / EACL / COLM / TACL / ICLR / ICML / NeurIPS / AAAI;
- older NLP, information science, IR, HCI, cognitive science, education, psychometrics, scientometrics and meta-science literature;
- 2025–2026 arXiv when the topic is moving quickly;
- appendices and negative ablations;
- GitHub issues / discussions / released training logs;
- practitioner blogs and research-engineering posts;
- repeated anomalies visible in checkpoint curves / RL traces / evaluation traces;
- authors' limitations and unresolved empirical contradictions.

For the old-problem route, deliberately search **pre-LLM literature (often 1990s–2021)** first to identify what was expensive / impossible, then audit whether LLMs genuinely remove the bottleneck rather than merely add a new baseline.

---

# 9. Breadth policy

Do **not** collapse this search into Japanese linguistics merely because several lab projects involve language-specific phenomena.

The current breadth-first baskets are:

1. **training / learning dynamics** — checkpoint-rich models, phase changes, collapse / recovery, order / plasticity, non-monotonic capability;
2. **representation / embeddings** — geometric structure, information loss / leakage, compression, relation structure;
3. **knowledge / metacognition / hallucination** — known-vs-unknown behavior, confidence, retrieval / use mismatch;
4. **special tasks / question science** — quiz, clue structure, distractors, item design, human-AI arbitration;
5. **scientific documents / meta-science** — citation, claims, review, rebuttal, evidence synthesis;
6. **IR / evaluation science** — expensive human judgments, pooling, judge disagreement, evaluation validity;
7. **older resource-heavy NLP / information-science problems** — ontologies, semantic resources, expert coding, lexicography, structured domain text;
8. **behavioral / cognitive science where LLMs become instruments** — only when the LLM enables a new experiment rather than a superficial human simulation;
9. **domain-specific structured generation / translation** — only when the object creates a real scientific question beyond application accuracy.

No basket is privileged merely because it is fashionable.

---

# 10. Round procedure

Every search round should proceed in this order:

1. **Phenomenon mining** — collect observations; do not invent a project immediately.
2. **Historical audit** — for old problems, identify the actual pre-LLM bottleneck.
3. **Exact collision audit** — search the precise question aggressively.
4. **Meaningful-regime audit** — verify the object exists and is strong enough.
5. **Interestingness test** — assume the cleanest positive result and ask whether it matters.
6. **Identification sketch** — only now ask whether one clean contrast exists.
7. **Resource estimate** — real wall-clock / engineering / human cost.
8. **Only then design G0.**

The log must preserve killed ideas as well as survivors so tempting weak directions are not repeatedly rediscovered.

---

# 11. Central rule

> **Do not search for a hypothesis that might be true. Search for a real phenomenon that demands an explanation, or an old scientific bottleneck that LLMs genuinely make possible to remove.**

And then apply the advisor's final bar:

> **Even if it is true, is it important / surprising enough that we would actually be happy to discover it?**

Method comes later.
