# Collision and Internal-History Policy

Date: 2026-08-24

This policy corrects two recurring search mistakes:

1. treating any nearby literature as a reason to kill a topic;
2. forgetting that this repository has already run and stopped closely related hypotheses.

Both errors waste topic-search effort.

---

# 1. External collision is not a zero-overlap test

AI/NLP is crowded. A viable ACL / EMNLP / NAACL topic is **not required to occupy an untouched field**.

The correct question is:

> **After the strongest neighboring papers are granted in full, is there still enough independent scientific narrative for our own main-conference paper?**

A neighboring paper does **not** kill a candidate merely because it studies the same broad area, variable, model family, cognitive phenomenon, or one component of our proposed story.

## Keep the candidate when the remaining paper can still have its own:

- title-level scientific question;
- main experimental contrast;
- main conclusion;
- behavioral characterization / boundary;
- mechanism or causal analysis;
- intervention / training implication;
- cross-setting or cross-model generalization.

A useful shorthand is:

```text
neighbor A establishes X
neighbor B establishes Y
our work asks whether X changes Y under a clean new scientific object
```

This can be a strong topic if the intersection itself supports a full paper. It is not automatically "just combining two papers."

## Kill for collision only when the nearest work already occupies most of the paper

Collision becomes terminal when the closest work has effectively already done the same:

1. core scientific question;
2. decisive experimental contrast;
3. title-level main conclusion;
4. principal mechanism/intervention story;

such that our remaining contribution is mainly:

- another model;
- another dataset/language;
- another probe;
- another minor control;
- a narrower parameter sweep;
- a small implementation variant.

The standard is **remaining narrative budget**, not absence of neighbors.

---

# 2. Internal history has higher authority than external brainstorming

Before promoting any candidate, search the numbered topic directories, root status index, archive summaries, and failure log for the same scientific object or identification route.

The precedence rule is:

```text
actual local result / ARCHIVE_SUMMARY
>
numbered-topic README
>
root STATUS_INDEX
>
advisor ACTIVE_CANDIDATES
>
ROUND search logs
>
new brainstorming
```

Round logs are historical search records. They must never resurrect a numbered topic whose later local experiment already stopped it.

## Internal stop categories must be respected precisely

### A. Substantive negative

Example: Topic 13.

The prerequisite repetition-damage phenomenon reproduced, but the frozen clustered-vs-even spacing contrast changed sign across locked replications.

Do not re-propose the same spacing explanation by changing schedules/models until it becomes positive.

### B. Prerequisite/platform failure

Example: Topic 21 SemTrace.

The official seed run completed on the frozen model/artifact regime but did not reproduce the prerequisite positional semantic effect. The custom mechanism G0 was therefore never justified.

Do not call SemTrace "executable" or active later simply because the conceptual question still sounds good.

### C. Conceptual identification failure

Example: Topic 05 Temporal Forgetting.

The broad storage-vs-access question remains scientifically meaningful, but prefix rescue changed the task and could not identify retained uncued competence.

A future topic may revisit storage-vs-access only with a genuinely new identification strategy. Reusing prefix/continuation rescue under a new name is forbidden.

### D. Measurement-invalid run

Example: Topic 22 MedEinst current G0b.

An invalid measurement is not a scientific negative. The frozen one-time measurement repair remains authorized; status must be `RERUN_REQUIRED`, not archived and not passed.

---

# 3. Mandatory internal-collision card before promotion

Every candidate that reaches the top pool must answer:

```text
Closest numbered/archived internal topics:
What exactly failed there?
Is the new candidate the same scientific claim?
Is the new candidate using the same identification route?
If related, what genuinely new premise or identification makes reopening legitimate?
```

If this cannot be answered, do not promote.

---

# 4. Reopening rule

An archived topic is not permanently forbidden as a broad domain. It may be revisited only when one of the following is true:

1. **new external evidence** establishes a premise that was previously absent;
2. **new experimental object** directly instantiates the target phenomenon;
3. **new identification strategy** removes the specific conceptual failure;
4. **new scientific question** is materially different even if it uses the same resource.

The reopening note must explicitly cite the archived topic and explain why the old stop does not apply.

What does **not** justify reopening:

- larger model;
- different seed;
- different layer;
- different prompt;
- different threshold;
- a new probe for the same failed route;
- changing the benchmark while keeping the same non-identifying intervention.

---

# 5. Final search principle

> **External overlap is normal; ask whether a full ACL/EMNLP/NAACL narrative remains. Internal failed experiments are not normal overlap; they are evidence and must constrain future search.**

The goal is neither "completely novel" nor "anything adjacent is fine." The goal is a paper-sized scientific object that survives both the literature and our own experimental history.
