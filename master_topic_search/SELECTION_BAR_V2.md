# Master Topic Search — Selection Bar V2

This file is authoritative for future `master_topic_search` rounds.

It adds two hard requirements that must be checked **before** a candidate is treated as a serious topic:

1. **Would a clean positive result actually be exciting?**
2. **If the phenomenon is real, does it expose a concrete method / intervention opening?**

A candidate that is merely clean, novel, or easy to test is not enough.

---

# 1. The five hard gates

## Gate 0 — Established substrate, not an attractive empty cell

Before proposing a new relationship, ask what is already known to exist.

Strong anchors:

- a robust anomaly reported across models / datasets / studies;
- a classic manual finding replicated in more than one setting;
- a repeated practitioner failure with concrete traces;
- a well-defined old scientific problem whose historical bottleneck is documented;
- a direct behavioral phenomenon from a close seed paper.

Weak anchor:

```text
paper A proves X
paper B proves Y
therefore maybe X and Y are related
```

or:

```text
paper proves property X exists in representations
therefore maybe X controls behavior Y
```

The repository has repeatedly shown that these bridges can be elegant and still simply not exist.

### Archive reminders

- Topic 01: a clean cross-paper empty cell did not imply the representation/behavior temporal decoupling existed.
- Topic 07: a real PI>RI phenomenon did not imply the proposed architecture axis mattered strongly.
- Topic 12: two stable profiles did not imply fine-grained correspondence.
- Topic 14: a huge motivating power-law effect did not imply temporal persistence caused it.

**Rule:** when the project lives or dies on a new explanatory relation, require independent evidence that makes that relation more than a plausible guess. Otherwise keep it as a cheap probe, not a full candidate.

---

## Gate 1 — Excitement / significance test

Assume the best possible clean result.

Ask:

> **Would a knowledgeable listener genuinely update their picture of the system or phenomenon?**

Good positive-result headlines have the form:

- a widely assumed process is systematically wrong;
- a familiar aggregate behavior has a qualitatively different cause;
- a scientific communication mechanism creates a large hidden failure;
- a capability boundary changes how systems should be trained or used;
- an old debated phenomenon is finally established at scale.

Bad positive-result headlines have the form:

- "this correlation is non-zero";
- "this curve appears slightly earlier";
- "this representation also predicts another variable";
- "this benchmark gap persists under one more control";
- "method X behaves somewhat differently on dataset Y".

### The advisor test

Before coding, finish the sentence:

> **If this is true, the exciting thing is that ________.**

If the blank cannot be filled with a claim stronger than the experimental detail, reject or reframe.

---

## Gate 2 — Method-opening test

A strong problem should naturally expose a lever.

Before G0, we do **not** need to know the final method. But we must be able to name the kind of intervention the phenomenon would motivate.

Ask:

> **If the positive result is true, what would we try to change?**

Healthy method openings include:

- a training objective that directly targets the discovered failure;
- a data-selection / curriculum variable identified by the phenomenon;
- an inference-time controller that detects the failure state and changes computation;
- a memory / retrieval mechanism aligned with the discovered bottleneck;
- an automatic verifier, guard, reconstruction system, or decision rule for a documented scientific-workflow failure;
- a representation intervention only when the representation has already been shown behaviorally / causally relevant.

Unhealthy answers:

- "then we can study the mechanism more";
- "then we can probe more layers";
- "then we can build a benchmark";
- "then perhaps someone could design a method";
- a method whose input requires the same unobservable construct that made G0 hard to identify.

### Strong form

The ideal paper trajectory is:

```text
real phenomenon
    ↓
clean diagnosis / causal factor
    ↓
obvious intervention target
    ↓
method that improves the same primary failure
```

A phenomenon-only paper can still be excellent, but for **our topic selection** it is lower priority unless the result is exceptionally important.

---

## Gate 3 — One-clean-contrast identification

The first experiment should ideally be:

```text
same object
same task / data / state
one scientifically meaningful variable changed
one primary observable
```

If the project requires:

```text
A vs B
but only after matching C
and controlling D
and excluding E
and defining F with an LLM judge
and filtering G
```

before the headline can be interpreted, invoke the archive's complexity-smell rule.

This is especially important when the candidate uses latent nouns such as:

- memory;
- route;
- strategy;
- awareness;
- consolidation;
- support;
- internalized knowledge.

Do not assume the verbal concept is operationally stable.

---

## Gate 4 — Meaningful regime and practical path

A positive toy result is not enough.

Require early evidence that:

- the phenomenon occurs at useful density;
- the system is competent in the relevant task;
- the data / experimental object already exists;
- the experiment does not require model / prompt / layer fishing;
- a first pilot is realistic in roughly 1–2 weeks;
- the path from G0 to a paper does not begin with months of data engineering.

---

# 2. The "then what?" test

For every candidate, write these four lines before it can enter the active shortlist:

```text
Positive headline:
If true, why should the field care?
Immediate lever exposed by the result:
Plausible method family that attacks that lever:
```

If the last two lines are vague, the topic is at risk of becoming:

> "We proved an interesting thing. Then what?"

That is now a **selection failure**, not something to repair after months of experiments.

---

# 3. Evidence-strength tiers

## Tier S — phenomenon already exists on the exact scientific object

Examples:

- multiple independent cases of citation transmutation;
- repeated same-system behavioral anomaly;
- a classic manual study plus modern independent replication.

Best starting point.

## Tier A — phenomenon exists, proposed factor has direct supporting evidence

Good candidate if G0 is clean.

## Tier B — phenomenon exists, but the proposed explanatory factor is only one plausible interpretation

Treat as a **cheap G0 lead**, not a full project.

This is where many archived topics failed.

## Tier C — seed paper contains representation X and we hypothesize behavior Y

Normally `HOLD` unless there is external behavioral evidence linking X and Y.

## Tier D — cross-paper empty cell / elegant analogy only

Normally `KILL` for this search process.

---

# 4. Method-space collision matters too

Collision audit must now search two things independently:

1. **Has the scientific question already been answered?**
2. **Has the obvious method opening already become crowded?**

A phenomenon can be novel while the only natural solution is already saturated.

In that case the project risks becoming:

```text
new diagnosis
→ obvious existing method
→ no real method contribution
```

This does not automatically kill a very important scientific result, but it lowers priority for our purposes.

---

# 5. Revised candidate statuses

Use these labels:

- `STRONG_KEEP` — phenomenon support, excitement, method opening, identification and regime all look healthy.
- `KEEP_REFRAME` — core object is good, but the current hypothesis / headline is too descriptive or guess-dependent; reframe before G0.
- `CHEAP_G0_ONLY` — potentially exciting and highly falsifiable, but the central explanatory relation is still a guess. Do not invest beyond the minimum prerequisite experiment.
- `HOLD_FOR_EXTERNAL_EVIDENCE` — attractive bridge, but archive lessons say not to bet a project on it yet.
- `KILL` — fails excitement, method opening, collision, identifiability, or meaningful-regime bar.

---

# 6. Final rule

The target is not:

> a hypothesis that is elegant and testable.

The target is:

> **a real problem whose positive result would be exciting, whose existence is already supported strongly enough that G0 is not a lottery ticket, and whose diagnosis naturally exposes something we can improve.**
