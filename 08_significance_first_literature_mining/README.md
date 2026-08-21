# Significance-First Literature Mining

**Status:** active literature-mining workspace; not a registered candidate topic.

This directory exists to separate **problem discovery** from method design.

The workflow is deliberately ordered as:

```text
scientific significance
-> concrete established anomaly / tension
-> exact collision check
-> minimal existence test
-> only then method search
```

The central advisor-style question is:

> **If the strongest plausible result were true, how happy should we be? What understanding would actually change?**

A topic is not promoted merely because it is novel, feasible, clean, or adjacent to a recent paper.

## Layer 0 — significance gate

Before designing experiments, every candidate must answer:

1. **Current view:** What does the field currently tend to assume or act as if it were true?
2. **Potential update:** If our result holds, what concrete belief, theory, training practice, or interpretation changes?
3. **Scope:** How much research or practice depends on that update?
4. **Non-obviousness:** After hearing the full story, would the result still be surprising rather than inevitable?
5. **Longevity:** Would the question remain interesting if today's model family / benchmark / training recipe disappeared?
6. **Three-times-so-what:** Can the result survive three successive `So what?` questions without collapsing into “we understand mechanism X a bit better”?

If these cannot be answered strongly, the item is kept only as a literature note, not promoted to a topic.

## What belongs here

- robust surprising / paradoxical observations in 2025H2–2026 work;
- contradictions between strong papers or adjacent fields;
- widely used assumptions challenged by new evidence;
- natural scientific questions for which modern AI creates a uniquely clean experiment;
- exact-neighbor papers that kill apparently attractive ideas;
- rejected ideas and why they are not significant enough.

## What does **not** qualify by itself

- an empty cell in a paper × variable grid;
- `representation A vs representation B` without a larger scientific consequence;
- a classic psychology effect merely re-run on an LLM;
- a method idea waiting for a question;
- a result whose strongest interpretation needs a long chain of exclusions;
- a technically clean effect that would still feel “理所当然” if confirmed.

## Files

- [`PHENOMENA_LEDGER.md`](./PHENOMENA_LEDGER.md): broad mining record of anomalies, tensions, and papers.
- [`SHORTLIST.md`](./SHORTLIST.md): only items that survive the current significance screen.
- [`REJECTED_OR_OCCUPIED.md`](./REJECTED_OR_OCCUPIED.md): attractive directions intentionally rejected, with collision/significance reasons.
- [`SEARCH_NOTES.md`](./SEARCH_NOTES.md): search scopes, query families, and literature neighborhoods checked.

## Promotion rule

An item may become a numbered candidate folder only after:

```text
A. significance is strong even before method design;
B. the motivating phenomenon/tension is supported by strong evidence;
C. the exact scientific question is not already occupied;
D. there is a simple existence test whose positive result would be genuinely exciting;
E. the selected AI system clearly instantiates the prerequisite phenomenon.
```

No method-heavy validation protocol is designed inside this directory unless an item first passes A–E.
