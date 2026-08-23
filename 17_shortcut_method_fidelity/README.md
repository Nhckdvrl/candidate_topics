# Topic 17 — Can a Cited Method Actually Be Reconstructed?

**Status:** `REGISTERED / G0_NOT_RUN`

## Natural question

> **When a paper says a method was performed “as described previously,” can the method actually used be reconstructed from the cited lineage, and how often do undeclared implementation differences appear?**

This is deliberately safer than the earlier speculative formulation “does information decay monotonically with citation depth?” We do **not** assume a depth law before observing it.

## Why this topic is registered

The empirical substrate is already real.

The 2024 PLOS Biology study *Shortcut citations in the methods section: Frequency, problems, and strategies for responsible reuse* examined hundreds of papers and found shortcut citations to be extremely common. Its manual citation-chain audit encountered inaccessible sources, recursively cited methods, insufficient details, and old references that may no longer match current practice.

Seed:

- https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002562

The open question is not whether method reporting is imperfect. It is whether a paper's **claimed procedural inheritance is actually reconstructible and faithful at the level of implementation-critical units**.

## Excitement test

The result worth a paper is:

> **A substantial fraction of methods claimed to be inherited from prior work cannot be reconstructed from the cited lineage, or silently diverge from the implementation they cite.**

That is stronger and more actionable than “methods sections omit details.”

A null is also decisive: if most shortcut-citation lineages remain reconstructible and declared modifications explain nearly all differences, this project stops.

## Method opening

A positive result directly creates a method problem:

```text
current methods statement
    ↓
resolve shortcut citation recursively
    ↓
extract implementation-critical units
    ↓
align current vs cited procedure
    ↓
flag missing / undeclared divergence
```

Possible later methods:

- **Protocol Lineage Resolver**;
- **Shortcut-Citation Linter**;
- **Method-Diff / Repair Generator** that surfaces missing fields and asks authors to state them explicitly.

The method opening exists because the failure is operational: a reader or reproducer cannot determine what was actually done.

## Why this is safer than our failed topics

- **Not a guessed monotonic law:** G0 asks reconstructible vs not reconstructible first. Citation depth is secondary.
- **No hidden mechanism claim:** units are directly extracted from documents and checked against cited sources.
- **Meaningful regime exists already:** the seed paper demonstrates real shortcut-citation chains and real recovery problems.
- **Complexity-smell stop:** if deciding inheritance and critical units requires bespoke expert interpretation for almost every paper, stop instead of building a giant annotation pipeline.

## G0: tiny lineage reconstruction pilot

Start with only `2–4` method families in one open biomedical / neuroscience subfield.

Construct roughly `20–50` genuine shortcut-citation lineages.

For each current paper, identify critical implementation units such as:

```text
material / reagent
concentration / dose
duration / timing
temperature
instrument / software version
preprocessing threshold
critical procedural order
```

Each unit is assigned one frozen status:

```text
same
explicitly_modified
omitted_but_recoverable
lost_or_unrecoverable
silent_divergence
```

### Primary paper-level outcomes

1. **reconstructible lineage** — every critical unit is either present, explicitly modified, or recoverable from the cited lineage;
2. **silent divergence** — at least one critical unit conflicts without an explicit modification statement;
3. **unrecoverable** — at least one required critical unit cannot be recovered from the lineage.

Citation depth is recorded but is **not** the primary hypothesis in G0.

### Survival conditions

Proceed only if:

- genuine procedural inheritance can be distinguished from historical/credit citations;
- critical units can be defined consistently within the selected method families;
- non-trivial unreconstructible or silent-divergence cases appear in independent lineages;
- human audit can validate extraction/alignment without per-paper expert reconstruction becoming the entire project.

### Kill conditions

Stop if:

- most apparent differences are explicit legitimate modifications;
- inheritance direction is usually unknowable;
- every method requires a different bespoke ontology;
- LLM extraction error is comparable to or larger than the observed phenomenon;
- the only finding is that some PDFs/supplements are inaccessible.

## Initial code

`g0_core.py` freezes a very small unit-level JSONL schema and computes paper/lineage outcomes.

```bash
python 17_shortcut_method_fidelity/g0_core.py \
  --input 17_shortcut_method_fidelity/example_units.jsonl
```

The code is intentionally independent of any extractor. Later LLM extraction must emit the same auditable unit records.
