# Do Scientific Claims Become More Certain as They Are Cited?

**Status:** `PROVISIONAL SURVIVOR — ROUND 02`

This is not a registered numbered Topic. It survived the first phenomenon / collision / data-path audit and is retained for a dedicated G-0 design pass.

---

## One-sentence question

> **Holding the scientific claim fixed, does its expressed certainty systematically increase as it propagates through citation chains when later papers add no new primary evidence for that claim?**

A more memorable version is:

> **Does a hypothesis become a fact merely by being cited?**

This is an old science-of-science question whose historical bottleneck was not lack of interest, but the cost of following the **same semantic claim** across many full-text papers and judging how each paper rephrased it.

---

## Why this question exists

### Historical seed: the phenomenon is real, not invented by us

Steven Greenberg's 2009 BMJ study, **[How citation distortions create unfounded authority](https://www.bmj.com/content/339/bmj.b2680)**, manually reconstructed a claim-specific biomedical citation network containing 242 papers and 675 citations.

Among the observed distortions was **citation transmutation**: statements about the same biomedical proposition changed from tentative hypothesis to asserted fact even when the citation chain had introduced no new supporting data.

The supplementary material gives the striking form directly:

```text
hypothesis / possibility
        ↓ citation
likelihood
        ↓ citation
fact
```

This is therefore not a speculative bridge between unrelated papers. The target phenomenon has a classic manually documented case.

### Old methodological bottleneck

To test whether this is a general phenomenon rather than a famous anecdote, one must repeatedly do all of the following:

1. identify a scientific claim;
2. find later papers that actually restate that same claim;
3. separate genuine semantic restatement from merely related discussion;
4. measure whether wording becomes more or less certain;
5. determine whether the later paper contributes new evidence for the same claim.

Historically, steps 2–5 were extremely expensive at scale.

Modern full-text corpora + citation graphs + LLM semantic matching make a qualitatively larger experiment plausible.

---

## Why existing work does not already settle it

### Aggregate certainty work exists

Small, Boyack & Klavans (2019), **[Citations and certainty: a new interpretation of citation counts](https://ideas.repec.org/a/spr/scient/v118y2019i3d10.1007_s11192-019-03016-z.html)**, found that hedging in biomedical citing sentences is inversely related to citation frequency and that later citations tend to be less hedged than early citations.

This is a major collision and means the broad statement

> “highly cited / later science is written more certainly”

is already occupied.

But that analysis does **not hold the proposition fixed**. A later citing sentence may discuss a different aspect of the cited paper, a method, or a genuinely better-supported result.

### Certainty classification already exists

Prieto et al. (2020), **[Data-driven classification of the certainty of scholarly assertions](https://pmc.ncbi.nlm.nih.gov/articles/PMC7182025/)**, demonstrates that scholarly assertions can be assigned meaningful certainty categories automatically.

That supplies a measurement component, not the longitudinal scientific answer.

### Automatic citation-distortion detection has begun

Sarol, Schneider & Kilicoglu (2025), **[Automatic Identification of Citation Distortions in Biomedical Literature: A Case Study](https://onlinelibrary.wiley.com/doi/10.1002/pra2.1281)**, uses modern NLP to replicate Greenberg's original biomedical case.

This is especially important: it establishes that automation is plausible, but it is still a **case-study replication of the original network**, not a broad claim-level test of whether certainty drift is a recurrent property of citation propagation.

### Collision boundary

The candidate survives only if its contribution remains:

> **claim-preserving, citation-edge-level, evidence-conditioned certainty drift across many independent claims**

—not generic hedging analysis, citation accuracy, or a classifier paper.

If a 2025–2026 study is found that already tracks hundreds/thousands of semantically identical scientific propositions along citation edges while conditioning on whether new evidence was added, kill this candidate.

---

## Exact scientific contrast

For a source paper containing claim `C`, consider a later citing paper that restates `C`.

The key split is:

```text
Edge A: same claim restated, NO new primary evidence for C
Edge B: same claim restated, NEW primary evidence for C
```

Measure:

```text
Δcertainty = certainty(citing restatement) - certainty(source statement)
```

The decisive phenomenon is not simply `Δcertainty > 0`.

It is whether **no-new-evidence edges** show systematic upward drift, especially across multi-hop chains.

If certainty rises only when new evidence accumulates, the alarming “citation alone turns hypothesis into fact” account does not generalize.

---

## Main conceptual attacks

### Attack 1 — “same claim” is the entire project

Semantic similarity is not enough. Two sentences can be topically similar while changing subject, population, direction, mechanism, modality, or scope.

A false claim match would manufacture apparent drift.

**Defense:** build a small high-quality human gold set for claim equivalence, including adversarial near-misses. Require a claim matcher to clear a strong precision threshold before any field-wide analysis.

### Attack 2 — certainty can legitimately increase

A later paper may add replication, stronger experiments, a meta-analysis, or independent evidence.

Without distinguishing evidence accumulation from rhetorical drift, a positive result is uninterpretable.

**Defense:** the primary analysis must separate `new-evidence` from `no-new-evidence` edges. This is not an optional ablation; it defines the scientific question.

### Attack 3 — hedging words are not certainty

Lexical markers such as `may` / `could` are useful but crude. Certainty can change through syntax and discourse without a simple keyword change.

**Defense:** use an established human-validated certainty rubric and compare an LLM semantic scorer with traditional hedging features. The LLM should be an instrument, not the source of the result.

### Attack 4 — selection bias in full-text corpora

PMC/OpenAlex/S2ORC availability is not uniform over disciplines or time.

**Defense:** first claim should be domain-scoped (probably open biomedical literature) and not pretend to be “all science.” Cross-domain replication is an extension, not a prerequisite.

---

## Cheapest decisive G-0

### Domain

Start with biomedical open full text because:

- the classic phenomenon was biomedical;
- PMC provides strong full-text and citation-context coverage;
- scientific claims and primary-evidence sections are relatively structured;
- the 2025 case-study automation provides a concrete reproduction target.

### Step 0 — reproduce the historical case

Before discovering anything new, run the pipeline on the Greenberg network / a reconstructable subset.

Required output:

- claim-equivalence precision;
- certainty ordering accuracy;
- whether known transmutation edges are recovered.

If the pipeline cannot recover the classic case reliably, stop.

### Step 1 — new claim cascades

Sample roughly `20–50` claim-centered citation cascades from open biomedical literature.

Prefer claims that begin in language explicitly marked as tentative and receive enough later restatements for a trajectory.

### Step 2 — human audit

Manually audit about `150–300` citation edges, stratified by predicted drift and evidence status.

This is small enough for a solo project but large enough to estimate precision of the critical labels.

### Primary figure

Do not start with a complicated graph neural network.

The first figure should simply compare the distribution of `Δcertainty` for:

```text
same claim + no new evidence
vs
same claim + new evidence
```

and then show multi-hop accumulation only if the one-edge phenomenon is real.

---

## Kill line

Kill or sharply downgrade if:

- high-precision claim equivalence cannot be achieved without extensive expert annotation;
- “new evidence for the same claim” cannot be classified reliably enough for the contrast;
- after holding claim identity fixed, certainty drift disappears;
- drift exists equally or more strongly on evidence-adding edges and therefore looks like ordinary epistemic updating;
- the phenomenon is confined to the original Greenberg case or one tiny topic.

Do **not** rescue the project by narrowing to a particular hedge word, journal, or citation phrase.

---

## Why a positive answer would be worth knowing

A strong result would say something much larger than “scientists use fewer hedges over time”:

> **Repeated citation itself systematically changes the epistemic status expressed for an unchanged claim, even when the literature has not added evidence for that claim.**

That would provide large-scale evidence for a classic information-cascade mechanism in science and directly affect how we interpret review articles, citations, and apparent consensus.

A strong negative answer is also valuable: it would show that Greenberg's famous “hypothesis→fact” story is not a generic law once semantic claim identity and evidence accumulation are controlled.

---

## Why this fits the advisor-style search

This candidate has the desired shape:

```text
old, distinctive scientific object
+ famous manual finding
+ historical labor bottleneck
+ LLM as a new experimental instrument
+ one crisp falsifiable question
```

The LLM is not the object being benchmarked. It enables a longitudinal science-of-science experiment that was previously painfully manual.

---

## Current verdict

`KEEP — HIGH PRIORITY, DATA/ANNOTATION RISK HIGHER THAN FACT-TRAJECTORY CANDIDATE`

The question is unusually memorable and long-lived. The main risk is not compute but identification quality. The next pass should therefore focus entirely on whether claim equivalence and evidence-addition labels can be made reliable on a small manually audited sample.