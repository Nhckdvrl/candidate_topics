# Does Methodological Information Decay Along Shortcut-Citation Chains?

**Status:** `PROVISIONAL SURVIVOR — ROUND 03`

This is not a registered numbered Topic. It survived a first old-problem / collision / identification audit and is retained for a small G-0.

---

## One-sentence question

> **When an experimental method is passed from paper to paper through “as described previously” shortcut citations, are critical implementation details faithfully transmitted, or do parameters and steps systematically disappear or silently change along the chain?**

Short form:

> **Do scientific methods degrade when they are transmitted by citation?**

The object is not generic citation quality. It is **method transmission fidelity**.

---

## Seed phenomenon: a real reproducibility problem already exists

The direct seed is the 2024 PLOS Biology meta-research study **“Shortcut citations in the methods section: Frequency, problems, and strategies for responsible reuse.”**

The paper examined more than 750 papers across neuroscience, biology, and psychiatry and found that more than 90% used shortcut citations: citations that stand in for a full description of a method.

It then manually traced shortcut-citation chains for 15 parent papers. Reviewers encountered several recurring problems:

- the cited material could not be identified or accessed;
- the cited method was difficult to locate;
- the cited source itself gave insufficient methodological detail;
- the source cited another source instead of describing the method;
- older shortcut sources might no longer reflect how the citing authors actually performed the method.

The authors explicitly note that following a chain is time-consuming and that **each additional step can amplify these problems**.

Crucially, the chain study was exploratory: the small expert-reviewed sample established *what kinds of problems occur*, not the prevalence or transmission dynamics of specific methodological details.

This gives a strong phenomenon-first foothold.

---

## The one-step rotation

The seed asks roughly:

```text
Can readers recover a method by following shortcut citations?
```

The proposed question moves one level deeper:

```text
When the same method is transmitted across citation generations,
what happens to its actual implementation information?
```

Instead of treating a citation chain only as a navigation/access problem, treat it as a **communication channel** for a structured procedure.

For a method with implementation units such as:

```text
reagent / material
concentration / dose
duration
temperature
instrument / software version
preprocessing threshold
critical step / ordering
```

ask whether each unit is:

```text
preserved
explicitly modified
omitted
silently inconsistent
```

from one generation to the next.

---

## Why this would matter

A shortcut citation implicitly asserts something close to:

> “The details needed to understand / reproduce this part of our procedure can be recovered from the cited source, except for modifications we explicitly state.”

If the scientific record behaves instead like:

```text
complete protocol
   ↓
partial restatement + citation
   ↓
older/modified restatement + citation
   ↓
method name with missing or inconsistent implementation details
```

then reproducibility failures can arise not only because papers are short or inaccessible, but because **methodological information itself drifts during scholarly transmission**.

That is a stable science-of-science problem independent of the current LLM generation.

---

## Why LLMs change the feasible experiment

Historically, large-scale study is painful because a researcher must repeatedly:

1. identify which exact method a shortcut refers to;
2. locate the corresponding passage in the cited paper / supplement / protocol;
3. extract method parameters and procedural steps;
4. align semantically equivalent parameters across heterogeneous prose;
5. determine whether a difference is a declared modification or an unexplained inconsistency;
6. continue recursively when the cited paper itself uses another shortcut.

The PLOS study solved this with expert manual inspection on 15 chains.

Modern LLM-based structured extraction can be used as an **experimental instrument** to propose protocol units and align them across papers, while human auditing is concentrated on the critical matches.

The contribution must not be “we made a method-extraction system.” The scientific result is the transmission pattern.

---

## Collision audit

### Direct seed is close but does not answer the proposed question

The 2024 PLOS Biology paper measures shortcut-citation frequency and manually catalogs problems encountered while following chains. It does not quantify parameter-level preservation / omission / modification as a function of citation generation across a large collection of method lineages.

### Reproducibility / methods-reporting literature is broad

Large literatures already establish that methods sections often omit essential details, that methodological reporting varies, and that indirect citations can make reproduction difficult.

Therefore claims such as:

> “methods are underreported”

or

> “shortcut citations can be bad”

are **not novel** and must never be the paper's headline.

### Protocol extraction and procedural NLP exist

Methods/protocol extraction, scientific IE and document parsing are also active. They can supply tools but are not the scientific object.

### Exact collision boundary

This candidate survives only as:

> **same-method, multi-generation transmission fidelity along shortcut-citation chains, measured at the level of critical implementation units and explicit vs silent modification.**

If a prior study is found that already performs this exact longitudinal parameter/step analysis over many scientific method lineages, kill the topic rather than narrowing it to one reagent, journal, or citation phrase.

The current Round-03 search did not find such a study.

---

## Main conceptual attacks

### Attack 1 — papers can legitimately modify methods

A difference from the cited source is not automatically corruption. The authors may intentionally adapt the method.

**Required defense:** distinguish:

```text
explicitly declared modification
vs
silent deviation / missing detail
```

The alarming quantity is not raw difference rate.

### Attack 2 — omission may be harmless shared knowledge

Some omitted parameters may be conventional defaults known to domain experts.

**Required defense:** begin with method families for which protocol papers or reporting standards specify a reasonably clear set of implementation-critical fields. Human experts should label which units are actually required to reconstruct the method.

### Attack 3 — citation direction is not necessarily inheritance direction

Authors sometimes cite the historical origin of a method for credit while implementing a modern variant learned elsewhere.

**Required defense:** G-0 must identify genuine **shortcut citations used as procedural pointers**, not arbitrary methods-section citations. Phrases such as “as described previously” help, but passage-level human validation is required.

### Attack 4 — LLM hallucination could manufacture drift

If extraction/alignment is unreliable, the project becomes an artifact of the measurement system.

**Required defense:** target high precision, not maximum scale. Build a human-audited gold subset and report extraction/alignment error separately from scientific drift. The first paper can be hundreds, not millions, of edges.

---

## Cheapest decisive G-0

### Domain / method choice

Start in one open biomedical / neuroscience subfield where:

- full-text methods and supplements are accessible;
- several standard methods recur across many papers;
- protocol structure has recognizable implementation parameters;
- citation chains are long enough to observe multiple generations.

Do not attempt “all science” first.

### Sample

Pick roughly `2–4` standard method families and construct about `20–50` genuine shortcut-citation chains.

For each chain, locate the oldest detailed methodological source reachable within a small hop limit and align its critical method units to every later paper.

### Structured unit table

For method unit `u`, paper generation `g`:

```text
present and same
present and explicitly modified
present but inconsistent without explicit modification
omitted locally but recoverable from cited source
lost / unrecoverable through the chain
```

### Primary figures

Keep the first experiment simple:

1. **critical-unit retention vs citation depth**;
2. **silent inconsistency rate vs citation depth**;
3. **probability that a chain still reaches an adequate implementation description vs depth**.

A single monotonic “information decay” curve would be much more informative than a complicated graph model.

### Human audit

Manually verify on the order of `100–300` parameter/step alignments, oversampling predicted omissions and silent changes.

---

## Strong positive

A strong result would look like:

> Even for the same identifiable experimental method, later shortcut-citation generations preserve progressively fewer implementation-critical details and accumulate a non-trivial rate of undeclared parameter/procedure divergence.

The strongest version would show that chain depth predicts transmission loss even after controlling for method family and publication age.

---

## Kill line

Kill or sharply downgrade if:

- genuine method inheritance cannot be distinguished reliably from citations given for credit;
- parameter/step alignment requires extensive expert annotation for every new paper;
- most apparent differences are explicitly declared legitimate modifications;
- chain depth does not predict any meaningful loss once paper age / method family are controlled;
- the project reduces to “some cited PDFs are inaccessible.”

Do not rescue it by narrowing to one journal or one phrase.

---

## Interestingness test

Assume the cleanest positive result:

> **Scientific procedures behave like lossy messages: as researchers repeatedly inherit a method through citation shortcuts, critical details are progressively dropped and silent variants accumulate.**

That is a concrete and memorable result about reproducibility and scientific communication.

A clean negative is also worthwhile: it would show that shortcut chains are mostly a retrieval inconvenience rather than a systematic mechanism of protocol drift, contradicting the stronger intuition suggested by the exploratory manual study.

---

## Why this fits the advisor-style search

It has the desired shape:

```text
specific unusual object: shortcut-method citation chains
+ established manual phenomenon
+ old labor bottleneck
+ LLM as scalable research instrument
+ one falsifiable scientific quantity: transmission fidelity
```

It is not tied to a fashionable model architecture and should remain meaningful years from now.

---

## Current verdict

`KEEP — PROVISIONAL TOP-5 CANDIDATE`

Strengths:

- immediately understandable;
- grounded in a strong prior manual study;
- genuinely old problem / new-instrument flavor;
- modest compute;
- clear negative result.

Main risk:

- method inheritance and critical-parameter equivalence may demand more domain expertise than expected.

The next step is a **tiny expert-audited reconstruction pilot**, not a large citation-corpus crawl.