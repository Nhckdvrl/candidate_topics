# Topic 16 — Does Citation Turn Hypotheses into Facts?

**Status:** `REGISTERED / G0_NOT_RUN`

## Natural question

> **When the same scientific claim is repeated through citations without new primary evidence, does its expressed epistemic certainty systematically increase?**

Memorable form:

> **Does a hypothesis become a fact merely by being cited?**

## Why this topic is registered

This is not a bridge invented from two unrelated papers. The core phenomenon has already been manually observed.

- Greenberg (BMJ 2009), *How citation distortions create unfounded authority*, reconstructed a claim-specific biomedical citation network and documented **citation transmutation**: tentative statements could become asserted facts without new supporting data.
- Later work on scholarly certainty and automatic citation-distortion analysis shows that certainty and source fidelity can be measured computationally, but the general claim-level, evidence-conditioned phenomenon is not settled.

Seed / neighbors:

- Greenberg 2009: https://www.bmj.com/content/339/bmj.b2680
- Small et al. 2019, *Citations and certainty*: https://doi.org/10.1007/s11192-019-03016-z
- Sarol et al. 2025, automatic citation distortions: https://doi.org/10.1002/pra2.1281

## Excitement test

The result worth a paper is not “later papers use fewer hedge words.” It is:

> **For semantically unchanged claims, citation propagation itself produces systematic epistemic inflation even when the later paper adds no primary evidence for that claim.**

That would be a direct result about how scientific authority is created and transmitted.

If the phenomenon disappears once claim identity and evidence accumulation are controlled, stop: the famous hypothesis→fact story is not a general law.

## Method opening

A positive result immediately leaves a non-trivial method problem:

```text
citing claim
    ↓
resolve exact cited-source claim
    ↓
trace supporting evidence
    ↓
compare epistemic strength
    ↓
flag unsupported amplification / transmutation
```

Possible later method direction:

- evidence-conditioned citation verifier;
- claim-provenance / epistemic-status tracker;
- citation-aware scientific writing guard that warns when wording is stronger than the cited evidence supports.

The method is **not** the G0. First establish that the distortion is common enough to deserve fixing.

## Why this is safer than our failed topics

Repository lessons applied explicitly:

- **Not Topic 01-style empty-cell betting:** citation transmutation already has a direct historical observation.
- **Not Topic 05-style latent identification:** the primary object is observable text + citation edge + evidence status; no hidden “route” is inferred.
- **Not Topic 12-style profile correlation:** the first statistic is an edge-level change in the same claim, not a correlation between two broad profiles.
- **Complexity smell:** G0 is killed if “same claim” or “no new evidence” cannot be labeled at high precision with a small audit. We do not compensate with increasingly elaborate controls.

## G0: small, high-precision replication + expansion

### Step 0 — reproduce a known case

Before searching thousands of papers, reconstruct a small subset of the Greenberg-style network and verify that the annotation pipeline can recover known transmutation examples.

### Step 1 — annotate new edges

Start with only ~100–300 manually auditable citation edges from open biomedical full text.

Each row records:

```text
edge_id
source_claim
citing_claim
same_claim                 # human-validated Boolean
new_primary_evidence       # whether citing paper adds primary evidence for this same claim
source_certainty           # frozen ordinal/continuous rubric
citing_certainty
```

Primary subset:

```text
same_claim == True
and
new_primary_evidence == False
```

Primary statistic:

```text
Δcertainty = citing_certainty - source_certainty
```

Secondary contrast:

```text
same claim + no new evidence
vs
same claim + new evidence
```

### G0 survival conditions

Proceed only if:

1. claim-equivalence and evidence-status annotation can be made high precision without expert effort exploding;
2. the known historical transmutation cases are recoverable;
3. new independent claim cascades contain a non-trivial rate of upward certainty shifts on no-new-evidence edges;
4. the effect is not reducible to a single hedge word / journal / claim family.

### Kill conditions

Stop if:

- claim identity cannot be judged reliably;
- new-evidence status is too ambiguous for the core contrast;
- certainty drift vanishes once claim identity is fixed;
- upward drift occurs mainly when genuinely new evidence is added;
- only the original historical network shows the effect.

Do not rescue by narrowing to one hedge word or one journal.

## Initial code

`g0_core.py` intentionally contains no LLM judge. It freezes the **data contract and primary statistic** first.

```bash
python 16_citation_transmutation/g0_core.py \
  --input 16_citation_transmutation/example_edges.jsonl \
  --bootstrap 2000
```

Later extraction / retrieval code should produce this same schema rather than changing the outcome definition after seeing results.
