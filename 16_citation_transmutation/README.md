# Topic 16 — Does Citation Turn Hypotheses into Facts?

**Status:** `REGISTERED / G0_NOT_RUN / IDENTIFICATION_HARDENED`

## Natural question

> **When the same scientific proposition is restated through a citation and no new supporting evidence has entered the citing paper's evidential basis, does its expressed epistemic certainty systematically increase?**

Memorable form:

> **Does a hypothesis become a fact merely by being cited?**

The key phrase is **no new supporting evidence**, not merely “the citing paper ran no new experiment.” A citing paper may legitimately become more certain because it relies on newer primary studies, replications, reviews, or meta-analyses. Those edges are not citation-alone transmutation.

## Why this topic is registered

The phenomenon is not invented from scratch.

- Greenberg (BMJ 2009), *How citation distortions create unfounded authority*, manually reconstructed a claim-specific biomedical citation network and documented **citation transmutation**: tentative statements could become asserted facts without new supporting data.
- Small et al. (2019) studied aggregate certainty and citation patterns, but did not hold a semantic proposition and evidence history fixed edge by edge.
- Sarol et al. (2025) showed that automated citation-distortion analysis is plausible, while fine-grained transmutation remains a difficult measurement problem.

Seed / neighbors:

- Greenberg 2009: https://www.bmj.com/content/339/bmj.b2680
- Small et al. 2019: https://doi.org/10.1007/s11192-019-03016-z
- Sarol et al. 2025: https://doi.org/10.1002/pra2.1281

## Exact G0 estimand

The unit is a citation edge, nested inside a semantic `claim_id` / citation cascade.

A row is primary-eligible only when all of the following hold:

```text
same_core_proposition == True
directly_supported_by_source == True
evidence_audit_complete == True
evidence_status == NONE
certainty_shift in {UP, SAME, DOWN}
```

### `same_core_proposition`

This intentionally **ignores epistemic modality** while keeping the scientific content fixed.

The following may be the same core proposition:

```text
"Treatment X may reduce marker Y."
"Treatment X reduces marker Y."
```

But changing subject/entity, relation, direction, population/condition, mechanism, or scope makes it a different proposition.

### Evidence provenance

`new_primary_evidence: bool` was removed because it did not identify the scientific question.

The frozen evidence label is:

```text
NONE              # complete audit found no new support beyond the cited source
OWN_PRIMARY       # citing paper adds its own primary evidence
EXTERNAL_PRIMARY  # other primary studies add new support
SYNTHESIS         # review/meta-analysis/synthesis adds new support
UNKNOWN           # incomplete or ambiguous audit
```

`NONE` is invalid unless `evidence_audit_complete == True`.

### Certainty measurement

The old arbitrary `0..1` certainty score was removed. Primary measurement is pairwise direction:

```text
UP
SAME
DOWN
UNKNOWN
```

The LLM judge sees only two claim statements in blinded order, without paper dates or source/citing identity. Each pair is judged twice with reversed presentation order. Disagreement becomes `UNKNOWN` and cannot enter the primary statistic.

### Primary statistic

Encode:

```text
UP   = +1
SAME =  0
DOWN = -1
```

For each `claim_id`, average its eligible edge scores. Then average across claims:

```text
claim-balanced net upward = mean_claim(P(UP) - P(DOWN))
```

This prevents one highly cited claim with dozens of edges from dominating a sample of otherwise independent claims.

Uncertainty is estimated by **cluster bootstrap over `claim_id`**, never iid edge bootstrap.

`secondary_with_new_support` is descriptive context, not a required superiority contrast. We do **not** require unsupported edges to drift more than evidence-adding edges; genuine evidence may quite reasonably produce a larger certainty increase.

## LLM as the experimental instrument

The LLM is not the research object. It replaces the historically expensive manual measurement loop.

```text
PMC / OpenAlex / full text
        ↓
resolve citation edge + exact citation context
        ↓
retrieve source statement supporting the cited proposition
        ↓
retrieve any newer supporting evidence available to the citing restatement
        ↓
LLM: same-core-proposition + source-support + evidence-provenance labels
        ↓
LLM: blinded pairwise certainty comparison, twice with reversed order
        ↓
human-audited high-precision measured edges
        ↓
g0_core.py
```

The critical rule is that an LLM cannot declare `evidence_status=NONE` from absence of text in a short prompt. The upstream retrieval/audit must be complete enough to justify that label. `llm_annotate.py` therefore forces incomplete audits to `UNKNOWN`.

### Run with our local OpenAI-compatible LLM

For a vLLM/SGLang service:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export MODEL=<served-model-name>

python 16_citation_transmutation/llm_annotate.py \
  --input 16_citation_transmutation/raw_edges.jsonl \
  --output 16_citation_transmutation/measured_edges.jsonl
```

The raw input contract is illustrated in `raw_edges.example.jsonl`. Each row must already contain retrieval-complete `source_context`, `citing_context`, and an `evidence_bundle` plus `evidence_audit_complete`.

Then score:

```bash
python 16_citation_transmutation/g0_core.py \
  --input 16_citation_transmutation/measured_edges.jsonl \
  --bootstrap 5000
```

`example_edges.jsonl` is a hand-written measured-schema smoke example and can be scored directly.

## G0 protocol

### G-1 — measurement calibration before discovery

Build a small human gold set containing clean matches and adversarial near-misses.

Required before treating LLM labels as measurements:

1. primary-eligibility precision (`same core + source support + evidence status`) should be at least `0.90` on the audit set;
2. certainty-direction agreement with human annotation should be at least `0.80` on determinate pairs;
3. known Greenberg-style transmutation examples must be recoverable;
4. if `NONE` cannot be assigned reliably without expert effort exploding, stop the project rather than adding more controls.

The LLM is allowed to abstain with `UNKNOWN`; precision is more important than coverage at G0.

### G0 — new independent claim cascades

Start with roughly:

```text
20–50 independent claim_ids
100–300 manually auditable citation edges
```

Primary report:

- number of eligible edges and independent claims;
- `UP / SAME / DOWN` counts;
- edge-weighted net upward shift (descriptive);
- **claim-balanced net upward shift (primary)**;
- 95% claim-cluster bootstrap CI.

### Survival / kill line

Promote only if all measurement gates pass and the new independent claims show a non-trivial positive unsupported shift. A practical G0 promotion bar is:

```text
>= 20 primary-eligible claim_ids
claim-balanced net upward >= +0.10
95% claim-cluster bootstrap lower bound > 0
```

Kill this formulation if the measurement object cannot be made reliable, the known phenomenon cannot be recovered, or a reasonably powered independent sample is centered near zero / downward after evidence provenance is fixed.

Do not rescue by selecting one hedge word, one journal, one claim family, or one favorable model.

## Why this is safer than earlier failed topics

- The phenomenon has a direct historical observation rather than an empty-cell bet.
- The object is observable text + citation edge + evidence provenance; no latent route must be inferred.
- The primary statistic is a direct within-proposition change, not correlation between broad profiles.
- The main alternative explanation—legitimate accumulation of evidence—is handled in the data contract rather than through an expanding post-hoc control stack.
- If claim identity or evidence provenance cannot be labeled at high precision, the project stops at measurement calibration.

## Method opening if G0 survives

A positive result immediately motivates a concrete method problem:

```text
citing claim
    ↓
resolve exact cited-source proposition
    ↓
trace evidence provenance
    ↓
compare epistemic strength
    ↓
flag unsupported amplification / transmutation
```

Possible follow-up contributions include an evidence-conditioned citation verifier, claim-provenance tracker, or scientific-writing guard that warns when wording becomes stronger than the evidence lineage supports.
