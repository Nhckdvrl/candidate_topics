# Master Topic Search — Current Candidate Status

**Authoritative status after Round 04 re-screen (2026-08-23).**

The individual candidate files preserve the audit state at the round when each idea was first promoted. This README records the **current** status after applying the stronger [`Selection Bar V2`](../SELECTION_BAR_V2.md):

```text
external support for the phenomenon
+ positive-result excitement
+ obvious method / intervention opening
+ clean identification
+ meaningful regime
```

The five provisional files are **not** all active survivors anymore.

---

## Current ranking

| Rank | Candidate | Current status | Excitement if positive | Method opening | Main concern |
| --- | --- | --- | --- | --- | --- |
| **1** | [Do Scientific Claims Become More Certain as They Are Cited?](./do_scientific_claims_become_more_certain_as_they_are_cited.md) | **STRONG_KEEP** | Very high | Very clear: evidence-conditioned citation/transmutation verifier; certainty-preserving scholarly writing guard | claim equivalence + evidence-status labels must remain reliable and simple enough |
| **2** | [Does Methodological Information Decay Along Shortcut-Citation Chains?](./does_methodological_information_decay_along_shortcut_citation_chains.md) | **KEEP_REFRAME** | High if framed as unreconstructible / silently divergent methods, not merely monotonic decay | Very clear: protocol-lineage resolver / shortcut-citation linter / missing-detail repair | exact inheritance and critical-method-unit extraction can become annotation-heavy |
| **3** | [Is Negative Behavioral Adaptation Intrinsically Harder?](./is_negative_behavioral_adaptation_intrinsically_harder.md) | **CHEAP_G0_ONLY** | Very high if matched inhibition gap survives | Clear: inhibition-aware implicit memory / negative-experience post-training | the causal axis is still inferred from a confounded benchmark comparison and may disappear when tasks are matched |
| — | [When Does a Fact Become Recallable?](./when_does_a_fact_become_recallable.md) | **KILL CURRENT FORMULATION** | Moderate | Weak in the current descriptive temporal-lag framing; nearest intervention is already being actively studied | new Jul-2026 continual fact-learning work directly studies recitation→use and shows diverse restatements reduce the gap 27.4→5.4 points |
| — | [Do Language-Model Memories Consolidate With Age?](./do_language_model_memories_consolidate_with_age.md) | **HOLD_FOR_EXTERNAL_EVIDENCE** | High | Clear if true: age-aware replay/editing/consolidation | classic representation→behavior bridge: timestamp existence does not provide evidence that acquisition age controls overwriteability |

---

# 1. STRONG_KEEP — Citation transmutation

## Positive headline

> **The epistemic status of the same scientific claim systematically inflates as it propagates through citations, even when no new evidence for that claim is added.**

This passes the excitement bar because it would establish at scale that scientific citation itself can create apparent certainty / authority rather than merely transmit evidence.

Crucially, this is **not** a new phenomenon invented by us:

- Greenberg (BMJ 2009) manually demonstrated `hypothesis → fact` citation transmutation in a 242-paper / 675-citation belief network;
- independent later case studies in ferritin and widely cited global irrigation statistics report analogous unsupported sharpening / transmutation;
- Sarol et al. (ASIS&T 2025) already show that LLM/NLP automation can partially reproduce Greenberg-style citation-distortion analysis, while finding the harder `invention` class remains difficult.

So the project asks whether an established phenomenon is **general**, not whether an elegant new bridge exists at all.

## Method opening

If large-scale transmutation is real, the lever is immediate:

```text
citing claim
→ exact cited source claim
→ supporting evidence
→ epistemic-status change
```

A method can flag:

- stronger certainty than the cited source supports;
- dead-end citations;
- claim drift;
- citation paths that amplify a claim without evidence.

Natural method families:

- **Evidence-Conditioned Citation Verifier**;
- **claim-provenance / epistemic-status tracker**;
- **certainty-preserving scholarly writing guard**.

Generic claim-evidence systems such as PaperTrail (CHI 2026) already exist, so our method cannot merely be another provenance interface. It must target **citation-to-source semantic and epistemic fidelity**, especially the distortion class that existing automatic work still finds difficult.

## Remaining danger

The current G0 has too many labels if implemented naively:

```text
same claim
+ certainty
+ new evidence / no new evidence
```

Before coding at scale, simplify the experimental object so that `no new primary evidence` is structurally obvious where possible, and validate high-precision claim identity on a small gold audit.

---

# 2. KEEP_REFRAME — Method transmission through shortcut citations

The original title asks whether information **decays with depth**. That monotonic law is still more speculative than we want.

The stronger and safer mother question is:

> **When a paper says a method was performed “as described previously,” can the method actually used be reconstructed from the cited lineage, and how often do undeclared implementation differences appear?**

This retains the special object while removing an unnecessary monotonic-decay hypothesis.

## Positive headline

> **A substantial fraction of scientific methods claimed to be inherited from prior work cannot actually be reconstructed from the cited lineage, or silently diverge from the method they cite.**

This is more exciting than merely showing that deeper chains contain fewer parameters.

The substrate is already strong:

- the 2024 PLOS Biology study finds shortcut citations in >90% of examined papers and directly documents inaccessible, incomplete, recursive and outdated method sources;
- journals frequently lack explicit policies for reporting modifications;
- recent publisher guidance already tells authors to avoid “as described previously” and instead state specific methods.

## Method opening

A positive phenomenon immediately motivates:

- **Protocol Lineage Resolver** — recursively trace the actual method source and align critical implementation units;
- **Shortcut-Citation Linter** — flag citations whose target does not contain a complete or matching procedure;
- **Method-Diff / Repair Generator** — surface missing or silently changed parameters and produce a self-contained patch for the current methods section.

Intern-Atlas (2026) shows that million-paper method-level evolution graphs are technically feasible, so generic method-lineage extraction is no longer itself the novelty. Our method opening must target **implementation fidelity / reconstructability**, not generic method genealogy.

## Remaining danger

If every method family requires new domain-expert definitions of “critical units,” the project violates the complexity / annotation bar. G0 must show that one structured domain supports reliable extraction with a modest gold audit.

---

# 3. CHEAP_G0_ONLY — Negative behavioral adaptation

This has perhaps the most exciting possible positive result:

> **Even with identical actions and matched experience, LLMs reliably acquire positive habits but fail to suppress actions that their own experience shows are bad.**

And the method opening is obvious:

- negative-experience-specific implicit memory;
- suppressive action memory / outcome-gated policy bias;
- post-training that explicitly teaches first-attempt behavioral inhibition after interference.

However, archive lessons force a downgrade.

ImplicitMemBench's headline `17.6% inhibition vs 75.0% preference` is real, but its illustrated comparisons change task family at the same time as inhibition/preference. Therefore our claim that **feedback sign itself** causes the gap is still an explanatory guess.

This is exactly the pattern that killed several archived projects:

```text
strong motivating phenomenon
≠
proposed explanatory axis is strong
```

There is also already an ICLR 2025 method paper explicitly devoted to learning from negative vs positive feedback. That does not answer implicit adaptation, but it means generic “learn better from negative feedback” is not a fresh method contribution.

### Decision

Keep only because the matched G0 is extremely cheap and decisive.

Do not build a project, mechanism analysis, benchmark expansion, or method until:

```text
same action space
same baseline preference
same event sequence
positive acquisition vs negative inhibition only
```

shows a large cross-model gap.

If the gap shrinks to a modest effect, archive immediately.

---

# Removed — When Does a Fact Become Recallable?

The candidate was initially attractive because `Empty Shelves or Lost Keys?` establishes a real encoded-but-inaccessible state and training checkpoints offer a natural axis.

Under the new bar, the **temporal-lag formulation is too descriptive and too guess-dependent**:

```text
final-model encoded/recall distinction exists
        ↓
maybe encoded→recall is a developmental stage during training
```

That is a cross-axis hypothesis, not evidence.

More importantly, **Can a Language Model Learn Facts Continually in Its Weights?** (Jul 2026) now directly studies the more actionable `recitation → usable knowledge` gap. It reports that broad/diverse restatements reduce this gap from `27.4` points to `5.4` points and substantially improve later retention.

So the most natural “then what?” — change the training data so stored facts become usable — is already being actively answered.

### Decision

`KILL CURRENT FORMULATION`.

Do not rescue by narrowing to one checkpoint family or fact relation. Re-enter only if a **new external anomaly** creates a qualitatively different question.

---

# Removed — Do Language-Model Memories Consolidate With Age?

`Fresh in Memory` provides a beautiful representation result: acquisition recency is linearly readable from activations.

But the proposed next claim is:

```text
recency is represented
        ↓
therefore acquisition age may determine overwrite resistance
```

This is precisely a **representation → behavior bridge without behavioral support**.

The result would be exciting if true and would leave a method opening (age-aware replay / knowledge editing / recency steering), but the archive shows repeatedly that this is not enough. We have no direct anomaly showing early- and late-learned equally strong facts actually differ in plasticity.

Recent continual factual-learning work is also rapidly mapping retention, access and interference under sequential fact writes, increasing the cost of betting on an unsupported age effect.

### Decision

`HOLD_FOR_EXTERNAL_EVIDENCE`.

Do not run the overwrite experiment simply because it is elegant. Promote again only if an external paper/log shows a robust age-dependent plasticity or interference pattern that the experiment can identify cleanly.

---

# Operational shortlist after re-screen

```text
STRONG_KEEP
1. citation transmutation at scale

KEEP_REFRAME
2. shortcut-citation method reconstructability / silent divergence

CHEAP_G0_ONLY
3. matched positive acquisition vs negative inhibition

OUT OF ACTIVE SHORTLIST
4. fact encoding→recall training lag
5. memory age→overwriteability
```

The target is **not to maintain five candidates**. The target is to keep searching until we have several candidates that all survive this stronger bar.