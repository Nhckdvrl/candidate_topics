# 30 — Normative Semantics Preservation under Text Simplification

**Status: REGISTERED / P0a AND P0b SUPPORT PASSED / GOLD-SPAN CONTROLLED G0 PASSED.**

## Mother question

> When text is simplified for readability, does it preserve the normative force of what people must, may, may not, or are entitled to do—including the conditions and exceptions that license those rules?

This is not generic semantic similarity. Legal/policy text can remain topically similar while changing obligation, permission, prohibition, entitlement, scope, condition, or exception.

## Scientific object

Represent a normative proposition as:

`actor -> modality -> action -> condition -> exception`

Primary drift classes: obligation↔permission, prohibition dropped/softened, entitlement lost, condition deleted (`if`, `unless`, `subject to`, `only if`), exception deleted (`except`, `notwithstanding`, `other than`), and negation polarity changes.

The representation itself is not claimed as novel. LexDeMod and newer deontic-temporal legal NLP systems already model closely related fields. The intended contribution is preservation under simplification, targeted failure measurement, and a preservation-constrained remedy.

## Existing gold anchor

LexDeMod releases agent-specific gold deontic labels and spans for contract text. Its public CSV schema contains a seven-way label vector plus annotated modality spans. This gives an independent validation anchor for the measurement before simplification outcomes are scored.

The simplification side should use SIMPLE-LAW if its aligned-pair artifact is available under a stable public contract; otherwise another public aligned legal/policy simplification corpus may be substituted only before any model outcome is inspected. The scientific variable is normative preservation, not the dataset name.

## Frozen P0a — deontic measurement receipt

Run `audit_lexdemod.py` on the released LexDeMod classification CSV. Gates:

- >=500 non-`none` gold rows;
- >=250 unique source clauses;
- >=99% parser validity.

The final extractor used for G0 must be calibrated against held-out LexDeMod gold; no simplification outcome is visible during calibration. The current regex implementation is only a support-audit and candidate-triage baseline. It does not recover agent-specific actor/action structure and must not be used as the paper's outcome labeler.

## Frozen P0b — simplification support audit

Run `audit_simplification_pairs.py` on aligned original/simplified pairs. Gates:

- >=300 deontic-eligible pairs;
- >=5% of corpus deontic-eligible;
- >=2 non-NONE modality classes represented.

This directly checks the real feasibility question: whether the simplification corpus actually contains enough normative structure.

## G0a — natural drift prevalence

On the frozen eligible pool, score modality/condition/exception/negation preservation and compare structured drift with ordinary lexical/semantic-preservation scores. The desired object is a nontrivial set of pairs that look generally faithful but alter normative structure.

## G0b — controlled perturbations

Create minimal paired rewrites from gold clauses in which one normative operator changes while lexical overlap remains high. Verify that general similarity/factuality metrics under-detect these changes relative to the structured scorer.

## Method runway

**Deontic-Structure-Constrained Simplification**: maintain

`actor -> modality -> action -> condition -> exception`

as invariants during rewriting, then optimize readability subject to preservation constraints. A stronger follow-up is an explicit normative-consistency critic/objective.

## Collision position

Legal simplification, legal meaning-preservation metrics, and deontic extraction already exist. The distinct question is whether **normative force is a systematically fragile semantic dimension under simplification**, which structures fail, and whether preserving those structures specifically fixes the problem.

## Public-data preflight receipt

- LexDeMod train/eval: 4,612 agent rows, 3,463 non-none rows, 1,470 unique clauses, 100% structural parser validity, and 387 multi-label rows. Test: 1,777 rows, 1,238 non-none, 477 unique clauses, 100% parser validity, and 162 multi-label rows. P0a passes.
- A stable public substitute corpus, Lex-Simple, provides two exactly aligned human reference files for 1,000 source lines. Across the two references, the support audit finds 446 deontic-eligible pairs (22.3%) spanning four target modality classes. P0b passes; there are 175 unique eligible source texts.
- The lexical extractor scores only 0.454 micro-F1 and 0.324 exact match on held-out LexDeMod for the four headline modalities. Its 82 Lex-Simple modality-change flags are therefore candidate retrieval, not a scientific prevalence estimate.
- A gold-span-controlled G0 produced 222 valid single-operator contrasts (100 obligation→permission, 87 permission→obligation, 35 prohibition-loss). MiniLM cosine averaged 0.986; 217/222 (97.75%) meaning-changing pairs still scored at least 0.95. TF-IDF cosine was at least 0.90 for 215/222 (96.85%).
- Code review and full interpretation are recorded in `REVIEW_AND_PREFLIGHT_RESULTS.md`.
- Full controlled-G0 receipt is recorded in `G0_RESULTS.md`.
- Next paper step: measure natural simplification drift with an actor-aware scorer plus human audit, and compare against stronger generic and legal-domain metrics. This is the main study, not another topic-killing gate.
