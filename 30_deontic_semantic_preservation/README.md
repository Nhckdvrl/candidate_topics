# 30 — Normative Semantics Preservation under Text Simplification

**Status: REGISTERED / G0 IMPLEMENTED / PUBLIC-DATA SUPPORT AUDIT NEXT.**

## Mother question

> When text is simplified for readability, does it preserve the normative force of what people must, may, may not, or are entitled to do—including the conditions and exceptions that license those rules?

This is not generic semantic similarity. Legal/policy text can remain topically similar while changing obligation, permission, prohibition, entitlement, scope, condition, or exception.

## Scientific object

Represent a normative proposition as:

`actor -> modality -> action -> condition -> exception`

Primary drift classes: obligation↔permission, prohibition dropped/softened, entitlement lost, condition deleted (`if`, `unless`, `subject to`, `only if`), exception deleted (`except`, `notwithstanding`, `other than`), and negation polarity changes.

## Existing gold anchor

LexDeMod releases agent-specific gold deontic labels and spans for contract text. Its public CSV schema contains a seven-way label vector plus annotated modality spans. This gives an independent validation anchor for the measurement before simplification outcomes are scored.

The simplification side should use SIMPLE-LAW if its aligned-pair artifact is available under a stable public contract; otherwise another public aligned legal/policy simplification corpus may be substituted only before any model outcome is inspected. The scientific variable is normative preservation, not the dataset name.

## Frozen P0a — deontic measurement receipt

Run `audit_lexdemod.py` on the released LexDeMod classification CSV. Gates:

- >=500 non-`none` gold rows;
- >=250 unique source clauses;
- >=99% parser validity.

The final extractor used for G0 must be calibrated against held-out LexDeMod gold; no simplification outcome is visible during calibration.

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

## Validation receipt

- Public LexDeMod schema checked: clause ID, text, 7-way label vector, and gold spans are present.
- Local measurement and end-to-end preflight fixtures passed; 4 total Topic29/30 unit tests pass.
- No simplification-model outcome was inspected during implementation.
- Next action: exact corpus-level support count, then G0a on the frozen eligible pool.
