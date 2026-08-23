# Topic 17 direct evidence review

## Decision

**STOP — G0 invalid because the measurement fails.**

This review does not establish that shortcut-citation lineages are generally
faithful, nor does it establish the opposite. It establishes the result needed
for a project decision: the current G0's apparent positive signal is generated
by ontology and retrieval errors, so it cannot test the conjecture in one clean
shot.

After two mechanical corrections, the screen flagged 14/21 lineages. Direct
inspection found:

- **7 definite false positives** caused by non-applicable universal units,
  regex misses despite explicit current-paper detail, or wrong method-family
  assignment;
- **7 unresolved flags** requiring another cited branch, recursive citation, or
  supplementary methods before a loss can be asserted;
- **0 confirmed failures** on the evidence actually retrieved.

The seven unflagged lineages plus seven definite false positives give at least
14/21 lineages with no demonstrated failure. The other seven are missing-data
cases, not positive observations.

## Line-by-line disposition of machine flags

| Lineage | Direct disposition | Decisive reason |
|---|---|---|
| PMC7054421-cit8 | unresolved | whole-mount makes sectioning inapplicable; shortcut cites multiple references but resolver follows only the first |
| PMC7058069-cit4 | false positive | semi-intact cultured cells do not require sectioning; all applicable frozen units are in the current paper |
| PMC7060125-cit13 | false positive | neuron/cell cultures do not require sectioning; the current paper supplies the remaining protocol |
| PMC7065974-cit8 | false positive | cultured-cell immunofluorescence does not require sectioning |
| PMC7082075-cit1 | false positive | coverslip cell immunofluorescence does not require sectioning |
| PMC7104492-cit22 | unresolved | paraffin-section protocol points outside the recovered main-text method; supplement/next pointer was not resolved |
| PMC7141835-cit2 | unresolved | two prior protocols are cited and the retrieved target does not expose the required method chain |
| PMC7060108-cit11 | false positive | current text explicitly says membranes were probed with the named primary antibody; regex missed it |
| PMC7065909-cit11 | unresolved | transfer/blocking is not in recovered text, but target supplements/deeper pointers were not resolved |
| PMC7080823-cit3 | unresolved | cited target explicitly delegates the western-blot method to its reference 22; resolver stops one hop early |
| PMC7090383-cit14 | false positive | shortcut concerns mitochondrial isolation, not western-blot inheritance; family assignment is wrong |
| PMC7093963-cit9 | unresolved | primary antibodies are explicit, but detection detail may live in cited supplemental data not retrieved |
| PMC7136023-cit45 | false positive | specialized purified-substrate kinase assay makes generic protein extraction inapplicable; detection is stated as anti-ThioP western blot |
| PMC7196238-cit6 | unresolved | current paper supplies assay and antibodies, but target method/supplement chain was not recovered |

## Why this is a one-shot project kill

The registered stop rule says to stop when every method needs bespoke ontology
judgment or expert reconstruction becomes the pipeline. Both conditions occur
before any scientific effect can be measured:

1. `sectioning` is critical for tissue slices but nonsensical for cultured cells
   and whole-mount embryos;
2. `protein_extraction` is critical for lysates but not purified-substrate
   assays;
3. a shortcut paragraph may cite several targets, and a target may recursively
   delegate the method again;
4. main-text PMC XML is not the complete documentary lineage when methods live
   in supplements.

Therefore adding more candidates or model judges would increase annotation and
retrieval complexity without making the present statistic identifiable. The
honest outcome is to stop Topic 17 in its current form.

## Audit identity

The user requested direct review by Codex because independent human annotation
is not available. This document is deliberately labeled as a single AI-assisted
evidence review. It is not represented as the frozen double-human audit, and the
formal scorer correctly remains `INVALID`.
