# Archive Summary — Topic 17: Can a Cited Method Actually Be Reconstructed?

**Final status: ARCHIVED / G0 INVALID, MEASUREMENT FAILURE / COMPLEXITY-SMELL STOP**

Archived 2026-08-23. The natural question remains scientifically plausible, but
the proposed one-shot G0 could not distinguish genuine documentary loss from
ontology inapplicability and incomplete citation-graph recovery. Direct review
found no confirmed failure among the machine-flagged lineages, so the apparent
large effect is not a valid scientific result.

## 1. Original claim and gate

Topic 17 asked whether a method claimed to be performed “as described
previously” can be reconstructed from the current document plus its cited
lineage. The G0 froze two method families, 20 or more independent lineages,
evidence-bearing critical-unit labels, a Wilson-interval decision bar, and an
independent reliability audit.

The intended positive result was operational: a substantial cross-family rate
of genuinely unrecoverable units, not merely inaccessible files or generic
reporting omissions. The registered stop rule rejected a project in which every
paper required a bespoke ontology or expert reconstruction.

## 2. Data and pipeline

Candidates came from the open Standvoss et al. (2024) shortcut-citation
prevalence workbook. The deterministic selection used the first distinct-parent
candidates in two frozen families:

- immunostaining;
- western blot.

The resolver fetched JATS XML from Europe PMC with an NCBI PMC fallback, matched
the shortcut paragraph, resolved the first cited target, and extracted
family-relevant method paragraphs. Of 50 selected candidates:

| Resolution status | Count | Treatment |
|---|---:|---|
| cited target not in open PMC | 28 | excluded, not called failure |
| open target with relevant text recovered | 13 | inspected |
| open target without relevant text recovered | 8 | inspected but retrieval-limited |
| parent shortcut sentence not matched | 1 | excluded |

The 21 inspected lineages emitted 105 unit rows.

## 3. Measurement repairs

Two independently visible implementation errors were repaired before final
review:

1. the first preflight searched only the cited target, falsely calling units
   missing even when the current paper explicitly supplied them;
2. the target extractor originally searched only sections whose titles matched
   “methods/materials,” missing valid JATS paragraphs under other headings.

After both repairs, the machine-only result was:

```text
lineages                         21
machine-flagged lineages         14
machine-flagged fraction       0.667
Wilson 95% CI             [0.454, 0.828]
present_in_current units         59
omitted_but_recoverable units    22
lost_or_unrecoverable flags      24
formal verdict              INVALID
```

The formal verdict remained `INVALID`; machine labels were never represented as
the frozen independent human audit.

## 4. Direct evidence review

At the user's request, Codex directly reviewed every one of the 14 flagged
lineages. The review was explicitly recorded as a single AI-assisted evidence
review, not mislabeled as two human annotators.

| Direct disposition | Lineages | Meaning |
|---|---:|---|
| definite measurement false positive | 7 | non-applicable unit, regex miss, or wrong family assignment |
| unresolved documentary branch | 7 | multiple citations, recursive pointer, or supplement not exhausted |
| confirmed documentary failure | 0 | no retrieved case supported this label |

Examples of decisive failures included requiring `sectioning` for cultured-cell
immunofluorescence, requiring generic protein extraction for a purified-substrate
kinase assay, and treating a target that delegates its method to another
reference as the end of the lineage.

## 5. Why the project stops

The raw 14/21 rate cannot be interpreted as shortcut-method failure. A complete
measurement would first need to determine specimen- and assay-specific unit
applicability, traverse every relevant citation branch recursively, recover
publisher supplements, and then distinguish absent detail from legitimate
protocol variation. Those are not minor annotations around a clean statistic;
they are the entire research pipeline.

This directly triggers the registered complexity-smell stop. Adding more papers,
regexes, LLM judges, or post-hoc ontology rules would make the result more
bespoke without repairing identification. Topic 17 therefore does not advance.

The precise conclusion is:

> The current G0 is falsified as a clean measurement strategy. The broader
> conjecture that shortcut-citation methods can be unreconstructible remains
> untested.

## 6. Preserved artifacts

- `DIRECT_REVIEW.md` — line-by-line disposition of all machine flags;
- `MACHINE_PREFLIGHT.md` and `MACHINE_PREFLIGHT_RESULTS.json` — corrected raw run;
- `resolved_lineages.jsonl` and `machine_preflight_units.jsonl` — evidence-bearing intermediate data;
- `prepare_osf_candidates.py`, `resolve_pmc_shortcuts.py`, and
  `build_machine_preflight.py` — reproducible acquisition and preflight code;
- `g0_core.py` and `VALIDATION.md` — strict schema and frozen decision gate;
- `tests/` — validation and pipeline regression tests.

Do not rescue Topic 17 by adding method families, model judges, ontology rules,
or citation-depth sweeps. A future project would need a fundamentally different,
fully graph-complete and assay-specific measurement design.
