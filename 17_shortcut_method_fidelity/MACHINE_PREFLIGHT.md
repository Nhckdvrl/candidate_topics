# Topic 17 machine preflight result

## Bottom line

The automated run is complete, but the formal G0 is **INVALID**, not positive or
negative. After counting details supplied by the current paper, 14 of 21
machine-inspected lineages were flagged as potentially unreconstructible
(66.67%; Wilson 95% CI 45.37%--82.81%). Keyword matching is not capable of
establishing that a method-critical unit is truly absent.

## Frozen provenance and selection

- Candidate source: Standvoss et al. (2024) OSF prevalence workbook.
- Families fixed before resolution: `immunostaining` and `western_blot`.
- Candidate order: deterministic `(family, parent_pmcid, shortcut_number)`.
- Resolution sample: first distinct parent papers, up to 30 per family. The
  source yielded 30 immunostaining and 20 western-blot candidates.
- Parent and target text: JATS XML from Europe PMC, with NCBI PMC fallback.
- A cited target absent from PMC was excluded. It was not counted as inaccessible
  or methodologically deficient.

## Resolution accounting

| Status | Count | Scientific denominator |
|---|---:|---:|
| Cited target not in open PMC | 28 | excluded |
| Open target, family-relevant method text not recovered | 8 | included |
| Open target, relevant method text recovered | 13 | included |
| Parent shortcut sentence not matched | 1 | excluded |
| **Total** | **50** | **21** |

The 21 included lineages produced five frozen ontology units each, or 105 rows.
The corrected machine screen labeled 59 `present_in_current`, 24
`lost_or_unrecoverable`, and 22 `omitted_but_recoverable`. These are review
candidates, not adjudicated labels.

## Formal one-shot gate

The scorer's verdict is `INVALID` for exactly one reason: the independent
double-human audit is missing. All 105 units must be independently labeled by at
least two humans, including at least 20 failure candidates. Pairwise status
agreement must reach 0.80 and inheritance agreement 0.90. Only then can the
frozen `SURVIVE` / `KILL` / `INCONCLUSIVE` rule execute.

Direct review was performed instead at the user's request. It exposed decisive
measurement failures and triggered the project's stop rule; see
[DIRECT_REVIEW.md](./DIRECT_REVIEW.md). Generating a second set of model labels
would not repair those failures and is explicitly rejected by the formal audit
loader.

## Reproduction

```bash
.venv/bin/python 17_shortcut_method_fidelity/prepare_osf_candidates.py \
  --workbook 17_shortcut_method_fidelity/data_sources/dataset_all_fields_prevalence.xlsx \
  --output 17_shortcut_method_fidelity/osf_candidates.jsonl

.venv/bin/python 17_shortcut_method_fidelity/resolve_pmc_shortcuts.py \
  --candidates 17_shortcut_method_fidelity/osf_candidates.jsonl \
  --output 17_shortcut_method_fidelity/resolved_lineages.jsonl \
  --per-family 30 --workers 6

.venv/bin/python 17_shortcut_method_fidelity/build_machine_preflight.py \
  --lineages 17_shortcut_method_fidelity/resolved_lineages.jsonl \
  --output 17_shortcut_method_fidelity/machine_preflight_units.jsonl

.venv/bin/python 17_shortcut_method_fidelity/g0_core.py \
  --input 17_shortcut_method_fidelity/machine_preflight_units.jsonl \
  > 17_shortcut_method_fidelity/MACHINE_PREFLIGHT_RESULTS.json
```
