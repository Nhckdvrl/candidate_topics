# Frozen G0 validation contract

## Identified claim

This G0 can establish whether the **documentary lineage** is reconstructible and
whether two documents contain a conflict that is not declared as a modification.
It cannot establish which procedure was physically performed in a laboratory.
Therefore the legacy status `silent_divergence` is reported as an **undeclared
documentary conflict**, not as proven implementation drift.

## Inputs

`--input` contains one adjudicated row per `(lineage_id, paper_id, unit_name)`.
Confirmatory rows must add these fields to the original schema:

```json
{
  "inheritance_kind": "procedural",
  "evidence_current": "verbatim location/quote in the current paper",
  "evidence_cited": "verbatim location/quote in the cited lineage",
  "failure_cause": "not_applicable"
}
```

Allowed failure causes are `content_missing`, `source_inaccessible`,
`ambiguous_pointer`, `undeclared_conflict`, and `not_applicable`.

`present_in_current` is a non-failure status for a critical unit supplied by the
current paper even when the cited text does not expose a matching unit. This is
necessary because reconstructibility is evaluated over the current document
plus its cited lineage; ignoring current-paper detail creates systematic false
failures.

`--audit` uses the same rows plus `annotator_id` and
`"annotator_type": "human"`. At least two human annotators must independently
label each audited unit. Machine/LLM labels are permitted only as preflight and
are rejected by the formal audit loader. At least 20 audited units must be failure
candidates so high agreement cannot be manufactured by a sea of easy `same`
rows. The scorer reports direct pairwise agreement as well as agreement with the
final adjudication; disagreements must be resolved transparently outside it.

## One-shot decision

- `INVALID`: fewer than 20 independent lineages, fewer than two method families,
  under 100 double-annotated units (including 20 failure candidates), evidence coverage below 95%, status agreement
  below 80%, or procedural-inheritance agreement below 90%.
- `SURVIVE`: at least 25% of lineages fail, the Wilson 95% lower bound exceeds
  10%, failures occur in at least two method families, and at least two failing
  lineages cannot be explained only by inaccessible sources.
- `KILL`: the Wilson 95% upper bound is below a 25% failure rate.
- `INCONCLUSIVE`: valid measurement whose interval overlaps the frozen bar.

Run with the repository environment:

```bash
.venv/bin/python 17_shortcut_method_fidelity/g0_core.py \
  --input adjudicated_units.jsonl --audit blind_audit.jsonl
```
