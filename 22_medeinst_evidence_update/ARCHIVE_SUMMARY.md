# Topic 22 Archive Summary — MedEinst Evidence Update

## Final decision

`ARCHIVED / MEASUREMENT_CANONICALIZATION_FAILURE / NO_SCIENTIFIC_VERDICT`

Topic 22 asked whether an exact MedEinst Bias Trap reflects failure to encode decisive counterfactual evidence or failure to use encoded evidence to update an established diagnosis.

The scientific distinction was **not falsified**. The local route stopped because the behavioral prerequisite never achieved the frozen measurement-support requirement after two principled, bounded measurement repairs.

## What succeeded

### Pair structure

G0a passed on all 5,383 released test pairs:

```text
valid pairs                  5383
malformed pairs              0
ground-truth flip rate       1.0000
age/sex match rate           1.0000
median changed-token frac    0.0726
p90 changed-token frac       0.2516
```

### Substantive Bias Trap signal

After fixing the original Qwen3 runtime/format issues, v2 already passed every substantive frozen gate on resolvable outputs.

V3 then rescored the exact same frozen v2 CoT outputs with an outcome-blind closed-label canonicalizer and again passed every substantive gate:

```text
control accuracy             0.4258   PASS
control-correct              109      PASS
Bias Trap count              43       PASS
Bias Trap Rate               0.3945   PASS
Wilson lower bound           0.3078   PASS
diagnosis transitions        14       PASS
```

Thus the final stop is not a null behavioral result.

## What failed

The frozen support gate required:

```text
invalid-output rate <= 0.10
```

Observed:

```text
v1 invalid rate = 0.8125
v2 invalid rate = 0.6250
v3 invalid rate = 0.3242
```

V3 resolved 111 previously unresolved branch outputs, but 32.42% of pairs still failed the support requirement.

Remaining unresolved branch counts:

```text
control unresolved_final = 64
trap unresolved_final    = 58
```

Final verdict:

```text
MEASUREMENT_CANONICALIZATION_FAILURE
NO_SCIENTIFIC_VERDICT
DIRECT_MODE_NOT_RUN
```

## Why v3 was legitimate but v4 is not authorized

V2 localized a new defect: free-form final diagnoses could not reliably be mapped to the benchmark's closed pathology vocabulary by exact/sub-string matching.

V3 was therefore constrained to a scoring-only repair:

- exact same frozen CoT outputs;
- no behavior regeneration;
- no model/sample/seed/threshold change;
- no ground truth or patient narrative visible to the mapper;
- explicit abstention;
- two fixed label orders with agreement required;
- self-mapping preflight before benchmark rescoring.

That repair substantially improved support but did not reach the frozen validity floor.

A further local repair would require additional degrees of freedom—aliases, mapper choice, prompt redesign, label-menu tuning, or threshold relaxation—after observing the same sample. That would be measurement optimization rather than a bounded defect correction.

## Failure type

Primary:

**Layer C — measurement / usable-support failure.**

More specifically:

**closed-label canonicalization remained insufficiently reliable after bounded repair.**

This is not:

- a failure to reproduce the Bias Trap signal on valid outputs;
- a scientific negative for MedEinst;
- evidence that encoding-vs-update is false.

## Why direct G0c was not run

Direct mode was preregistered as downstream of a healthy CoT prerequisite. Because G0b-v3 still failed measurement support, direct G0c was correctly not used as a rescue.

No mechanism analysis was run.

## Reusable lessons

1. **Separate scientific gates from support gates.** Strong substantive metrics do not compensate for unusable support.
2. **Diagnose measurement failures before repairing them.** V1 and V2 exposed distinct defects; each repair was tied to an explicit diagnosis.
3. **Output-preserving repair is preferable to behavioral regeneration.** V3 reused the exact frozen outputs.
4. **Repair budgets should be defect-based, not count-based—but they still need a hard end.** A newly localized defect can justify another repair; persistent failure after that bounded repair should stop the route.
5. **Closed-set benchmark evaluation can itself be a research-risk bottleneck when models answer in open vocabulary.** Exact scoring, semantic canonicalization, and outcome-blindness must be designed before mechanism work.
6. **Do not use downstream regimes as post-hoc rescue.** Direct mode remained unrun after the upstream support failure.

## Evidence

- final v3 commit: `285ea8b7530ca24f14b721046efd584be8668499`
- `G0_RESULTS.md`
- `artifacts/g0_behavior_cot_v3/summary.json`
- `VALIDATION_AUDIT.md`
- `MEASUREMENT_FAILURE_V2.md`
