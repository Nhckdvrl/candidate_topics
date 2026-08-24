# Topic 22 Validation Audit

**Audit status: `ARCHIVED / G0b-v3 MEASUREMENT_CANONICALIZATION_FAILURE / NO_SCIENTIFIC_VERDICT`.**

## Claim hierarchy

Keep these levels separate.

1. **Pair structure:** released counterfactual pairs are sufficiently local/aligned for paired analysis.
2. **Behavioral phenomenon:** Qwen3-14B reproduces dense exact Bias Trap events under a valid CoT measurement.
3. **Mechanism-tractable regime:** the same event remains dense with thinking disabled and fixed answer sites.
4. **Mechanism:** decisive new evidence is encoded but fails to update the diagnostic state.

Topic 22 established only level 1 cleanly. Levels 2–4 were not licensed because the CoT measurement-support gate remained unhealthy.

## G0a — passed

Full test-set audit:

- 5,383 valid pairs;
- 0 malformed pairs;
- ground-truth flip rate 1.0;
- age/sex match rate 1.0;
- median changed-token fraction 0.0726;
- p90 changed-token fraction 0.2516.

Verdict: `PAIR_STRUCTURE_OK`.

## G0b v1 — invalidated correctly

The first local Qwen3-14B CoT run had three demonstrated defects:

1. greedy decoding in Qwen3 thinking mode;
2. only 1,024 new tokens;
3. mandatory custom `FINAL_DIAGNOSIS:` parsing.

Its 81.25% invalid rate was not interpreted scientifically.

## G0b v2 — runtime healthy, label interface unhealthy

V2 preserved model/sample/seed/scientific gates while repairing the Qwen3 runtime stack.

```text
control accuracy             0.3555  PASS
control-correct              91      PASS
Bias Trap count              34      PASS
Bias Trap Rate               0.3736  PASS
Wilson lower bound           0.2812  PASS
diagnosis transitions        12      PASS
invalid rate                 0.6250  FAIL
```

All control/trap thinking traces closed and no branch hit the 32,768-token ceiling.

The dominant unresolved counts were:

```text
control unresolved_final = 109
trap unresolved_final    = 124
```

Therefore v2 localized the remaining measurement defect to free-form diagnosis -> closed-label canonicalization.

## G0b v3 — bounded scoring-only repair

V3 was explicitly designed to avoid behavioral or outcome-driven tuning.

It reused the exact v2 `records.jsonl`; no CoT was regenerated.

For each unresolved branch, the fallback canonicalizer saw only:

```text
post-thinking final-answer text
+
closed benchmark labels
```

It did not receive:

- patient narrative;
- ground truth;
- case type;
- control/trap identity;
- paired branch information.

Additional guards:

- explicit abstention;
- two fixed deterministic label orders;
- accept only cross-order agreement;
- self-mapping preflight before benchmark rescoring;
- already-resolved v2 predictions preserved exactly.

The cached test split contained 46 distinct canonical labels, and the 46/46 self-mapping preflight passed under both orders.

## G0b v3 result

V3 resolved 111 previously unresolved branch outputs.

```text
invalid rate: 0.6250 -> 0.3242
```

Substantive metrics:

```text
control accuracy             0.4258  PASS
control-correct              109     PASS
Bias Trap count              43      PASS
Bias Trap Rate               0.3945  PASS
Wilson lower bound           0.3078  PASS
diagnosis transitions        14      PASS
```

Support metric:

```text
invalid-output rate          0.3242  FAIL  (required <=0.10)
```

Remaining branch failures:

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

Result commit: `285ea8b7530ca24f14b721046efd584be8668499`.

## Interpretation

The correct interpretation is deliberately narrow:

> The released pair structure is clean, and the scorable subset shows a strong Bias Trap signal, but the frozen local measurement interface cannot score enough of the same 256 CoT outputs reliably enough to establish a population-level behavioral prerequisite.

Do **not** conclude:

- MedEinst failed to reproduce scientifically;
- the Einstellung effect is absent;
- encoding-vs-update is false.

Do **not** promote the valid-subset signal to a mechanism object because support selection is too severe.

## Why no v4 local repair

V3 was already the bounded repair specifically justified by the defect v2 localized.

A further repair would introduce new researcher degrees of freedom after observing the same sample, including aliases, mapper model choice, prompt/menu design, semantic thresholds, or behavior regeneration.

Accordingly:

> **No further local parser/canonicalizer/prompt/model/order/threshold tuning is authorized on this result.**

A future revisit requires a genuinely new external evaluation interface or artifact whose validity is established independently of these 256 outcomes.

## G0c — not run

Direct mode was preregistered as downstream of a healthy CoT prerequisite.

Because v3 invalid rate remained above 10%, G0c was correctly not run as a rescue.

## Mechanism analysis — not run

No hidden-state probe, patching, steering, causal tracing, or encoding-vs-update conclusion is licensed.

Had all G0s passed, G1 would have required both:

1. a positive-control evidence-state manipulation check; and
2. a causal downstream test separating evidence representation from diagnostic updating without directly overwriting the answer.

That stage was never reached.

## Final status

```text
G0a: PASS
G0b-v1: MEASUREMENT INVALID
G0b-v2: MEASUREMENT INVALID; CANONICALIZATION DEFECT LOCALIZED
G0b-v3: MEASUREMENT_CANONICALIZATION_FAILURE
G0c: NOT RUN
G1: NOT RUN
NO SCIENTIFIC VERDICT
TOPIC22 LOCAL MEASUREMENT ROUTE ARCHIVED
```
