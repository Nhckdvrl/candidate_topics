# 22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?

**Status: `ARCHIVED / MEASUREMENT_CANONICALIZATION_FAILURE / NO_SCIENTIFIC_VERDICT`**

## Natural question

A classic explanation of the **Einstellung effect** is that an established solution becomes a mental set: decisive new evidence appears, but the reasoner remains trapped by the old interpretation.

For an LLM, an exact counterfactual Bias Trap leaves two different possibilities:

> **Was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?**

That scientific question remains unresolved. Topic 22 stops because the frozen local behavioral measurement route never achieved sufficient valid support to authorize direct-mode or mechanism analysis.

## Seed and experimental object

Seed: ACL 2026 long paper **MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**.

- ACL: https://aclanthology.org/2026.acl-long.1847/
- official repository: https://github.com/zhui711/MedEinst
- dataset: https://huggingface.co/datasets/zhui711/MedEinst
- seed-supported model: `Qwen/Qwen3-14B`

The released test set contains 5,383 counterfactual pairs. The exact Bias Trap event is:

```text
model(control) = control ground truth
AND
model(trap) = control ground truth
AND
trap ground truth != control ground truth
```

## G0a — pair structure: PASSED

```text
valid pairs                  5383
malformed pairs              0
ground-truth flip rate       1.0000
age/sex match rate           1.0000
median changed-token frac    0.0726
p90 changed-token frac       0.2516
```

Verdict: `PAIR_STRUCTURE_OK`.

## G0b v1 — invalid measurement

The first Qwen3-14B CoT run had three demonstrated measurement defects:

- greedy decoding in thinking mode;
- only 1,024 new tokens;
- mandatory custom `FINAL_DIAGNOSIS:` parsing.

Invalid-output rate was 81.25%. This run is provenance only.

## G0b v2 — runtime repaired, canonicalization defect localized

V2 kept the same model, fixed 256 pair IDs, seed, Bias Trap definition and scientific gates, while repairing the Qwen3 thinking/runtime stack.

All thinking traces closed and no branch hit the 32,768-token ceiling. Every substantive gate passed on resolvable outputs:

| Metric | v2 | Frozen gate |
|---|---:|---:|
| control accuracy | 0.3555 (91/256) | >=0.35 |
| control-correct count | 91 | >=50 |
| exact Bias Trap count | 34 | >=20 |
| Bias Trap Rate | 0.3736 | >=0.30 |
| Wilson lower bound | 0.2812 | >=0.20 |
| diagnosis transitions | 12 | >=8 |
| invalid-output rate | **0.6250** | **<=0.10** |

The dominant failure was `unresolved_final`, localizing the measurement problem to free-form diagnosis text -> closed benchmark label mapping.

Historical record: [`MEASUREMENT_FAILURE_V2.md`](./MEASUREMENT_FAILURE_V2.md).

## G0b v3 — scoring-only canonicalization repair: FAILED SUPPORT GATE

V3 reused the **exact frozen v2 CoT outputs**. It did not regenerate behavior.

The semantic fallback canonicalizer was outcome-blind:

- input: post-thinking final-answer text + closed benchmark labels only;
- no patient narrative;
- no ground truth;
- no control/trap identity;
- explicit abstention allowed;
- two fixed label orders;
- mapping accepted only on order agreement;
- canonical-label self-mapping preflight passed completely on the 46 labels present in the cached test split.

V3 resolved 111 previously unresolved branch outputs and reduced pair invalidity substantially:

```text
invalid rate: 62.50% -> 32.42%
```

Final v3 metrics:

| Metric | v3 | Frozen gate |
|---|---:|---:|
| control accuracy | 0.4258 (109/256) | pass (>=0.35) |
| control-correct count | 109 | pass (>=50) |
| exact Bias Trap count | 43 | pass (>=20) |
| Bias Trap Rate | 0.3945 | pass (>=0.30) |
| Wilson lower bound | 0.3078 | pass (>=0.20) |
| diagnosis transitions | 14 | pass (>=8) |
| invalid-output rate | **0.3242 (83/256 pair-validity failure)** | **fail (<=0.10)** |

Remaining branch-level unresolved counts:

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

Recorded result commit: `285ea8b7530ca24f14b721046efd584be8668499`.

Artifact: `artifacts/g0_behavior_cot_v3/summary.json`.

## Why the route stops here

V3 was the deliberately bounded scoring-only repair after v2 localized the defect. It improved support substantially but still left invalidity at 32.42%, more than 3x the frozen 10% ceiling.

Continuing now would require additional researcher choices such as:

- more permissive semantic equivalence rules;
- diagnosis-specific alias tables;
- another mapper model;
- different label menus/order schemes;
- prompt changes or behavior regeneration;
- lowering the invalidity threshold.

Those would turn a measurement repair into open-ended measurement optimization on the same observed sample.

Therefore:

> **Stop the local Topic 22 measurement route. Do not interpret the stop as evidence against the MedEinst phenomenon or against encoding-vs-update failure.**

A future revisit is justified only by a genuinely new external measurement object or evaluation interface that removes the closed-label ambiguity without being tuned on this run.

## Direct G0c and mechanism work

Direct G0c was a frozen downstream prerequisite and was **not run** because G0b-v3 did not achieve healthy support.

No hidden-state probe, patching, steering, or causal mechanism experiment is authorized from the current data.

## Reusable lesson

A strong substantive signal on the scorable subset is not enough when the support mechanism remains unhealthy.

At the same time, measurement failure and scientific failure must remain distinct:

```text
substantive gates: PASS
measurement support: FAIL
=> archive measurement route, not scientific hypothesis
```

## Files

- `g0_pair_locality.py`
- `g0_bias_trap_screen.py`
- `g0_recanonicalize_v3.py`
- `tests/test_g0_helpers.py`
- `tests/test_g0_v3_canonicalizer.py`
- `G0_RESULTS.md`
- `VALIDATION_AUDIT.md`
- `MEASUREMENT_FAILURE_V2.md`
- `ARCHIVE_SUMMARY.md`

## Scientific invariant

> **Same released patient pair, same model and frozen case set: control is correct, trap ground truth flips, and the model persists on the old diagnosis. Measurement must be healthy before this event can justify mechanism analysis.**