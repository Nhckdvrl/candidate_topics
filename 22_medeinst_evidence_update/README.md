# 22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?

**Status: `ARCHIVED / MEASUREMENT_RUNTIME_FAILURE / NO_SCIENTIFIC_VERDICT`**

## Natural question

A classic explanation of the Einstellung effect is that an established solution becomes a mental set: decisive new evidence appears, but the reasoner remains trapped by the old interpretation.

For an LLM, a wrong counterfactual diagnosis leaves two different possibilities:

> **Was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?**

The scientific question remains legitimate. Topic 22 is archived because the frozen local measurement route did not produce a sufficiently valid behavioral object on which that mechanism distinction could be tested.

## Seed

ACL 2026 long paper: **MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**.

- Official repository: `zhui711/MedEinst`
- Dataset: `zhui711/MedEinst`
- Seed-supported model: `Qwen/Qwen3-14B`
- Released test set: 5,383 counterfactual pairs

Exact Bias Trap event:

```text
model(control) = control ground truth
AND
model(trap) = control ground truth
AND
trap ground truth != control ground truth
```

## G0a — pair structure passed

The full released test set supplied a clean structural object:

```text
valid pairs                  5383
malformed pairs              0
ground-truth flip rate       1.0000
age/sex match rate           1.0000
median changed-token frac    0.0726
p90 changed-token frac       0.2516
```

Verdict: `PAIR_STRUCTURE_OK`.

This established useful paired support, but did not yet establish a reliable model-level Bias Trap measurement.

## First G0b — invalid measurement

The first Qwen3-14B CoT run produced an `81.25%` invalid-output rate. The implementation used greedy thinking, a 1,024-token reasoning budget, and a mandatory custom `FINAL_DIAGNOSIS:` marker, so that run was correctly invalidated rather than interpreted scientifically.

One explicit measurement repair was then frozen before rerunning:

```text
thinking enabled
sampling: temperature=0.6, top_p=0.95, top_k=20
max_new_tokens=32768
token-level </think> separation
post-thinking answer only
conservative canonical-label extraction
same 256 pair IDs
same model / seed / scientific thresholds
```

## Repaired G0b — completed, still measurement-invalid

Commit recording the final run:

```text
2a6f9712bd5e799b237be455f79a5b24c648fc06
```

Frozen repaired results:

| Metric | Value | Gate |
|---|---:|---|
| control accuracy | `0.3555` (91/256) | pass (`>=0.35`) |
| control-correct count | `91` | pass (`>=50`) |
| Bias Trap count | `34` | pass (`>=20`) |
| Bias Trap rate | `0.3736` | pass (`>=0.30`) |
| Wilson lower bound | `0.2812` | pass (`>=0.20`) |
| diagnosis transitions | `12` | pass (`>=8`) |
| invalid-output rate | **`0.6250` (160/256)** | **fail (`<=0.10`)** |

All 256 control and trap thinking traces closed, and neither side hit `max_new_tokens`.

The remaining invalids were overwhelmingly `unresolved_final`:

```text
control unresolved_final = 109
trap unresolved_final    = 124
```

Therefore the frozen verdict is:

```text
MEASUREMENT_RUNTIME_FAILURE
```

No direct-mode G0c was run.

## Final interpretation

This is **not** a scientific negative for the MedEinst phenomenon and not evidence against encoding-vs-update failure.

The valid subset actually passed every substantive behavioral gate. The problem is that the selected CoT inference/scoring interface fails to yield a valid final diagnosis for 62.5% of the frozen pairs, so the exact event set is not trustworthy enough for downstream causal analysis.

The repository stops here because the project has already used its one principled measurement repair. Continuing with a third parser/prompt/extraction redesign would turn candidate validation into measurement tuning.

A future revisit is allowed only if there is a genuinely different external measurement object—for example, an official released response format / evaluator or another seed-supported setup that removes the unresolved-final problem without tuning on these outcomes. It must not simply be another local parser repair.

## Why direct mode was not used as a rescue

The frozen protocol required CoT G0b to pass before direct-mode G0c. Once repaired G0b returned `MEASUREMENT_RUNTIME_FAILURE`, running direct mode would change the prerequisite sequence after seeing the result.

Therefore direct mode remains unrun. It cannot be used post hoc to rescue Topic 22.

## Archive files

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — final decision and transferable lessons.
- [`G0_RESULTS.md`](./G0_RESULTS.md) — exact first and repaired G0b provenance.
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md) — identification and measurement audit.
- `artifacts/g0_behavior_cot/summary.json` — repaired frozen summary.
- `g0_pair_locality.py`, `g0_bias_trap_screen.py`, `run_g0.sh` — frozen implementation retained for provenance.

## Reusable lesson

> **A strong signal on the valid subset does not rescue a measurement with catastrophic invalid-output support. After one principled repair, persistent measurement invalidity is a reason to archive the route rather than keep tuning the interface until the desired phenomenon becomes measurable.**
