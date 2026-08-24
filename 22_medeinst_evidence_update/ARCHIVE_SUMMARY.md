# Topic 22 Archive Summary — MedEinst Evidence Update

## Final decision

`ARCHIVED / MEASUREMENT_RUNTIME_FAILURE / NO_SCIENTIFIC_VERDICT`

Topic 22 asked whether an exact MedEinst Bias Trap reflects failure to encode decisive counterfactual evidence or failure to use that evidence to update an already established diagnosis.

The scientific question was not falsified. The project stopped because the frozen behavioral measurement route remained invalid after one principled repair.

## What worked

The released pair structure was excellent for controlled analysis:

```text
valid pairs                  5383
malformed pairs              0
ground-truth flip rate       1.0000
age/sex match rate           1.0000
median changed-token frac    0.0726
p90 changed-token frac       0.2516
```

G0a verdict: `PAIR_STRUCTURE_OK`.

The repaired Qwen3-14B CoT run also showed substantial Bias Trap signal on the subset for which a final diagnosis could be resolved:

```text
control accuracy             0.3555  pass
control-correct              91      pass
Bias Trap count              34      pass
Bias Trap rate               0.3736  pass
Wilson lower bound           0.2812  pass
diagnosis transitions        12      pass
```

Thus the stop is not "nothing happened."

## What failed

The frozen invalid-output gate required:

```text
invalid rate <= 0.10
```

The repaired run produced:

```text
invalid rate = 0.6250 = 160/256
```

All thinking traces closed and none hit the 32,768-token ceiling. The dominant failure was unresolved final-answer extraction rather than unfinished reasoning:

```text
control unresolved_final = 109
trap unresolved_final    = 124
```

Final verdict:

```text
MEASUREMENT_RUNTIME_FAILURE
```

The frozen protocol therefore stopped before direct-mode G0c.

## Why we archive instead of repairing again

The first G0b was legitimately invalidated because three concrete implementation defects were identified before interpreting the scientific result:

1. greedy decoding in Qwen3 thinking mode;
2. only 1,024 new tokens;
3. mandatory custom `FINAL_DIAGNOSIS:` output marker.

One explicit repair corrected those defects while keeping model, sample, seed and scientific gates frozen.

After that repair, invalidity remained catastrophic at 62.5%. A third local redesign of prompt/parser/extraction would no longer be a simple correction of the demonstrated defect; it would create substantial researcher degrees of freedom after observing the outcome.

Accordingly, this repository stops the measurement route rather than tuning until the phenomenon becomes convenient to score.

## Failure type

Primary:

**Layer C — measurement / usable-support failure after one permitted repair.**

Secondary process lesson:

**Complexity-smell / repair-budget exhaustion.**

The underlying MedEinst phenomenon and the encoding-vs-update scientific distinction remain unresolved by this local experiment.

## Reusable lessons

1. **Valid-subset strength does not compensate for catastrophic invalid support.** A high Bias Trap rate among the resolvable cases cannot establish a population-level mechanism object when most frozen examples are unscorable.
2. **Distinguish runtime termination from semantic-output validity.** Here every thinking trace closed and none hit the token budget; adding more generation budget would not address the observed failure.
3. **Allow one principled measurement repair, not an open-ended rescue loop.** After the identified defects are corrected, persistent invalidity is evidence that the selected measurement interface is a bad experimental object.
4. **Do not skip a failed prerequisite.** Direct mode was downstream of repaired CoT reproduction in the frozen protocol, so it was correctly not run as a post-hoc rescue.
5. **A scientific question may remain good even when a local implementation route is archived.** Revisit only with a genuinely new external measurement object, not a new parser tuned on the same failed run.

## Evidence

- final recording commit: `2a6f9712bd5e799b237be455f79a5b24c648fc06`
- [`G0_RESULTS.md`](./G0_RESULTS.md)
- `artifacts/g0_behavior_cot/summary.json`
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md)
