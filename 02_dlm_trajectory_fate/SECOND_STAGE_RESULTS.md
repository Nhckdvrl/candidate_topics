# Stage 2 Run Results

Run date: 2026-08-21  
Code revision: `37d87fc` (latest `main` at execution time)  
Execution branch: `server/02-dlm-trajectory-fate-g1`

## Validation

- `pytest -q tests/test_stage2.py`: **3 passed**
- Python compilation and `bash -n run_stage2_4gpu.sh`: **passed**
- Locked geometry was unchanged: 64 steps, 128 generated tokens, temperature 0, strict boxed parser.

## G1-A — GSM8K holdout

The untouched GSM8K examples 1000–1318 were evaluated with LLaDA. The result was:

```text
AUDIT_ONE_DIRECTIONAL
```

The recovery cell was directionally positive; the overwrite cell was not. This audit is not the decisive confirmation test.

## G1-B — GSM1K surface preflight (historical first run)

The first 200 GSM1K examples were evaluated before hidden-state extraction. The original heuristic support requirement was at least 6 positives and 20 negatives for each locked task.

| task | positives | negatives | result |
|---|---:|---:|---|
| transient recovery (step 16, lead >= 4) | 4 | 26 | below old support gate |
| transient overwrite (step 4, lead >= 16) | 5 | 13 | below old support gate |

The historical runner returned:

```text
STOP_LOW_LOCKED_SUPPORT
```

and therefore did not launch the 1,205-example hidden confirmation or Dream. No step, layer, lead, parser, or sampler change was made.

## Protocol correction before any GSM1K hidden result

The 200-example stopping heuristic was subsequently audited and found to have inadequate power: under the G0 locked-cell event frequencies, even a perfectly stable phenomenon would pass that gate with only about 57% probability for recovery and 75% for overwrite. Therefore this historical stop is **not interpreted as a completed scientific negative**.

The corrected protocol is documented in [`STAGE2_PROTOCOL_REVISION.md`](STAGE2_PROTOCOL_REVISION.md). It keeps every scientific variable locked and changes only the support-decision sample size:

```text
old: first 200 GSM1K -> heuristic stop
new: all 1,205 GSM1K -> require >=25 positives and >=25 negatives per locked task
```

The full run stores the four preregistered steps and two preregistered layers once, checks surface support first, and fits hidden probes only if at least one task has adequate full-dataset support.

The large raw artifacts remain excluded by `.gitignore`.

## Retry — full GSM1K locked confirmation

Following protocol revision 1, the complete 1,205-example GSM1K run was performed without rerunning G1-A. The full-data support gate passed:

| task | positives | negatives |
|---|---:|---:|
| transient recovery (step 16, layer 25, lead >= 4) | 33 | 163 |
| transient overwrite (step 4, layer 28, lead >= 16) | 34 | 100 |

The positive control also passed (AUC 0.896 at both locked layers, with step-0 deltas above 0.04). The locked confirmation result was:

```text
FAIL_BOTH
```

Recovery had AUC 0.498, delta versus surface -0.135, and delta versus step 0 -0.017. Overwrite had AUC 0.434, delta versus surface -0.119, and delta versus step 0 -0.011. Both 97.5% bootstrap lower bounds for the confirmation margin were below zero.

This is a valid negative confirmation under the preregistered geometry, not a geometry failure. The retry runner therefore stopped before Dream, as required. The broad Topic 02 claim should be treated as falsified or sharply demoted; no new cell search was performed.
