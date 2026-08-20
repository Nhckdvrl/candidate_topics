# Stage 2 Run Results

Run date: 2026-08-21  
Code revision: `37d87fc` (latest `main`)  
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

## G1-B — GSM1K surface preflight

The first 200 GSM1K examples were evaluated before hidden-state extraction. The preregistered support requirement was at least 6 positives and 20 negatives for each locked task.

| task | positives | negatives | result |
|---|---:|---:|---|
| transient recovery (step 16, lead >= 4) | 4 | 26 | below support gate |
| transient overwrite (step 4, lead >= 16) | 5 | 13 | below support gate |

Overall status:

```text
STOP_LOW_LOCKED_SUPPORT
```

Per protocol, the run stopped before the 1,205-example hidden extraction and did not launch Dream. No step, layer, lead, parser, or sampler changes were made to rescue the result.

The large raw artifacts are intentionally excluded by `.gitignore`; the machine-local reports are under `artifacts/g1a_gsm8k_holdout/` and `artifacts/g1b_gsm1k_preflight/`.
