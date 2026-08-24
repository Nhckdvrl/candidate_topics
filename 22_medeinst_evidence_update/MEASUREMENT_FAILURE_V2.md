# Topic 22 — G0b v2 Measurement Failure (Historical, Nonterminal)

This file preserves the result previously recorded as a terminal archive. It is **not** a scientific negative and is no longer the current Topic 22 status.

## V2 result

The repaired Qwen3-14B CoT run used the frozen 256 test pairs, seed `20260823`, Qwen3-recommended thinking sampling, and a 32,768-token ceiling.

All substantive gates on resolvable outputs passed:

```text
control accuracy             0.3555  pass
control-correct              91      pass
Bias Trap count              34      pass
Bias Trap rate               0.3736  pass
Wilson lower bound           0.2812  pass
diagnosis transitions        12      pass
```

But the frozen invalid-output gate failed:

```text
invalid rate = 160/256 = 62.5%   required <=10%
```

Diagnostics localized the failure:

```text
control unresolved_final = 109
trap unresolved_final    = 124
thinking closed          = 256/256 on both branches
hit max tokens           = 0 on both branches
```

So v2 did **not** fail because reasoning failed to terminate or because the token budget was too small. The unresolved branches contained post-thinking final-answer text that the deterministic exact/sub-string parser could not map onto the benchmark's closed 49-pathology label space.

Verdict at this stage:

```text
MEASUREMENT_RUNTIME_FAILURE   # historical v2 label
NO_SCIENTIFIC_VERDICT
```

## Why this does not exhaust the repair budget

The first repair fixed known generation and parser implementation mistakes. The v2 rerun then exposed a distinct, newly localized measurement defect: **semantic canonicalization from open diagnosis phrasing to a closed benchmark label set**.

A further repair is permissible only because v3 is outcome-blind and scoring-only:

- reuse the exact frozen v2 generations;
- do not change model, pair IDs, seed, prompt, decoding, or thresholds;
- do not inspect the patient case or ground truth when canonicalizing;
- map only post-thinking final-answer text to the 49 closed labels;
- allow abstention;
- require agreement under two frozen label orders;
- require a 49/49 canonical-label self-mapping preflight.

If that v3 measurement still leaves invalid rate above the frozen 10% support gate, the local measurement route should stop. If measurement is healthy but substantive gates fail, that becomes a scientific reproduction stop.

## Evidence

- original v2 recording commit: `2a6f9712bd5e799b237be455f79a5b24c648fc06`
- `G0_RESULTS.md`
- `artifacts/g0_behavior_cot/summary.json`
- v3 contract: `g0_recanonicalize_v3.py`
