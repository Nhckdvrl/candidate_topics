# 21 — Where Does Long-Context Semantic Execution Break?

**Status: `ARCHIVED / STOP_UPSTREAM_SEED_NOT_REPRODUCED`**

Topic 21 was registered to test whether long-context position can selectively destroy semantic execution while lexical access to the same code remains intact, and then localize the failure to state formation, propagation, or readout.

The project had a mandatory seed-reproduction prerequisite before any custom mechanism G0.

## Final result

The exact official forced-sequential SemTrace run completed on the frozen local platform:

- model: `Qwen/Qwen2.5-Coder-7B-Instruct`;
- 80 functions, 800 contexts, position step 8, seed 42;
- 8,800 completed examples;
- all 11 target positions present.

Observed semantic accuracy was essentially zero at every position:

```text
0: 0.000
8: 0.000
16: 0.000
24: 0.000
32: 0.000
40: 0.000
48: 0.000
56: 0.000
64: 0.000
72: 0.000
80: 0.00125
```

Frozen prerequisite gates therefore failed:

```text
edge mean >= 0.30            observed 0.000625  FAIL
edge-to-middle drop >= 0.20  observed 0.000625  FAIL
```

Verdict: `UPSTREAM_SEED_NOT_REPRODUCED`.

Per protocol, the paired semantic-vs-lexical G0 was not run and no model/prompt/parser/context/seed rescue was attempted.

## Interpretation

This does **not** claim the seed paper is false generally. It says the exact local model/artifact regime selected for Topic 21 did not instantiate the prerequisite phenomenon, so there is no justified mechanism object for this repository.

The topic is terminal unless genuinely new external evidence changes the premise. It must not be listed as active merely because the scientific question remains interesting.

## Archive files

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — final decision and transferable lesson.
- [`G0_RESULTS.md`](./G0_RESULTS.md) — exact frozen run and metrics.
- [`VALIDATION_AUDIT.md`](./VALIDATION_AUDIT.md) — identification and validation contract.
- `g0_upstream_contract.py` — frozen seed gate.
- `g0_position_dissociation.py` — unrun paired mechanism-support screen retained for provenance.

## Reusable lesson

> **Artifact completeness is not reproduction. A mechanism project cannot proceed when the exact prerequisite phenomenon is absent on the system being analyzed.**
