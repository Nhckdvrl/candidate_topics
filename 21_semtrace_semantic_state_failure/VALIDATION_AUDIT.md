# Topic 21 Validation Audit

**Audit status: G0 logic accepted after hardening; model run still required.**

## Claim hierarchy

The project must keep three claims separate.

1. **Seed reproduction:** the official forced-sequential SemTrace setup shows a strong edge-to-middle semantic drop on the chosen open model.
2. **Mechanism-support object:** on our locked same-item screen, lexical access survives while semantic execution fails after moving the target from early to middle context.
3. **Mechanism:** a later intervention identifies whether the failure occurs at state formation, state propagation, or final readout.

G0 tests only (1) and (2). It must never be reported as evidence for (3).

## Audit findings and fixes

### A. Seed reproduction was previously implicit

A custom generator alone cannot distinguish "the seed phenomenon failed locally" from "our generator did not instantiate it".

**Fix:** `g0_upstream_contract.py` now requires an official `fsyn_output_prediction` reproduction summary on the same Qwen2.5-Coder-7B-Instruct model before our paired screen runs.

### B. Target placement was previously block-centered, not token-centered

Different helper functions have different tokenizer lengths, so splitting the distractor list in half is not a valid middle-position intervention.

**Fix:** the new context builder searches candidate insertion sites and selects the one whose target-token center is closest to 0.5. The frozen contract requires middle target center in `[0.40, 0.60]` and early target center `<=0.12`.

### C. Local-neighbor interference was a real confound

If the target has different immediate helper functions next to it at the edge and middle, a same-item failure could be attributed to local distractor interference rather than long-range position.

**Fix:** the target is now moved as a fixed three-block package:

```text
[guard_before, target, guard_after]
```

The same two immediate neighbors surround the target in both conditions. Only distant prefix/suffix context changes. A neighbor digest is recorded for every pair.

### D. Formatting failures could masquerade as semantic failures

A malformed answer is not evidence that operational computation failed.

**Fix:** the critical cell requires the middle semantic output to parse as an integer list of the expected length and nevertheless be numerically wrong. The aggregate invalid-output rate must be `<=0.10`.

### E. Lexical accessibility needed to be bilateral

Middle lexical success alone does not show that the lexical task itself is stable across the intervention.

**Fix:** the eligible cell requires lexical correctness at both start and middle, and both aggregate lexical accuracies must be `>=0.80`.

## What a positive G0 identifies

With the above contracts, the strongest valid G0 statement is:

> For a dense subset of same-program examples, changing only long-range context position while preserving the target, distractor multiset/order, immediate target neighbors, model, and task causes exact operational execution to fail even though the queried target assignment remains lexically retrievable.

This supports a position-sensitive semantic-computation failure object. It does **not** yet identify which internal computation is responsible.

## Remaining mechanism risk

A future probe that decodes intermediate state is insufficient. Topic 20 already demonstrated that a highly decodable coordinate can be causally inert.

G1 should therefore be frozen around:

- exact intermediate-state labels from the program generator;
- a very small predeclared set of state-bearing token sites/depth fractions;
- same-item edge-success -> middle-failure activation patching;
- a manipulation check that the intended intermediate state actually changes;
- behavioral rescue as the causal endpoint.

If meaningful rescue requires a broad site/layer/coefficient search, stop rather than convert the project into post-hoc circuit fishing.

## Current verdict

**RUN G0.** The prerequisite logic is now sufficiently clean to justify a local run. Do not implement G1 until both the official seed gate and exact paired critical-cell gate pass.