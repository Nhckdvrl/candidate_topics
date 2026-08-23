# Topic 22 Validation Audit

**Audit status: G0 logic accepted after hardening; model run still required.**

## Claim hierarchy

Keep these three levels separate.

1. **Pair structure:** the released counterfactual pairs are sufficiently local and aligned for later paired intervention.
2. **Behavioral phenomenon:** a seed-supported open model reproduces exact Bias Trap events, and those events remain dense in a fixed-position direct-answer regime.
3. **Mechanism:** the new discriminative evidence is internally represented but fails to causally update the diagnostic state.

G0 establishes only (1) and (2). It cannot by itself establish (3).

## Audit findings and fixes

### A. The original Qwen3-8B direct screen was not seed-faithful

The ACL paper reports open-model baselines under zero-shot CoT and provides a strong published reference for `Qwen/Qwen3-14B` (Baseline Accuracy 44.12%, Bias Trap Rate 54.19%).

**Fix:** G0b now uses Qwen3-14B + zero-shot CoT first. Direct answer is a later mechanism-feasibility gate, not a substitute for seed reproduction.

### B. Free-text reasoning could contaminate diagnosis scoring

A CoT may mention several diagnoses before choosing one. Searching the whole output for disease names can falsely classify a reasoning mention as the final prediction.

**Fix:** only the explicit final marker

```text
FINAL_DIAGNOSIS: <diagnosis>
```

is scored. If no canonical final diagnosis can be resolved, the sample is invalid rather than guessed.

### C. Stratified sampling distorted the benchmark prevalence

Diagnosis-balanced sampling is useful for analysis but not for reproducing a benchmark-level conditional rate.

**Fix:** G0b/c use one fixed random sample of benchmark pairs. The exact `case_id` list is written to both summaries, and `run_g0.sh` checks CoT and direct modes used the identical pair set.

### D. Pair locality was previously overinterpretable

A small text edit does not itself prove that the changed tokens are medically decisive evidence.

**Fix:** G0a is explicitly only an **alignment/intervention-feasibility** audit. It saves exact changed spans in `pair_diffs.jsonl`. Medical validity of the counterfactual flip comes from the MedEinst construction/validation, not from text-distance statistics.

### E. Variable-length CoT would recreate a mechanism-identification problem

If the Bias Trap exists only in open-ended reasoning traces, token-level comparisons require trajectory alignment and can quickly become another expanding-control project.

**Fix:** after seed-faithful CoT reproduction, G0c requires a dense exact Bias Trap subset on the same model and same pairs with thinking disabled and a fixed final-answer format. If direct mode is too weak, stop the simple mechanism route rather than probing arbitrary CoT states.

### F. CoT truncation could create fake invalid outputs

A 512-token generation cap could terminate Qwen3 reasoning before the required final marker.

**Fix:** CoT now has a frozen default of `1024` new tokens (direct mode `64`), and the actual budget is recorded in the summary.

## What positive G0s identify

If G0a+b+c all pass, the valid prerequisite statement is:

> The released MedEinst pairs are sufficiently aligned for paired analysis; Qwen3-14B reproduces a dense old-diagnosis persistence effect under the seed's zero-shot CoT regime; and a dense subset of the same phenomenon remains in a fixed-position direct-answer regime suitable for controlled internal intervention.

This still does **not** show that the new evidence was encoded.

## What G1 must do to distinguish encoding vs updating

A valid updating-failure result needs at least two ingredients:

1. **Evidence-state manipulation check:** in direct-mode trap inputs, the changed evidence must produce a measurable internal state that is absent/different in the paired control and is also present in correctly updated trap positive controls.
2. **Causal downstream test:** manipulating that evidence-related state must shift the diagnostic state/behavior away from the old control diagnosis without directly overwriting the answer representation.

A generic trap-vs-control linear probe is not enough: it could decode lexical edits, disease identity, or formatting. Same-pair / diagnosis-transition-matched intervention is preferred.

## Remaining mechanism risk

The changed span may contain several correlated lexical changes rather than one atomic evidence variable. If useful donor/recipient matching requires many post-hoc covariates, or if only a broad layer/token sweep produces rescue, the encoding-vs-updating distinction is not clean enough and the topic should stop.

## Current verdict

**RUN G0a -> G0b -> G0c in order.** Do not implement G1 until all three pass.