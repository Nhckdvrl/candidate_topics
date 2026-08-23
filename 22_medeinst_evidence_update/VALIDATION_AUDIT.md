# Topic 22 Validation Audit

**Audit status: G0a passed; first G0b run invalidated by measurement bugs; repaired G0b v2 ready for rerun.**

## Claim hierarchy

Keep these three levels separate.

1. **Pair structure:** the released counterfactual pairs are sufficiently local and aligned for later paired intervention.
2. **Behavioral phenomenon:** a seed-supported open model reproduces exact Bias Trap events, and those events remain dense in a fixed-position direct-answer regime.
3. **Mechanism:** the new discriminative evidence is internally represented but fails to causally update the diagnostic state.

G0 establishes only (1) and (2). It cannot by itself establish (3).

## Current empirical status

G0a passed on the full released test set:

- 5,383 valid pairs;
- 0 malformed pairs;
- ground-truth flip rate 1.0;
- age/sex match rate 1.0;
- median changed-token fraction 0.0726;
- p90 changed-token fraction 0.2516.

The first Qwen3-14B G0b run produced 81.25% invalid pairs. That run is **measurement-invalid** and its previous scientific-stop label is withdrawn.

## Audit findings and fixes

### A. The original Qwen3-8B direct screen was not seed-faithful

The ACL paper reports open-model baselines under zero-shot CoT and provides a strong published reference for `Qwen/Qwen3-14B` (Baseline Accuracy 44.12%, Bias Trap Rate 54.19%).

**Fix:** G0b uses Qwen3-14B + zero-shot CoT first. Direct answer is a later mechanism-feasibility gate, not a substitute for seed reproduction.

### B. The first Qwen3 thinking run used an invalid decoding regime

The first local G0b used greedy decoding with `enable_thinking=True`.

The official Qwen3 model card explicitly recommends thinking-mode sampling (`temperature=0.6`, `top_p=0.95`, `top_k=20`) and warns against greedy decoding because it can degrade performance and cause pathological repetition.

**Fix:** repaired G0b v2 uses exactly those Qwen3-recommended thinking settings. The scientific model/sample/gate are unchanged.

### C. The first reasoning budget was far too short

The first local run used only 1,024 new tokens. Qwen3's official thinking example allows up to 32,768 new tokens and separates thinking from final answer at the `</think>` token.

A sample still inside reasoning at token 1,024 cannot be treated as a wrong or unparsable diagnosis.

**Fix:** CoT budget is now 32,768 new tokens. The evaluator records `hit_max_tokens`, `thinking_not_closed`, and token counts explicitly.

### D. Free-text reasoning and answer extraction were conflated

Searching the whole CoT for disease names is invalid because reasoning may discuss several diagnoses. Conversely, requiring our custom literal `FINAL_DIAGNOSIS:` marker made valid free-text final answers appear invalid.

**Fix:** repaired scoring first separates Qwen3's post-`</think>` final-answer content at token level. Only this final segment is eligible for scoring. The preferred marker is accepted but no longer mandatory. Conservative canonical-label extraction also accepts an unambiguous final diagnosis phrased naturally. No LLM judge or semantic fuzzy matcher is used.

### E. Stochastic thinking needs reproducibility

Switching from greedy to correct thinking-mode sampling introduces sampling variance.

**Fix:** every `case_id` receives a deterministic pair seed derived from `(global_seed, case_id)`. Control and trap branches use the same pair seed (common random numbers). Re-running the same experiment is therefore reproducible while respecting Qwen3's recommended thinking mode.

### F. Invalid outputs now have scientific meaning

Previously all unresolved examples collapsed into one `invalid` bucket.

**Fix:** v2 records branch-level failure reasons:

- `hit_max_tokens`;
- `thinking_not_closed`;
- `unresolved_final`.

If pair-level invalid rate remains above 0.10, the verdict is now `MEASUREMENT_RUNTIME_FAILURE`, not `SEED_PHENOMENON_NOT_REPRODUCED`. A scientific negative is assigned only when the measurement is healthy and the substantive frozen gates fail.

### G. Stratified sampling distorted benchmark prevalence

Diagnosis-balanced sampling is useful for analysis but not for reproducing a benchmark-level conditional rate.

**Fix:** G0b/c use one fixed random sample of benchmark pairs. The exact `case_id` list is written to both summaries, and `run_g0.sh` checks CoT and direct modes used the identical pair set.

### H. Pair locality must not be overinterpreted

A small text edit does not itself prove that the changed tokens are medically decisive evidence.

**Fix:** G0a is only an **alignment/intervention-feasibility** audit. Medical validity of the counterfactual flip comes from MedEinst construction/validation, not from edit distance.

### I. Variable-length CoT remains a bad mechanism object

Even if repaired G0b reproduces the seed, open-ended CoT trajectories are not our intended mechanism substrate.

**Fix:** G0c still requires a dense Bias Trap subset on the same model and same exact pairs with thinking disabled. If direct mode is too weak, stop the simple mechanism route rather than probing arbitrary CoT states.

## Frozen repaired G0b contract

Unchanged scientific choices:

- model: `Qwen/Qwen3-14B`;
- split: MedEinst `test`;
- 256 fixed random pairs;
- seed: `20260823`;
- exact Bias Trap definition;
- all original G0b thresholds.

Measurement repair only:

- thinking sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`;
- CoT max-new-token ceiling: 32,768;
- post-`</think>` final-answer scoring;
- robust-but-conservative canonical diagnosis extraction;
- common-random-number control/trap sampling;
- explicit invalid/truncation diagnostics.

## What positive G0s identify

If G0a+b+c all pass, the valid prerequisite statement is:

> The released MedEinst pairs are sufficiently aligned for paired analysis; Qwen3-14B reproduces a dense old-diagnosis persistence effect under a valid Qwen3 thinking regime; and a dense subset of the same phenomenon remains in a fixed-position direct-answer regime suitable for controlled internal intervention.

This still does **not** show that the new evidence was encoded.

## What G1 must do to distinguish encoding vs updating

A valid updating-failure result needs at least two ingredients:

1. **Evidence-state manipulation check:** in direct-mode trap inputs, the changed evidence must produce a measurable internal state that is absent/different in the paired control and is also present in correctly updated trap positive controls.
2. **Causal downstream test:** manipulating that evidence-related state must shift the diagnostic state/behavior away from the old control diagnosis without directly overwriting the answer representation.

A generic trap-vs-control linear probe is not enough: it could decode lexical edits, disease identity, or formatting. Same-pair / diagnosis-transition-matched intervention is preferred.

## Remaining mechanism risk

The changed span may contain several correlated lexical changes rather than one atomic evidence variable. If useful donor/recipient matching requires many post-hoc covariates, or if only a broad layer/token sweep produces rescue, the encoding-vs-updating distinction is not clean enough and the topic should stop.

## Current verdict

**RERUN repaired G0b v2.** If invalid rate is healthy (`<=0.10`) and substantive gates fail, stop scientifically. If invalid remains high due nontermination/unresolved final answers, treat it as measurement/runtime failure. Do not implement G1 until G0b and G0c both pass.
