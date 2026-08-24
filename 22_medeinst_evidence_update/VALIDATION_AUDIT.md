# Topic 22 Validation Audit

**Audit status: G0a passed; G0b-v2 localized a closed-label canonicalization failure; G0b-v3 scoring-only repair frozen and ready.**

## Claim hierarchy

Keep these levels separate.

1. **Pair structure:** released counterfactual pairs are sufficiently local/aligned for paired analysis.
2. **Behavioral phenomenon:** Qwen3-14B reproduces dense exact Bias Trap events under a valid CoT measurement.
3. **Mechanism-tractable regime:** the same event remains dense with thinking disabled and fixed answer sites.
4. **Mechanism:** decisive new evidence is encoded but fails to update the diagnostic state.

G0 can establish only 1–3. It cannot establish 4.

## G0a — passed

Full test-set audit:

- 5,383 valid pairs;
- 0 malformed pairs;
- ground-truth flip rate 1.0;
- age/sex match rate 1.0;
- median changed-token fraction 0.0726;
- p90 changed-token fraction 0.2516.

Verdict: `PAIR_STRUCTURE_OK`.

Pair locality is an intervention/alignment check; medical validity of the counterfactual flip comes from the benchmark construction, not edit distance.

## G0b v1 — correctly invalidated

The first local Qwen3-14B CoT run had three demonstrated measurement defects:

1. greedy decoding in Qwen3 thinking mode;
2. only 1,024 new tokens;
3. a mandatory custom `FINAL_DIAGNOSIS:` marker.

Its 81.25% invalid rate could not be read scientifically.

## G0b v2 — what it established

V2 repaired those defects while preserving the scientific model/sample/gates:

- Qwen3-recommended thinking sampling (`temperature=0.6`, `top_p=0.95`, `top_k=20`);
- 32,768-token ceiling;
- post-`</think>` final segment only;
- conservative exact/sub-string canonical label extraction;
- deterministic pair-level common random numbers;
- same fixed 256 pairs and seed `20260823`.

The rerun is diagnostically important:

```text
control accuracy             0.3555  PASS
control-correct              91      PASS
Bias Trap count              34      PASS
Bias Trap Rate               0.3736  PASS
Wilson lower bound           0.2812  PASS
diagnosis transitions        12      PASS
invalid rate                 0.6250  FAIL
```

All 256 control and trap traces closed and no branch hit max tokens.

The unresolved counts were:

```text
control unresolved_final = 109
trap unresolved_final    = 124
```

Thus V2 does **not** support the claim that Qwen3 failed to finish reasoning. It localizes the measurement problem to the interface between free-form final diagnosis wording and the benchmark's finite pathology vocabulary.

## Why v3 is not ordinary post-hoc tuning

A repair after viewing results is dangerous when it searches across models, prompts, aliases, thresholds or evaluation rules until the desired scientific effect appears.

V3 is permitted because its design satisfies a stricter criterion:

> **It changes only the semantic canonicalization of already-frozen outputs and has no access to information that could tell it whether a mapping helps the scientific hypothesis.**

Specifically:

- no CoT regeneration;
- no model/sample/seed/decoding change;
- no scientific threshold change;
- no diagnosis-specific hand-written synonym table;
- no patient narrative in the mapper;
- no ground truth in the mapper;
- no control/trap identity in the mapper;
- no paired branch in the mapper;
- explicit abstention is allowed;
- label-order invariance is required.

This is a measurement-interface repair, not a search for a stronger effect.

## V3 canonicalizer contract

Implementation: `g0_recanonicalize_v3.py`.

### Input boundary

For each unresolved branch the mapper receives exactly:

```text
post-thinking final-answer text
+
49 closed benchmark labels
```

It is explicitly told to map semantic equivalence rather than diagnose from a patient case.

### Output boundary

Output is a single numeric ID:

```text
0     = abstain / ambiguous / no equivalent
1..49 = one closed label
```

Any extra prose or out-of-range output is unresolved.

### Dual-order guard

Two fixed deterministic label orders are used. A canonical label is accepted only if both orders return that same nonzero label.

This guards against a single menu-order artifact without opening a label-order sweep.

### Preflight

Before touching benchmark outputs:

```text
Final diagnosis: <canonical label>
```

must map back to itself for every one of the 49 labels under both orders.

A single preflight failure returns:

`CANONICALIZER_PREFLIGHT_FAILURE`

and blocks benchmark interpretation.

### Fallback-only rule

V3 does not rewrite any old resolved v2 prediction. It is called only when the old parser returned `None`, the thinking trace closed, and the branch did not hit the token ceiling.

### Frozen source outputs

The original v2 `records.jsonl` is mandatory. The v3 script records its SHA-256 digest. `run_g0.sh` refuses to silently regenerate CoT if the file is missing.

This matters because regeneration under stochastic CoT after observing v2 would create a different behavioral sample.

## Frozen G0b gates — unchanged

- control accuracy `>=0.35`;
- control-correct count `>=50`;
- exact Bias Trap count `>=20`;
- Bias Trap Rate among control-correct `>=0.30`;
- Wilson lower bound `>=0.20`;
- at least 8 diagnosis transitions;
- pair invalid rate `<=0.10`.

Decision logic:

```text
canonicalizer preflight fails
    -> CANONICALIZER_PREFLIGHT_FAILURE

invalid rate > 0.10
    -> MEASUREMENT_CANONICALIZATION_FAILURE
    -> no scientific verdict

measurement healthy + substantive gate fails
    -> SEED_PHENOMENON_NOT_REPRODUCED
    -> real local scientific stop

all gates pass
    -> SEED_PHENOMENON_REPRODUCED
    -> run direct G0c
```

After a v3 measurement-support failure, no further local alias/prompt/mapper-model/order/threshold tuning is authorized on this same result.

## G0c — direct mode remains a separate prerequisite

Even a successful CoT reproduction is not enough for the planned simple causal mechanism study.

Variable-length CoT trajectories are not the intended token-local mechanism substrate. Therefore the same exact 256 pair IDs must also produce a dense Bias Trap event under thinking-disabled direct answers.

Only after G0b-v3 passes does `run_g0.sh` generate direct outputs. Those raw outputs use the same v3 canonicalization interface before the pre-existing direct gates are read.

Direct gates remain:

- control accuracy `>=0.30`;
- control-correct count `>=40`;
- exact Bias Trap count `>=16`;
- Bias Trap Rate `>=0.20`;
- Wilson lower bound `>=0.10`;
- at least 6 diagnosis transitions;
- invalid-output rate `<=0.10`.

A healthy direct-mode substantive failure is terminal for the fixed-position mechanism route; do not silently migrate to open-ended CoT probing.

## What G1 would need to identify encoding vs updating

Only if G0a+b+c pass:

1. **Evidence-state manipulation check:** paired changed evidence must induce a measurable internal transition also present in correctly updated trap positive controls.
2. **Causal downstream test:** intervention on that evidence-related state must move the diagnostic state/behavior away from the old control diagnosis without directly overwriting the answer representation.

A control-vs-trap linear probe is insufficient because it can decode lexical edits, diagnosis identity or formatting.

Prefer same-pair or diagnosis-transition-matched causal intervention at a small predeclared site set. If the effect requires broad model/layer/token/coefficient search, stop.

## Current verdict

```text
G0a: PASS
G0b-v1: MEASUREMENT INVALID
G0b-v2: MEASUREMENT INVALID, DEFECT LOCALIZED TO CLOSED-LABEL CANONICALIZATION
G0b-v3: FROZEN / READY
G0c: NOT RUN
G1: NOT RUN
NO SCIENTIFIC VERDICT YET
```
