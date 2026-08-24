# 22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?

**Status: `ACTIVE / G0a PASSED / G0b-v3 MEASUREMENT REPAIR READY`**

## Natural question

A classic explanation of the **Einstellung effect** is that an established solution becomes a mental set: decisive new evidence appears, but the reasoner remains trapped by the old interpretation.

For an LLM, an exact counterfactual Bias Trap leaves two different possibilities:

> **Was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?**

This remains the scientific question. No mechanism claim has been established yet.

## Seed and experimental object

Seed: ACL 2026 long paper **MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**.

- ACL: https://aclanthology.org/2026.acl-long.1847/
- official repository: https://github.com/zhui711/MedEinst
- dataset: https://huggingface.co/datasets/zhui711/MedEinst
- seed-supported model: `Qwen/Qwen3-14B`

The released test set contains 5,383 counterfactual pairs spanning a closed diagnosis vocabulary of 49 pathologies. The exact Bias Trap event is:

```text
model(control) = control ground truth
AND
model(trap) = control ground truth
AND
trap ground truth != control ground truth
```

The paper reports for Qwen3-14B:

```text
Baseline Accuracy = 44.12%
Bias Trap Rate    = 54.19%
```

## G0a — pair structure: PASSED

The full released test set passed the frozen alignment audit:

```text
valid pairs                  5383
malformed pairs              0
ground-truth flip rate       1.0000
age/sex match rate           1.0000
median changed-token frac    0.0726
p90 changed-token frac       0.2516
```

Verdict: `PAIR_STRUCTURE_OK`.

This establishes a clean paired object for later analysis. It does not by itself prove that every changed span is the medically decisive variable.

## G0b v1 — invalid measurement

The first Qwen3-14B CoT run used an unsuitable measurement stack:

- greedy decoding despite Qwen3 thinking-mode recommendations;
- only 1,024 new tokens;
- mandatory custom `FINAL_DIAGNOSIS:` parsing.

Invalid-output rate was 81.25%. That run is provenance only and carries no scientific verdict.

## G0b v2 — substantive signal appears, but open-text canonicalization fails

The principled v2 repair froze the model, 256 pair IDs, seed, Bias Trap definition and all scientific thresholds while fixing the demonstrated Qwen3 inference issues:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
max_new_tokens = 32768
post-</think> final-answer scoring only
```

The rerun completed all 256 pairs. Every thinking trace closed and no branch hit the token ceiling.

Substantive gates on the resolvable outputs all passed:

| Metric | v2 | Frozen gate |
|---|---:|---:|
| control accuracy | 0.3555 (91/256) | >=0.35 |
| control-correct count | 91 | >=50 |
| exact Bias Trap count | 34 | >=20 |
| Bias Trap Rate | 0.3736 | >=0.30 |
| Wilson lower bound | 0.2812 | >=0.20 |
| diagnosis transitions | 12 | >=8 |

But pair-level invalid rate remained:

```text
160/256 = 62.5%    required <=10%
```

The dominant branch failure was `unresolved_final`:

```text
control unresolved_final = 109
trap unresolved_final    = 124
```

Crucially, this is not a thinking/runtime termination failure. The final-answer segments exist; the deterministic parser cannot map many open-vocabulary diagnosis phrases onto the benchmark's closed 49-label vocabulary.

Therefore v2 is recorded as a **measurement failure, not a scientific negative**.

Historical record: [`MEASUREMENT_FAILURE_V2.md`](./MEASUREMENT_FAILURE_V2.md).

## Why a v3 repair is justified

The v2 failure localized a new, narrower defect that was not visible before the rerun:

> **free-form final diagnosis text -> closed benchmark label canonicalization**

A third repair is allowed here because it is deliberately **scoring-only and outcome-blind**:

- the 256 v2 CoT generations are frozen and must not be regenerated;
- model, pair IDs, seed, decoding and all scientific thresholds are unchanged;
- the fallback canonicalizer sees only the post-thinking final-answer text and the 49 closed labels;
- it never sees the clinical narrative, ground truth, case type, pair identity as control/trap, or paired branch;
- it may abstain rather than force a label;
- two deterministic label orders are evaluated and a mapping is accepted only if both agree;
- every canonical label must pass an exact self-mapping preflight under both orders before any benchmark output is rescored.

This is categorically different from adding hand-written synonyms after inspecting which diagnoses would improve the result, or changing prompts/models/samples until the gate passes.

Implementation: [`g0_recanonicalize_v3.py`](./g0_recanonicalize_v3.py).

## Frozen G0b-v3 contract

Input must be the original v2 record file:

```text
artifacts/g0_behavior_cot/records.jsonl
```

`run_g0.sh` fails fast if this file is absent; it does **not** silently regenerate CoT.

V3 first runs a 49-label canonicalizer self-mapping preflight. Then it applies the semantic canonicalizer only to branches that:

```text
old deterministic parser -> unresolved
AND thinking closed
AND did not hit max tokens
```

Already-resolved v2 predictions are preserved exactly.

The original frozen CoT gates remain unchanged:

- control accuracy `>=0.35`;
- control-correct count `>=50`;
- Bias Trap count `>=20`;
- Bias Trap Rate `>=0.30`;
- 95% Wilson lower bound `>=0.20`;
- at least 8 distinct diagnosis transitions;
- invalid-output rate `<=0.10`.

### V3 decisions

```text
CANONICALIZER_PREFLIGHT_FAILURE
    -> measurement object invalid; stop before rescoring

MEASUREMENT_CANONICALIZATION_FAILURE
    -> invalid rate remains >10%; stop measurement route, no scientific negative

SEED_PHENOMENON_NOT_REPRODUCED
    -> measurement healthy but substantive frozen gates fail; real scientific stop

SEED_PHENOMENON_REPRODUCED
    -> proceed to direct-mode G0c
```

## G0c — direct-answer mechanism eligibility

Only after G0b-v3 is measurement-healthy and reproduces the seed do we run direct mode on the **same exact 256 pairs, model and seed**.

Reason: variable-length CoT is a poor substrate for simple token-local causal analysis. The downstream mechanism study needs the Bias Trap to remain dense in a fixed-position direct-answer regime.

Direct raw generations are also passed through the same closed-label v3 canonicalizer before applying the pre-existing direct gates:

- direct control accuracy `>=0.30`;
- control-correct count `>=40`;
- exact Bias Trap count `>=16`;
- direct Bias Trap Rate `>=0.20`;
- Wilson lower bound `>=0.10`;
- at least 6 diagnosis transitions;
- invalid-output rate `<=0.10`.

If CoT reproduces but direct mode fails a healthy substantive gate, stop the simple fixed-position mechanism route. Do not replace it with open-ended CoT state fishing.

## If all G0s pass

Only then is the mechanism distinction eligible for study:

1. align changed evidence spans within each control/trap pair;
2. use correctly updated trap cases as positive controls for an evidence-sensitive internal transition;
3. compare exact Bias Trap cases at a small predeclared site set;
4. prefer same-pair / diagnosis-transition-matched causal intervention;
5. require a manipulation check that distinguishes evidence representation from direct answer overwriting.

A generic trap-vs-control probe is not enough.

## Run

```bash
cd 22_medeinst_evidence_update
MODEL=Qwen/Qwen3-14B \
N_PAIRS=256 \
SEED=20260823 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_g0.sh
```

If the v2 records are elsewhere:

```bash
V2_COT_RECORDS=/path/to/original/v2/records.jsonl bash run_g0.sh
```

## Files

- `g0_pair_locality.py` — full pair-structure audit.
- `g0_bias_trap_screen.py` — original CoT/direct generator and deterministic parser.
- `g0_recanonicalize_v3.py` — frozen scoring-only closed-label repair.
- `tests/test_g0_helpers.py`
- `tests/test_g0_v3_canonicalizer.py`
- `G0_RESULTS.md`
- `VALIDATION_AUDIT.md`
- `MEASUREMENT_FAILURE_V2.md`

## Scientific invariant

> **Same released patient pair, same model and frozen case set: control is correct, trap ground truth flips, and the model persists on the old diagnosis. Measurement repairs may make the benchmark label interface valid; they may not change the underlying CoT generations or use ground truth to decide how to score them. Only after this exact event is reproducible and dense do we ask what evidence the model encoded and why it failed to update.**
