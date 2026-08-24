# G0 Results — Topic 22

## Current status

**G0a PASSED. G0b-v2 was measurement-invalid. G0b-v3 is frozen and ready. No scientific verdict yet.**

Topic 22 has not reached direct-mode G0c or mechanism analysis.

## G0a pair locality — VALID

| Metric | Value | Gate |
|---|---:|---|
| valid pairs | 5383 | pass (>=5000) |
| malformed pairs | 0 | pass |
| ground-truth flip rate | 1.0000 | pass (>=0.99) |
| age/sex match rate | 1.0000 | pass (>=0.99) |
| changed-token fraction median | 0.0726 | pass (<=0.12) |
| changed-token fraction p90 | 0.2516 | pass (<=0.30) |

G0a verdict: `PAIR_STRUCTURE_OK`.

## G0b v1 — invalidated measurement

The first Qwen3-14B CoT screen used greedy thinking-mode decoding, a 1,024-token reasoning budget, and a mandatory custom final-answer marker. Its 81.25% invalid rate is provenance only and carries no scientific verdict.

## G0b v2 — completed

Frozen run:

- model: `Qwen/Qwen3-14B`;
- dataset: `zhui711/MedEinst`, test split;
- fixed random sample: 256 pairs;
- seed: `20260823`;
- four A100 GPUs;
- `temperature=0.6`, `top_p=0.95`, `top_k=20`;
- `max_new_tokens=32768`;
- score only post-`</think>` final-answer content.

### V2 metrics

| Metric | Value | Frozen gate |
|---|---:|---|
| control accuracy | 0.3555 (91/256) | pass (>=0.35) |
| control-correct count | 91 | pass (>=50) |
| exact Bias Trap count | 34 | pass (>=20) |
| Bias Trap Rate among control-correct | 0.3736 | pass (>=0.30) |
| 95% Wilson lower bound | 0.2812 | pass (>=0.20) |
| diagnosis transitions | 12 | pass (>=8) |
| invalid-output rate | **0.6250 (160/256)** | **fail (<=0.10)** |

All 256 control and trap thinking traces closed. Neither side hit the 32,768-token ceiling.

The dominant failures were:

```text
control unresolved_final = 109
trap unresolved_final    = 124
```

The extraction-method audit also showed that unresolved outputs were not empty/nonterminated generations: the model had produced post-thinking final text, but the exact/substring parser could not resolve it to a canonical benchmark pathology.

Historical v2 verdict: `MEASUREMENT_RUNTIME_FAILURE`.

Interpretation: **measurement invalid, no scientific hypothesis verdict**.

Artifact: `artifacts/g0_behavior_cot/summary.json`.

## V3 failure diagnosis

V2 isolates a different defect from v1:

> open-vocabulary diagnosis phrasing cannot reliably be scored against the benchmark's closed 49-pathology label space with exact/sub-string matching.

This is a label-interface problem, not evidence that Qwen3 failed to produce a diagnosis and not evidence that the MedEinst phenomenon is absent.

Adding hand-written aliases after inspecting failed examples would create outcome-dependent researcher degrees of freedom. Regenerating the 256 CoTs with a stronger formatting instruction would also change the frozen behavioral sample.

Therefore v3 is a **scoring-only semantic canonicalization repair**.

## Frozen G0b-v3 repair

Implementation: `g0_recanonicalize_v3.py`.

### Inputs that remain frozen

- exact v2 `records.jsonl` generations;
- Qwen3-14B;
- same 256 case IDs;
- same seed;
- same original prompts and decoding already used to produce v2;
- same Bias Trap definition;
- every scientific threshold above.

The script hashes the input record file and records the exact case-ID sequence.

### Canonicalizer information boundary

The semantic canonicalizer sees only:

```text
post-thinking final-answer text
+
closed list of 49 benchmark diagnosis labels
```

It does **not** receive:

```text
clinical narrative
ground truth
case type
control/trap identity
paired branch output
```

The mapper is instructed to perform semantic label canonicalization only, not clinical diagnosis. It returns a numeric label ID or `0` to abstain.

### Order-bias guard

Each unresolved final answer is canonicalized under two deterministic permutations of the 49 labels.

A prediction is accepted only when both orders independently map to the same non-abstaining canonical label.

### Preflight

Before any benchmark output is rescored, all 49 canonical labels are fed back as trivial final diagnoses. Every label must self-map correctly under both frozen label orders.

If any label fails:

```text
CANONICALIZER_PREFLIGHT_FAILURE
```

and no benchmark row is interpreted.

### Fallback only

V3 does not reconsider predictions already resolved by v2. Semantic canonicalization is invoked only when:

```text
v2 pred is None
AND thinking closed
AND no max-token failure
```

This prevents the semantic mapper from rewriting already-scored outcomes.

## Frozen G0b-v3 decisions

All original G0b thresholds remain unchanged.

```text
invalid_rate > 0.10
    -> MEASUREMENT_CANONICALIZATION_FAILURE
    -> no scientific negative

invalid_rate <= 0.10 but any substantive gate fails
    -> SEED_PHENOMENON_NOT_REPRODUCED
    -> scientific stop for this local seed regime

all gates pass
    -> SEED_PHENOMENON_REPRODUCED
    -> proceed to direct-mode G0c
```

If v3 still fails measurement support, stop the local measurement route. No synonym tuning, alternate mapper model, alternate prompt, or threshold relaxation is authorized after seeing v3.

## Direct-mode G0c

Direct mode remains downstream of successful CoT reproduction and has **not yet run**.

If G0b-v3 passes, `run_g0.sh` will:

1. generate direct outputs on the same 256 pairs/model/seed;
2. apply the same v3 closed-label canonicalization contract;
3. verify CoT/direct case IDs are identical;
4. apply the already-frozen direct gates.

Direct gates:

- control accuracy `>=0.30`;
- at least 40 control-correct cases;
- at least 16 exact Bias Trap events;
- Bias Trap Rate `>=0.20`;
- Wilson lower bound `>=0.10`;
- at least 6 diagnosis transitions;
- invalid-output rate `<=0.10`.

Only `DIRECT_MODE_MECHANISM_OBJECT_READY` allows mechanism work.

## Run

The v3 repair requires the original v2 records and deliberately refuses to regenerate them implicitly:

```bash
cd 22_medeinst_evidence_update
MODEL=Qwen/Qwen3-14B \
N_PAIRS=256 \
SEED=20260823 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_g0.sh
```

If necessary:

```bash
V2_COT_RECORDS=/path/to/original/v2/records.jsonl bash run_g0.sh
```

## Evidence / provenance

- v2 final recording commit: `2a6f9712bd5e799b237be455f79a5b24c648fc06`
- historical record: `MEASUREMENT_FAILURE_V2.md`
- v3 implementation: `g0_recanonicalize_v3.py`
- v3 tests: `tests/test_g0_v3_canonicalizer.py`

Current verdict:

```text
G0B_V3_READY
NO_SCIENTIFIC_VERDICT_YET
DIRECT_MODE_NOT_RUN
```
