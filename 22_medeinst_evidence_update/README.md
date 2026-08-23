# 22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?

**Status: CANDIDATE / STRUCTURAL G0 + BEHAVIORAL G0 READY**

## Natural question

A classic explanation of the **Einstellung effect** is that an established solution becomes a mental set: decisive new evidence appears, but the reasoner remains trapped by the old interpretation.

For an LLM, a wrong counterfactual diagnosis leaves two very different possibilities:

> **Was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?**

## Seed

ACL 2026 long paper: **MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**.

- ACL: https://aclanthology.org/2026.acl-long.1847/
- Official repository: https://github.com/zhui711/MedEinst
- Dataset: https://huggingface.co/datasets/zhui711/MedEinst

The released test set contains paired `control` and `trap` cases. In trap cases, decisive evidence changes so that the correct diagnosis flips; the Einstellung failure is continuing to output the old control diagnosis.

## New mechanism distinction

### Encoding failure

Trap-specific decisive evidence never becomes a usable internal state.

### Updating failure

Trap-specific evidence is represented, but the old diagnosis remains dominant and the representation fails to revise the decision.

We do **not** start with hidden states. First we require clean pairs and a dense exact Bias Trap event set.

## Why this is feasible

- ACL main seed.
- Released paired data and exact diagnosis labels.
- No paid API needed.
- No new human annotation for G0.
- Open local model can be used for behavioral screening.
- Counterfactual pairing cancels most patient/narrative identity.
- If the mechanism is real, it opens a direct evidence-update training target.

## G0a — full pair-locality audit

`g0_pair_locality.py` audits every released test pair using only the dataset.

For each `case_id`, require exactly one control and one trap and measure:

- diagnosis flip;
- age/sex preservation;
- token-level edit fraction;
- changed block count;
- largest changed span.

### Frozen G0a gate

Proceed only if all hold:

- at least `5000` valid pairs;
- `0` malformed pairs;
- diagnosis flip rate `>=0.99`;
- age+sex match rate `>=0.99`;
- median changed-token fraction `<=0.12`;
- p90 changed-token fraction `<=0.30`.

If this fails, stop. Do not cherry-pick visually clean pairs after seeing model behavior.

## G0b — exact Bias Trap density

Default model:

```text
Qwen/Qwen3-8B
```

Default screen:

```text
512 fixed diagnosis-stratified test pairs
```

For each pair, predict one diagnosis for control and trap.

Primary event:

```text
control prediction = control ground truth
AND
trap prediction = control ground truth
AND
trap ground truth != control ground truth
```

This excludes unrelated trap errors and control cases the model never knew.

### Frozen G0b gate

Proceed only if all hold:

- control accuracy `>=0.35`;
- at least `50` control-correct cases;
- at least `20` exact Bias Trap events;
- Bias Trap Rate among control-correct cases `>=0.20`;
- invalid-output rate `<=0.10`.

If the exact event is sparse, stop. Do not model/prompt-shop to manufacture it.

## Run

```bash
cd 22_medeinst_evidence_update
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_g0.sh
```

Outputs:

```text
artifacts/g0_pair_locality/summary.json
artifacts/g0_pair_locality/pair_metrics.csv
artifacts/g0_pair_locality/most_diffuse_200.jsonl
artifacts/g0_behavior/records.jsonl
artifacts/g0_behavior/summary.json
```

## If both G0s pass

1. Automatically locate the changed evidence span from each control/trap pair.
2. At a small predeclared depth set, test whether trap-specific evidence becomes internally distinguishable.
3. Track whether the diagnostic state nevertheless remains aligned to the old control diagnosis.
4. Prefer natural matched activation patching: donor = correctly updated trap; recipient = Bias Trap case.

The useful dissociation is:

```text
trap-specific evidence is represented
but diagnostic state remains aligned to the old diagnosis
```

A probe alone is not a mechanism result.

## Mechanism kill lines

Stop if:

- pair locality fails;
- exact Bias Trap events are sparse;
- trap evidence is not recoverable even in correctly updated positive controls;
- matching donor/recipient cases requires many post-hoc covariates;
- only broad layer/token/coefficient search finds rescue;
- intervention works only by overwriting the final diagnosis representation.

## Collision boundary

This is not another medical-bias benchmark, not another prompt-mitigation paper, and not generic anchoring/sycophancy probing.

The scoped question is:

> **Within exact counterfactual Bias Trap pairs, can we causally separate failure to encode decisive evidence from failure to use encoded evidence to update the diagnostic state?**

## Method opening

A genuine evidence-integration bottleneck gives a concrete target for counterfactual evidence-update consistency training, revision-sensitive representation regularization, or an evidence-triggered revision gate.

## Files

- `g0_pair_locality.py`
- `g0_bias_trap_screen.py`
- `run_g0.sh`
- `requirements.txt`
- `tests/test_g0_helpers.py`

## Scientific invariant

> **same released patient pair; control is correct; trap flips the true diagnosis; model stays on the old control diagnosis. Then ask whether the new evidence was encoded but failed to update the diagnosis.**
