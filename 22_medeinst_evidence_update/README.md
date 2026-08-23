# 22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?

**Status: CANDIDATE / FROZEN STRUCTURE + SEED-REPRO + DIRECT-MODE G0 READY**

## Natural question

A classic explanation of the **Einstellung effect** is that an established solution becomes a mental set: decisive new evidence appears, but the reasoner remains trapped by the old interpretation.

For an LLM, a wrong counterfactual diagnosis leaves two very different possibilities:

> **Was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?**

## Seed

ACL 2026 long paper: **MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**.

- ACL: https://aclanthology.org/2026.acl-long.1847/
- Official repository: https://github.com/zhui711/MedEinst
- Dataset: https://huggingface.co/datasets/zhui711/MedEinst

The released test set has **5,383 counterfactual pairs**. Each control/trap pair shares a `case_id`; the trap changes key discriminative evidence so the ground-truth diagnosis flips. The paper defines the Bias Trap event exactly as:

```text
model(control) = control ground truth
AND
model(trap) = control ground truth
```

The paper reports on `Qwen/Qwen3-14B`:

```text
Baseline Accuracy = 44.12%
Bias Trap Rate    = 54.19%
```

All baseline models are evaluated under a **zero-shot Chain-of-Thought** setting. Therefore our first behavioral gate uses that same model family and reasoning mode rather than starting from an unverified 8B direct-answer setting.

## New mechanism distinction

### Encoding failure

Trap-specific evidence never becomes a usable internal state.

### Updating failure

Trap-specific evidence affects internal computation, but the diagnostic state remains locked to the old control diagnosis.

These are mechanism claims. G0 does **not** prove either one. G0 only establishes whether we have a clean enough paired object and a tractable direct-answer regime on which the distinction can later be tested.

## Why this is feasible

- ACL main seed.
- Released paired data and exact diagnosis labels.
- Seed-supported open model `Qwen/Qwen3-14B`, easily within local GPU capacity.
- No paid API needed.
- No new human annotation for the prerequisite gates.
- Pair-level ground truth and Bias Trap definition are exact.
- The dataset construction already provides medical validation of the counterfactual diagnosis flip.

## G0a — pair-locality / alignment audit

`g0_pair_locality.py` audits every released test pair using only the dataset.

For each `case_id` it checks:

- exactly one control and one trap;
- ground-truth diagnosis flip;
- age/sex preservation;
- token-level edit fraction;
- changed block count;
- largest changed span;
- the actual aligned changed text spans, saved in `pair_diffs.jsonl`.

### Important interpretation

This gate answers only:

> **Are the released pairs local enough to support aligned causal intervention later?**

It does **not** independently prove that every token diff is medically decisive evidence. The seed benchmark's construction and physician/LLM validation establish the medical counterfactual validity. We must not reinterpret a small text diff as a mechanism label by itself.

### Frozen G0a gate

Proceed only if all hold:

- at least `5000` valid pairs;
- `0` malformed pairs;
- diagnosis flip rate `>= 0.99`;
- age+sex match rate `>= 0.99`;
- median changed-token fraction `<= 0.12`;
- p90 changed-token fraction `<= 0.30`.

If this fails, the **aligned-intervention route** is not clean enough. Do not cherry-pick visually clean pairs after seeing model behavior.

## G0b — reproduce the published Bias Trap phenomenon

Default model:

```text
Qwen/Qwen3-14B
```

Default sample:

```text
256 fixed random test pairs, seed 20260823
```

The sample is random rather than diagnosis-balanced so its prevalence is comparable to the benchmark distribution.

The prompt uses zero-shot CoT and requires a machine-readable final marker:

```text
FINAL_DIAGNOSIS: <diagnosis>
```

Only the final marker is scored. Diagnosis names mentioned inside the reasoning trace cannot accidentally count as predictions.

Primary event:

```text
control prediction = control ground truth
AND
trap prediction = control ground truth
AND
trap ground truth != control ground truth
```

### Frozen G0b gate

Proceed only if all hold:

- control accuracy `>= 0.35`;
- at least `50` control-correct cases;
- at least `20` exact Bias Trap events;
- Bias Trap Rate among control-correct cases `>= 0.30`;
- 95% Wilson lower bound for Bias Trap Rate `>= 0.20`;
- Bias Trap events cover at least `8` distinct control→trap diagnosis transitions;
- invalid-output rate `<= 0.10`.

These thresholds are deliberately below the published Qwen3-14B point estimates but still require a dense, statistically non-fragile event set.

If G0b fails, stop. Do not switch models/prompts until one produces the desired effect.

## G0c — direct-answer mechanism eligibility

Variable-length CoT trajectories are a bad object for simple token-level causal analysis: different cases and outcomes can follow different reasoning paths, recreating the trajectory-alignment problem that killed prior topics.

Therefore, on **the exact same fixed random pairs and same Qwen3-14B model**, rerun a direct-answer condition with thinking disabled:

```text
FINAL_DIAGNOSIS: <diagnosis>
```

This is not used to reproduce the paper. It is a feasibility gate for our mechanism route.

### Frozen G0c gate

Proceed to simple hidden-state mechanism work only if all hold:

- direct control accuracy `>= 0.30`;
- at least `40` direct control-correct cases;
- at least `16` exact direct Bias Trap events;
- direct Bias Trap Rate `>= 0.20`;
- 95% Wilson lower bound `>= 0.10`;
- Bias Trap events cover at least `6` diagnosis transitions;
- invalid-output rate `<= 0.10`.

If CoT reproduces the seed but direct mode fails this gate, the Einstellung phenomenon remains real, but **our simple fixed-position mechanism design is not justified**. Do not silently move to open-ended CoT trajectory probing.

## Run

```bash
cd 22_medeinst_evidence_update
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_g0.sh
```

The script runs in order and stops early if G0a or G0b fails.

Outputs:

```text
artifacts/g0_pair_locality/summary.json
artifacts/g0_pair_locality/pair_metrics.csv
artifacts/g0_pair_locality/pair_diffs.jsonl
artifacts/g0_pair_locality/most_diffuse_200.jsonl
artifacts/g0_behavior_cot/records.jsonl
artifacts/g0_behavior_cot/summary.json
artifacts/g0_behavior_direct/records.jsonl
artifacts/g0_behavior_direct/summary.json
```

## What positive G0s prove

If G0a+b+c all pass, we may claim only:

> The released counterfactual pairs are sufficiently local for aligned analysis; the published Bias Trap phenomenon reproduces on a seed-supported open model; and a dense subset of the same exact old-diagnosis persistence events also exists without variable-length CoT.

This gives a tractable mechanism object. It still does **not** prove that the new evidence is internally encoded.

## If all G0s pass: next identification step

Before training any generic probe, freeze an explicit evidence/update experiment.

1. Use `pair_diffs.jsonl` to align the small changed spans within each control/trap pair.
2. Restrict mechanism analysis to direct-mode cases so answer sites are fixed.
3. Use correctly updated trap cases as positive controls for what an evidence-sensitive internal transition looks like.
4. Compare them against exact Bias Trap cases at a **small predeclared site set**.
5. Prefer same-pair or diagnosis-transition-matched causal patching/ablation over a global learned steering vector.

A valid updating-failure result must show more than decodability. The evidence-related state must survive a manipulation check and the downstream diagnostic state must remain causally biased toward the old diagnosis.

If identifying the decisive internal state requires broad matching, model/layer/token sweeps, or a GPT/API evidence annotator, stop and reconsider the topic.

## Mechanism kill lines

Stop if:

- pair locality fails;
- seed-faithful Qwen3-14B Bias Trap fails to reproduce;
- direct-mode exact Bias Trap events are sparse;
- the intended changed spans cannot be aligned at useful density;
- positive-control correctly updated traps do not show a recoverable evidence-sensitive state;
- matching donor/recipient cases requires many post-hoc covariates;
- only broad layer/token/coefficient search finds rescue;
- intervention works only by overwriting the final answer representation.

## Collision boundary

This is not another medical-bias benchmark, not another prompt-mitigation paper, and not generic anchoring/sycophancy probing.

The scoped question is:

> **Within exact counterfactual Bias Trap pairs, can we causally separate failure to encode discriminative evidence from failure to use evidence-related internal state to update the diagnosis?**

## Method opening

A genuine evidence-integration bottleneck gives a concrete target for counterfactual evidence-update consistency training, revision-sensitive representation regularization, or an evidence-triggered revision mechanism.

## Files

- `g0_pair_locality.py` — full test-pair structure and aligned diff audit.
- `g0_bias_trap_screen.py` — seed-faithful CoT and direct-answer Bias Trap screens.
- `run_g0.sh`
- `requirements.txt`
- `tests/test_g0_helpers.py`

## Scientific invariant

> **same released patient pair; control is correct; trap ground truth flips; model persists on the old control diagnosis. First reproduce that exact event, then require it to exist in a mechanism-tractable direct regime before asking what the model encoded.**
