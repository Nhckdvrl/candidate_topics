# 22 — Does the Model Encode New Evidence but Fail to Update Its Diagnosis?

**Status: CANDIDATE / G0a PASSED / REPAIRED G0b v2 READY**

## Natural question

A classic explanation of the **Einstellung effect** is that an established solution becomes a mental set: decisive new evidence appears, but the reasoner remains trapped by the old interpretation.

For an LLM, a wrong counterfactual diagnosis leaves two very different possibilities:

> **Was the decisive new evidence never encoded, or was it encoded but unable to update the old diagnosis?**

## Seed

ACL 2026 long paper: **MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**.

- ACL: https://aclanthology.org/2026.acl-long.1847/
- Official repository: https://github.com/zhui711/MedEinst
- Dataset: https://huggingface.co/datasets/zhui711/MedEinst

The released test set has **5,383 counterfactual pairs**. Each control/trap pair shares a `case_id`; the trap changes key discriminative evidence so the ground-truth diagnosis flips.

The exact Bias Trap event used here is:

```text
model(control) = control ground truth
AND
model(trap) = control ground truth
AND
trap ground truth != control ground truth
```

The paper reports on `Qwen/Qwen3-14B`:

```text
Baseline Accuracy = 44.12%
Bias Trap Rate    = 54.19%
```

## New mechanism distinction

### Encoding failure

Trap-specific evidence never becomes a usable internal state.

### Updating failure

Trap-specific evidence affects internal computation, but the diagnostic state remains locked to the old control diagnosis.

These are mechanism claims. G0 does **not** prove either one. G0 only establishes whether we have a clean enough paired object and a tractable direct-answer regime on which the distinction can later be tested.

## Why this is feasible

- ACL main seed.
- Released paired data and exact diagnosis labels.
- Seed-supported open model `Qwen/Qwen3-14B`.
- No paid API needed.
- No new human annotation for prerequisite gates.
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
- actual aligned changed text spans saved in `pair_diffs.jsonl`.

### G0a result

The full released test set passed:

```text
valid pairs                  5383
malformed pairs              0
ground-truth flip rate       1.0000
age/sex match rate           1.0000
median changed-token frac    0.0726
p90 changed-token frac       0.2516
```

G0a verdict: `PAIR_STRUCTURE_OK`.

This means the pairs are sufficiently local for aligned analysis. It does **not** independently prove that every changed token is the medically decisive variable.

## Why the first G0b run is invalid

The first local Qwen3-14B CoT run reported 81.25% invalid pairs. That run is **not a scientific negative** and must not be used to archive the topic.

The measurement implementation had three material problems:

1. thinking mode used greedy decoding, although the official Qwen3 model card explicitly warns against greedy decoding for thinking mode and recommends `temperature=0.6`, `top_p=0.95`, `top_k=20`;
2. thinking was capped at only 1,024 new tokens, while the official Qwen3 example allows up to 32,768 new tokens;
3. scoring required our custom literal `FINAL_DIAGNOSIS:` marker, so a valid final diagnosis phrased naturally could be counted invalid.

The old `STOP_SEED_PHENOMENON_NOT_REPRODUCED` verdict is therefore withdrawn. See `G0_RESULTS.md` and `VALIDATION_AUDIT.md`.

## G0b v2 — repaired seed reproduction

Frozen scientific choices are unchanged:

```text
model       Qwen/Qwen3-14B
sample      256 fixed random test pairs
seed        20260823
split       test
```

The repaired measurement uses:

```text
Qwen3 thinking enabled
temperature = 0.6
top_p       = 0.95
top_k       = 20
max_new_tokens = 32768
```

Control and trap use the same deterministic per-pair sampling seed (common random numbers).

### Final-answer extraction

The evaluator separates Qwen3 thinking from the final answer using the generated `</think>` token. **Only post-thinking final-answer content is scored.**

The preferred format remains:

```text
FINAL_DIAGNOSIS: <diagnosis>
```

but the literal marker is no longer mandatory. An unambiguous canonical dataset diagnosis in the post-thinking final answer can also be resolved conservatively.

No disease mention inside the reasoning trace can count as the prediction. No LLM judge or semantic fuzzy matcher is used.

The evaluator now records:

- extraction method;
- generated token count;
- whether `</think>` closed;
- whether generation hit the token ceiling;
- branch-level invalid reason (`hit_max_tokens`, `thinking_not_closed`, `unresolved_final`).

### Frozen G0b gate — unchanged

Proceed only if all hold:

- control accuracy `>= 0.35`;
- at least `50` control-correct cases;
- at least `20` exact Bias Trap events;
- Bias Trap Rate among control-correct cases `>= 0.30`;
- 95% Wilson lower bound for Bias Trap Rate `>= 0.20`;
- Bias Trap events cover at least `8` distinct control→trap diagnosis transitions;
- invalid-output rate `<= 0.10`.

### Verdict semantics

- `SEED_PHENOMENON_REPRODUCED`: measurement healthy and all substantive gates pass.
- `SEED_PHENOMENON_NOT_REPRODUCED`: invalid rate is healthy (`<=0.10`) but substantive frozen gates fail. This is a real reproduction stop.
- `MEASUREMENT_RUNTIME_FAILURE`: invalid rate remains `>0.10`. This is **not** evidence that the MedEinst phenomenon is false.

No model, prompt, sample, threshold, or diagnosis subset may be searched to rescue a healthy scientific failure.

## G0c — direct-answer mechanism eligibility

Variable-length CoT trajectories are a poor object for simple token-level causal analysis. Therefore, only after repaired G0b passes, rerun the **same exact 256 pair IDs** on the same Qwen3-14B with thinking disabled.

Direct mode remains deterministic and asks for one concise diagnosis.

### Frozen G0c gate

Proceed to simple hidden-state mechanism work only if all hold:

- direct control accuracy `>= 0.30`;
- at least `40` direct control-correct cases;
- at least `16` exact direct Bias Trap events;
- direct Bias Trap Rate `>= 0.20`;
- 95% Wilson lower bound `>= 0.10`;
- Bias Trap events cover at least `6` diagnosis transitions;
- invalid-output rate `<= 0.10`.

If CoT reproduces the seed but direct mode fails this gate, the Einstellung phenomenon may remain real, but **our simple fixed-position mechanism design is not justified**. Do not silently move to open-ended CoT trajectory probing.

## Run

```bash
cd 22_medeinst_evidence_update
pip install -r requirements.txt
MODEL=Qwen/Qwen3-14B \
N_PAIRS=256 \
SEED=20260823 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_g0.sh
```

The script runs in order and stops early if G0a or G0b fails.

Outputs:

```text
artifacts/g0_pair_locality/summary.json
artifacts/g0_pair_locality/pair_metrics.csv
artifacts/g0_pair_locality/pair_diffs.jsonl
artifacts/g0_pair_locality/most_diffuse_200.jsonl
artifacts/g0_behavior_cot/records.jsonl
artifacts/g0_behavior_cot/invalid_examples.jsonl
artifacts/g0_behavior_cot/summary.json
artifacts/g0_behavior_direct/records.jsonl
artifacts/g0_behavior_direct/invalid_examples.jsonl
artifacts/g0_behavior_direct/summary.json
```

## What positive G0s prove

If G0a+b+c all pass, we may claim only:

> The released counterfactual pairs are sufficiently local for aligned analysis; the published Bias Trap phenomenon reproduces on a seed-supported open model under a valid Qwen3 thinking regime; and a dense subset of the same exact old-diagnosis persistence events also exists without variable-length CoT.

This gives a tractable mechanism object. It still does **not** prove that the new evidence is internally encoded.

## If all G0s pass: next identification step

Before training any generic probe, freeze an explicit evidence/update experiment.

1. Use `pair_diffs.jsonl` to align the changed spans within each control/trap pair.
2. Restrict mechanism analysis to direct-mode cases so answer sites are fixed.
3. Use correctly updated trap cases as positive controls for an evidence-sensitive internal transition.
4. Compare them against exact Bias Trap cases at a **small predeclared site set**.
5. Prefer same-pair or diagnosis-transition-matched causal patching/ablation over a global learned steering vector.

A valid updating-failure result must show more than decodability. The evidence-related state must survive a manipulation check and the downstream diagnostic state must remain causally biased toward the old diagnosis.

If identifying the decisive internal state requires broad matching, model/layer/token sweeps, or a GPT/API evidence annotator, stop and reconsider the topic.

## Mechanism kill lines

Stop if:

- repaired G0b completes with healthy measurement but fails the frozen seed gate;
- direct-mode exact Bias Trap events are sparse;
- intended changed spans cannot be aligned at useful density;
- correctly updated trap positive controls do not show a recoverable evidence-sensitive state;
- matching donor/recipient cases requires many post-hoc covariates;
- only broad layer/token/coefficient search finds rescue;
- intervention works only by overwriting the final answer representation.

Do **not** archive on `MEASUREMENT_RUNTIME_FAILURE`; fix only the demonstrated measurement/runtime defect first.

## Collision boundary

This is not another medical-bias benchmark, not another prompt-mitigation paper, and not generic anchoring/sycophancy probing.

The scoped question is:

> **Within exact counterfactual Bias Trap pairs, can we causally separate failure to encode discriminative evidence from failure to use evidence-related internal state to update the diagnosis?**

## Method opening

A genuine evidence-integration bottleneck gives a concrete target for counterfactual evidence-update consistency training, revision-sensitive representation regularization, or an evidence-triggered revision mechanism.

## Files

- `g0_pair_locality.py` — full test-pair structure and aligned diff audit.
- `g0_bias_trap_screen.py` — repaired Qwen3 CoT and direct-answer Bias Trap screens.
- `run_g0.sh`
- `requirements.txt`
- `tests/test_g0_helpers.py`
- `G0_RESULTS.md` — provenance for the invalid first G0b and rerun instructions.
- `VALIDATION_AUDIT.md` — identification and measurement audit.

## Scientific invariant

> **same released patient pair; control is correct; trap ground truth flips; model persists on the old control diagnosis. First reproduce that exact event with a valid Qwen3 inference/evaluation stack, then require it to exist in a mechanism-tractable direct regime before asking what the model encoded.**
