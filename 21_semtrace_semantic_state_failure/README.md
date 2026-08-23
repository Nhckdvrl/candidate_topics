# 21 — Where Does Long-Context Semantic Execution Break?

**Status: CANDIDATE / FROZEN SEED-REPRO + G0 READY**

## Natural question

A model can sometimes **find the right code in a long context but still fail to execute what that code means**.

> **When the same code remains lexically accessible in the middle of a long context, where does its operational computation break?**

The critical same-instance contrast is:

- target program is semantically solved at the context edge;
- the target assignment is lexically retrievable both at the edge and in the middle;
- the same program produces a parseable but wrong semantic answer in the middle.

If this exact cell is sparse, stop before mechanism work.

## Seed

ACL 2026 long paper: **Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding**.

- ACL: https://aclanthology.org/2026.acl-long.19/
- Official code: https://github.com/adamstorek/long-context-code-understanding

The official package includes the forced-sequential synthetic output-prediction experiment (`fsyn_output_prediction`) built from `SimpleFunctionEval(force_sequential=True)` with CodeSearchNet distractors. The seed reports a strong position-dependent semantic-recall failure while lexical recall remains substantially stronger.

## New question

We are **not** redoing lost-in-the-middle and not asking generically how transformers track state.

The narrower computational distinction is:

> **Given that the relevant code is still retrievable, does middle position prevent formation of the required intermediate state, prevent state propagation through later transitions, or preserve the state but fail at final readout?**

G0 does not answer that mechanism question. It only establishes a dense, clean set of same-item failures on which the question is identifiable.

## Why this is feasible

- ACL main seed.
- Official reproduction package.
- Seed-supported open model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- No paid API requirement.
- No manual labels: outputs and intermediate program states are exact.
- Same-program positional intervention.
- Local GPUs are useful only after the exact behavioral object is locked.

## G0-0 — official seed reproduction is mandatory

Before running our custom paired screen, reproduce the official forced-sequential semantic-recall experiment on the **same model**.

From a clone of the official repository:

```bash
python -m long_context_understanding.experiments.fsyn_output_prediction \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --num-functions 80 \
  --num-contexts 800 \
  --position-step 8 \
  --seed 42
```

This writes a `summary.json` under the official `results/fsyn_output_prediction/...` directory.

Our `g0_upstream_contract.py` requires:

- at least three evaluated target positions;
- mean edge accuracy `>= 0.30`;
- edge-to-middle semantic accuracy drop `>= 0.20`.

This gate is intentionally broad: it verifies that the selected local stack reproduces the seed phenomenon before our mechanism-specific screen is interpreted.

If the official seed reproduction fails, stop the platform. Do not tune our custom generator to manufacture the effect.

## G0-1 — exact paired mechanism-support screen

`g0_position_dissociation.py` constructs deterministic forced-sequential target programs and one fixed distractor set per item, then places **the same target and the same distractors** at `start` vs token-centered `middle` positions.

The script records explicit context-contract checks:

- same distractor digest by construction;
- start/middle context token lengths differ by at most 16 tokens;
- start target center is within the first 12% of the context;
- middle target center lies in `[0.40, 0.60]`.

It asks two questions in each position:

1. **Lexical:** copy one exact assignment line.
2. **Semantic:** execute the target function and output its exact integer array.

Semantic failures count only if the model still emits a parseable integer list of the correct length. Pure formatting failures are excluded from the critical cell.

Primary event:

```text
context contract passes
AND semantic(start) = correct
AND lexical(start) = correct
AND lexical(middle) = correct
AND semantic(middle) is valid but wrong
```

Default model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Default screen:

```text
64 items, ~8192-token contexts, 8-step sequential programs, seed 20260823
```

### Frozen G0-1 gate

Proceed only if all hold:

- context-contract pass rate `= 1.00`;
- start semantic accuracy `>= 0.50`;
- start lexical accuracy `>= 0.80`;
- middle lexical accuracy `>= 0.80`;
- middle semantic invalid-output rate `<= 0.10`;
- aggregate semantic start→middle drop `>= 0.15`;
- at least `16` exact critical-cell examples;
- critical-cell rate among eligible examples `>= 0.20`.

If this fails after G0-0 passed, stop this mechanism object. Do not sweep model families, context lengths, prompts, layers, or parsing rules.

## What a positive G0 proves

Only this:

> On a seed-supported open model, there exists a dense set of same-program long-context cases where lexical access survives a position shift but exact operational execution fails.

It **does not** prove that an internal state is present, absent, or causally used. Those are G1 questions.

## Run

First reproduce G0-0 in the official repository, then:

```bash
cd 21_semtrace_semantic_state_failure
pip install -r requirements.txt
export UPSTREAM_SUMMARY=/path/to/official/results/fsyn_output_prediction/Qwen/Qwen2.5-Coder-7B-Instruct/80/42/summary.json
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_g0.sh
```

Outputs:

```text
artifacts/g0_upstream_contract.json
artifacts/g0/records.jsonl
artifacts/g0/summary.json
```

## If G0 passes

Mechanism work proceeds in this order:

1. **Exact state supervision:** the generator already records the numerical intermediate state after every assignment.
2. **Bounded state readout:** inspect only a predeclared small set of depth fractions and target-state token sites.
3. **Same-item natural patching:** use successful edge runs as donors for the failed middle run of the **same program**, rather than global learned steering directions.
4. **Transition localization:** identify the first sequential transition at which a recoverable edge state is no longer recoverable/usable in the middle condition.

The claim should be phrased operationally: formation / propagation / readout failure under a positional intervention. A probe alone is not a mechanism result.

## Mechanism kill lines

Stop if:

- the official seed does not reproduce;
- the paired critical cell is sparse;
- exact intermediate state is not recoverable even in edge-correct positive controls;
- same-item edge→middle patching gives no rescue at the frozen small site set;
- rescue requires broad layer/token/coefficient search;
- the effect disappears when context content and length are exactly matched.

## Method opening

If a specific state-propagation bottleneck exists, a later method can target position-robust state propagation or semantic-state consistency. The benchmark gives exact intermediate-state and behavioral ground truth for before/after evaluation.

## Files

- `g0_upstream_contract.py` — checks the official seed reproduction summary.
- `g0_position_dissociation.py` — same-content start/middle critical-cell screen.
- `run_g0.sh`
- `requirements.txt`
- `tests/test_g0_helpers.py`

## Scientific invariant

> **same program + same distractors + same model; edge semantic success, lexical success at both positions, parseable middle semantic failure. Only after that do we ask where the computation breaks.**
