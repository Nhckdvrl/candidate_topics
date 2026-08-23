# 21 — Where Does Long-Context Semantic Execution Break?

**Status: CANDIDATE / FROZEN G0 READY**

## Natural question

A model can sometimes **find the right code in a long context but still fail to execute what that code means**.

> **When the same code remains lexically accessible in the middle of a long context, where does its operational computation break?**

The critical same-instance contrast is:

- target program is semantically solved at the context edge;
- the target statement is still lexically retrievable in the middle;
- the same program is semantically wrong in the middle.

If this cell is sparse, stop before mechanism work.

## Seed

ACL 2026 long paper: **Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding**.

- ACL: https://aclanthology.org/2026.acl-long.19/
- Official code: https://github.com/adamstorek/long-context-code-understanding

The seed already establishes severe positional degradation of semantic recall while lexical recall remains much stronger. Its synthetic sequential programs have exact outputs and exact intermediate states, so no human annotation is needed.

## New question

We are not redoing lost-in-the-middle and not asking generically how transformers track state.

The new computational distinction is:

> **Does middle position prevent the model from forming the required intermediate state, prevent that state from propagating through later transitions, or preserve the state but fail at final readout?**

## Why this is feasible

- ACL main seed.
- Open local models in the 7B–32B regime.
- Official reproduction package.
- No paid API requirement.
- No manual labels: program outputs and intermediate states are exact.
- Same-program positional intervention.
- Large seed effect, so G0 is not betting on a brand-new phenomenon.

## Frozen G0

`g0_position_dissociation.py` builds SemTrace-style sequential programs and the same distractor context in two positions: `start` and `middle`.

It asks two questions per position:

1. **Lexical:** copy one exact assignment line.
2. **Semantic:** execute the target function and output its exact integer array.

Primary event:

```text
semantic(start) = correct
lexical(middle) = correct
semantic(middle) = wrong
```

Default model:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Default screen:

```text
64 items, ~8192-token contexts, seed 20260823
```

### Frozen gate

Proceed only if all hold:

- start semantic accuracy `>= 0.50`;
- middle lexical accuracy `>= 0.80`;
- at least `16` critical-cell examples;
- critical-cell rate among eligible examples `>= 0.20`.

Eligible means `semantic(start)=correct AND lexical(middle)=correct`.

If this fails, do not rescue by sweeping models, context lengths, prompts, or layers.

## Run

```bash
cd 21_semtrace_semantic_state_failure
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_g0.sh
```

Outputs:

```text
artifacts/g0/records.jsonl
artifacts/g0/summary.json
```

## If G0 passes

Mechanism work proceeds in this order:

1. **State decodability:** at a small predeclared set of depth fractions, test whether exact intermediate program state is represented.
2. **Same-item activation patching:** patch state from the same program at the successful edge condition into the failed middle condition.
3. **Transition localization:** find the earliest program transition where middle computation becomes irrecoverable.

Prefer natural same-item patching before any learned global steering vector. Topic 20 already taught us that a perfectly decodable direction can be causally inert.

## Mechanism kill lines

Stop if:

- lexical-middle support disappears;
- state is not locally recoverable even in edge-correct runs;
- same-item edge→middle patching gives no rescue under a frozen small site set;
- rescue requires broad layer/token/coefficient search;
- the effect disappears under exactly matched context content/length.

## Method opening

If a specific state-propagation bottleneck exists, possible follow-up methods include position-robust state-consistency training or semantic-state routing. The diagnostic has exact intermediate-state and behavioral ground truth.

## Files

- `g0_position_dissociation.py`
- `run_g0.sh`
- `requirements.txt`
- `tests/test_g0_helpers.py`

## Scientific invariant

> **same program + same distractors + same model; edge semantic success, middle lexical success, middle semantic failure.**
