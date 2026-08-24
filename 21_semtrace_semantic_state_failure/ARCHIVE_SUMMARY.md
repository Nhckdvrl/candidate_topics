# Topic 21 Archive Summary — SemTrace Semantic-State Failure

## Final decision

**ARCHIVED / STOP_UPSTREAM_SEED_NOT_REPRODUCED**

Topic 21 was registered to study a same-program dissociation in long-context code understanding: lexical access survives a start→middle position shift while exact semantic execution fails, enabling a later formation/propagation/readout analysis.

The project had a mandatory prerequisite: reproduce the official forced-sequential SemTrace positional phenomenon on the exact seed-supported local platform before running our custom paired G0.

That prerequisite completed and failed.

## Frozen official run

- Official repository commit: `0f8b327097f2a34bbc8d1c603480982e65053384`
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- HF revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Host: `fvcrc10`
- GPUs: 4 × A100 80 GB
- Seed: `42`
- `num-functions=80`
- `num-contexts=800`
- `position-step=8`
- Completed examples: `8,800`

Official command:

```bash
python -m long_context_understanding.experiments.fsyn_output_prediction \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --num-functions 80 --num-contexts 800 --position-step 8 --seed 42
```

## Result

All 11 frozen positions were present.

```text
position accuracy
0   0.000
8   0.000
16  0.000
24  0.000
32  0.000
40  0.000
48  0.000
56  0.000
64  0.000
72  0.000
80  0.00125
```

Frozen prerequisite gates:

```text
edge mean >= 0.30              observed 0.000625  FAIL
edge-to-middle drop >= 0.20    observed 0.000625  FAIL
>= 3 evaluated positions       observed 11        PASS
```

Contract verdict: `UPSTREAM_SEED_NOT_REPRODUCED`.

Per the preregistered protocol, the custom paired semantic/lexical G0 was **not run**. No prompt, model, parser, seed, context length, threshold, or subset rescue was attempted.

## What this means

This is a **platform/prerequisite failure for Topic 21**, not evidence that the ACL paper is false in general. It means the exact open-model/artifact regime selected for our mechanism project did not instantiate the prerequisite phenomenon strongly enough to support the intended causal story.

The candidate is therefore terminal in this repository unless genuinely new external evidence changes the experimental premise. It must not reappear in `advisor_topic_search/ACTIVE_CANDIDATES.md` merely because the abstract scientific question remains interesting.

## Reusable lesson

> **An externally reported phenomenon is not an internal experimental object until the exact seed cell reproduces on the system we intend to analyze. Artifact completeness is not reproduction.**

This is exactly why the repository now requires reproduction receipts before mechanism investment.

See also:

- `G0_RESULTS.md`
- `VALIDATION_AUDIT.md`
- `advisor_topic_search/REPRODUCTION_RECEIPT_POLICY.md`
