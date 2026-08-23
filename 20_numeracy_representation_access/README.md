# 20 — Representation or Access? Why Can LLMs Encode Numerical Magnitude but Fail to Use It?

**Status: ARCHIVED / ROBUST BEHAVIORAL PHENOMENON, FROZEN CAUSAL ROUTES FAILED**

This topic is closed. See [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) for the full decision record.

## Natural question

> **When an LLM fails to compare two numbers written in different notations, is the correct ordering absent internally, or is it represented but not used by generation?**

Seed paper: Yuchi, Du, and Eisner, **LLMs Know More About Numbers than They Can Say** (EACL 2026).

- Paper: https://aclanthology.org/2026.eacl-short.47/
- Official code: https://github.com/VCY019/Numeracy-Probing
- Model: `Qwen/Qwen3-8B`
- Primary task: official `int_sci_compare`
- Prompt: exact balanced official 5-shot prompt

## What survived

### G0: same-prompt representation/behavior dissociation

G0 passed strongly:

```text
hard N                         = 129
hard probe accuracy            = 0.961240
hard generation accuracy       = 0.682171
probe-correct/generation-wrong = 38/129
error coverage                 = 38/41 = 92.68%
```

So the project did not fail because the object was absent. The correct numerical ordering remained linearly readable on almost every hard generation error under the same prompt and same pre-generation state.

See [`G0_RESULTS.md`](./G0_RESULTS.md).

### Scientific-notation attractor

A post-G0 exploratory pattern then replicated on two untouched seeds:

```text
seed 20260824: 55/60 hard exact-operand errors chose scientific notation = 91.67%
seed 20260825: 39/41 hard exact-operand errors chose scientific notation = 95.12%
```

This is a real behavioral phenomenon.

## What failed

### G1: original rank-causality route

Fresh seed `20260824` produced:

```text
hard frozen-probe accuracy = 124/138 = 0.898551
frozen threshold           = >= 0.90
```

Only this preregistered point threshold failed; all other fresh-object conditions passed. Per protocol, rank reflection was not run.

Historical verdict:

`STOP_G1_NONREPLICATION`

See [`G1_RESULTS.md`](./G1_RESULTS.md).

### G2: notation-competition causal route

On untouched seed `20260825`, the notation representation checks were extremely strong:

```text
notation validation accuracy = 1.000000
notation/rank cosine          = 1.96e-17
rank preservation            = PASS
notation neutralization       = PASS
```

But the frozen intervention produced:

```text
primary population              = 32
notation-neutralization rescue  = 0/32
8 norm-matched random nulls     = 0/32 each
DeltaR                          = 0
95% bootstrap CI                = [0, 0]
invalid/neither                 = 0%
```

Frozen verdict:

`NOTATION_READABLE_BUT_NOT_CAUSAL_AT_LSAT`

See [`G2_RESULTS.md`](./G2_RESULTS.md).

## Final decision

The behavioral object is strong, but the clean causal story did not survive. Continuing would require opening a post-hoc search over layer, token, strength, nonlinear subspace, prompt, or model. That violates the project's no-rescue rule.

The remaining notation-attractor observation is interesting but insufficient by itself for the intended paper scale, especially given nearby 2026 work establishing broad numeral-format sensitivity.

Therefore:

**ARCHIVE. Do not continue layer/token/subspace/model search under Topic 20.**

## Canonical record

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md)
- [`G0_RESULTS.md`](./G0_RESULTS.md)
- [`G1_CAUSAL_ACCESS.md`](./G1_CAUSAL_ACCESS.md)
- [`G1_RESULTS.md`](./G1_RESULTS.md)
- [`G2_NOTATION_COMPETITION.md`](./G2_NOTATION_COMPETITION.md)
- [`G2_RESULTS.md`](./G2_RESULTS.md)
- `artifacts/g0/`
- `artifacts/g1/`
- `artifacts/g2/`

The scientific-notation attractor may be retained as a clue for a future independently motivated topic, but it does not reopen this one.