# Archive Summary — Topic 20: Representation or Access?

**Final status: ARCHIVED / ROBUST BEHAVIORAL PHENOMENON, FROZEN CAUSAL ROUTES FAILED**

## Original question

> When an LLM fails to compare two numbers written in different notations, is the correct numerical ordering absent internally, or is it already represented but not used by generation?

The topic was seeded by Yuchi, Du, and Eisner, **LLMs Know More About Numbers than They Can Say** (EACL 2026), which reports a large gap between linearly decodable numerical ranking and explicit mixed-notation comparison.

The project deliberately tightened that observation into a same-prompt, same-instance mechanism question rather than treating cross-condition probe/generation numbers as an access result.

---

## G0 — the experimental object was real

Frozen setting:

- model: `Qwen/Qwen3-8B`
- model snapshot: `b968826d9c46dd6066d109eabc6255188de91218`
- official `int_sci_compare`, seed `0`
- exact balanced official 5-shot prompt
- hard regime: `|log2(a/b)| < 0.1`

Result:

| subset | N | probe accuracy | generation accuracy | gap |
|---|---:|---:|---:|---:|
| full | 1600 | 0.996875 | 0.817500 | 0.179375 |
| hard | 129 | 0.961240 | 0.682171 | 0.279070 |

Hard critical cell:

```text
probe correct / generation wrong = 38 / 129
error coverage = 38 / 41 = 92.68%
```

So the project did **not** fail because the representation/behavior dissociation was absent. Under the same prompt and same pre-generation state, correct ranking remained linearly readable on almost every hard generation error.

G0 verdict: `GO_CAUSAL_G1`.

---

## G1 — the original rank-causality route stopped at fresh replication

G1 used a fresh untouched seed `20260824` and a frozen seed-0 ranking probe at the predeclared saturation layer 20.

Fresh object:

```text
unique hard                = 138
hard frozen-probe accuracy = 124/138 = 0.898551
hard generation accuracy   = 0.565217
hard critical              = 51/138 = 36.96%
invalid                     = 0%
```

The preregistered hard-probe threshold was `>= 0.90`, which would require `125/138`. The run therefore missed the point threshold by exactly one hard example.

Historical verdict: `STOP_G1_NONREPLICATION`.

Per protocol, rank reflection and its random controls were not run. This must not be rewritten as a causal null, and the threshold must not be relaxed after seeing the result.

Scientifically, the failure was narrow: the behavioral object remained large, but the exact frozen prerequisite for the rank-reflection experiment did not pass.

---

## Post-G0 discovery — a robust scientific-notation attractor

Inspection of seed-0 errors suggested a sharper behavioral pattern: when the model was wrong, it tended to choose the operand written in scientific notation.

Because this was discovered after inspecting seed 0, it was treated as exploratory and required untouched confirmation.

Fresh seed `20260824` confirmed:

```text
hard exact-operand errors          = 60
errors choosing scientific operand = 55
scientific-operand error rate      = 91.67%
```

A second untouched causal seed `20260825` confirmed again:

```text
hard exact-operand errors          = 41
errors choosing scientific operand = 39
scientific-operand error rate      = 95.12%
```

Thus the notation-side behavioral attractor is real and stable across discovery plus two untouched seeds.

This is not merely the seed paper's one-shot answer-position effect: the project used the balanced 5-shot prompt, while the error followed whichever side carried scientific notation.

---

## G2 — the obvious notation coordinate was readable but causally inert

G2 asked whether a one-dimensional representation of **which side is written in scientific notation** causally competes with the already-correct numerical ranking at the frozen layer-20 decision site.

The notation classifier was trained only on seed-0 data and orthogonalized against the frozen ranking direction.

Representation checks were exceptionally strong:

```text
notation validation accuracy = 1.000000
notation/rank cosine          = 1.96e-17
rank preservation            = PASS
notation neutralization       = PASS
```

Primary causal population on untouched seed `20260825`:

```text
N = 32
ranking probe correct
baseline generation wrong
wrong answer = scientific-notation operand
```

Causal result:

```text
notation neutralization rescue = 0/32
8 norm-matched random nulls     = 0/32 each
DeltaR                          = 0.000000
bootstrap 95% CI                = [0.000000, 0.000000]
invalid/neither                 = 0%
```

Frozen verdict:

`NOTATION_READABLE_BUT_NOT_CAUSAL_AT_LSAT`

This is a clean null for the tested one-dimensional notation coordinate at the frozen layer/token. It does **not** prove that no notation-related mechanism exists anywhere in the network. It does prove that the most direct, predeclared causal continuation of the newly discovered behavioral attractor failed completely.

---

## Why the topic is archived

The project now has a precise split outcome:

1. **The behavioral phenomenon is strong.** Correct ranking is often decodable when output is wrong, and scientific notation is a highly reproducible wrong-answer attractor.
2. **The clean causal stories did not earn continuation.** The original rank route stopped at its frozen fresh prerequisite; the independent notation route produced an exact zero rescue under a manipulation whose representation checks all passed.

Continuing from here would require opening a search over layer, token, intervention strength, nonlinear subspace, feature family, prompt, or model. That is exactly the post-hoc mechanism search the project rules were designed to prevent.

The remaining behavioral observation is also too narrow, by itself, for the intended paper scale. Recent work such as **1,729 vs. 1729: The Effect of Scripts and Formats on LLM Numeracy** (ACL Findings 2026) already establishes broad format sensitivity in LLM numeracy. The specific scientific-notation attractor is interesting, but without a validated mechanism or stronger method opening it is not enough reason to continue this candidate.

Therefore the correct project decision is:

**ARCHIVE. Do not rescue with a layer/token/subspace/model sweep.**

---

## Failure / stop type

**Layer D — strong prerequisite phenomenon, failed explanatory/causal axis.**

This is not a phenomenon-existence failure and not an implementation failure. The best description is:

> **The behavior replicated more strongly than expected, but the most natural frozen causal readouts did not control it.**

---

## Main lessons

1. **Decodability plus behavioral alignment is still not mechanism.** A feature can be perfectly linearly readable, perfectly orthogonal to the target variable, and behaviorally predictive, yet neutralizing that feature can do absolutely nothing.
2. **A strong phenomenon does not license an open-ended mechanism hunt.** The notation attractor survived two untouched seeds, but the first clean causal hypothesis returned `0/32`. That is a reason to stop, not to search layers until one moves.
3. **Preregistered point thresholds should remain historical even when they miss by one sample.** G1's `124/138` versus `125/138` boundary was practically tiny but procedurally real. The right response was to preserve the stop, not redefine `0.90` after observing `0.898551`.
4. **A genuinely new observation may motivate one new frozen branch, but not unlimited branches.** The notation attractor was discovered on seed 0, confirmed on seed 20260824, and therefore legitimately motivated G2 on seed 20260825. Once that branch produced a clean null, the rescue budget was exhausted.
5. **Separate behavioral novelty from paper-level significance.** A crisp new bias can be real without supporting a full research program, especially when nearby literature already establishes the broader phenomenon class.
6. **The best negative mechanism result is one that cannot be blamed on plumbing.** G2 had 100% notation decoding, near-zero rank/notation cosine, verified neutralization, verified rank preservation, zero invalid outputs, and exact zero rescue. That makes the stop trustworthy.

---

## Preserved artifacts

- `G0_RESULTS.md`
- `G1_CAUSAL_ACCESS.md`
- `G1_RESULTS.md`
- `G2_NOTATION_COMPETITION.md`
- `G2_RESULTS.md`
- `artifacts/g0/`
- `artifacts/g1/`
- `artifacts/g2/`
- `g1_rank_reflection.py`
- `g2_notation_competition.py`
- `run_g0.sh`
- `run_g1.sh`
- `run_g2.sh`

The empirical notation attractor may be useful as a clue for a future, independently motivated project, but Topic 20 itself is closed.