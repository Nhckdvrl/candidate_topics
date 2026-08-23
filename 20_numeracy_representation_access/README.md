# 20 — Representation or Access? Why Can LLMs Encode Numerical Magnitude but Fail to Use It?

**Status: CANDIDATE / G0 PASS / G1 STOP / G2 NOTATION COMPETITION FROZEN**

## Natural question

A system can fail because the relevant information was never represented correctly, or because the information is present internally but the decision process fails to use it. For numerical reasoning this is the old distinction between **representation deficit** and **access deficit**.

Topic 20 asks:

> **When an LLM fails to compare two numbers written in different notations, is the ordering absent internally, or is the correct ordering already present but defeated by another decision signal?**

The project is not a generic probing paper. The seed already establishes a representation/behavior gap. Our contribution begins with same-prompt, same-instance failures and then asks what actually controls the output.

---

## Seed

Fengting Yuchi, Li Du, Jason Eisner. **LLMs Know More About Numbers than They Can Say.** EACL 2026 Oral / Short.

- Paper: https://aclanthology.org/2026.eacl-short.47/
- arXiv: https://arxiv.org/abs/2602.07812
- Official code: https://github.com/VCY019/Numeracy-Probing
- Upstream revision used here: `9e1be04b69965662886c79d543936389c5407d27`

The seed reports on Qwen3-8B in the primary `int-sci` setting roughly:

```text
one-shot verbalization = 70.00%
zero-shot classifier probe = 98.88%
```

That cross-condition gap was not enough for us because the prompts differed and the paper itself reports answer-position effects under one-shot prompting. Topic 20 therefore first required a same-prompt mechanism-level object.

---

# G0 — same-prompt mechanism prerequisite

Frozen setting:

```text
model: Qwen/Qwen3-8B
HF snapshot: b968826d9c46dd6066d109eabc6255188de91218
dataset: official seed-0 int_sci_compare
train / val / test: 8000 / 1600 / 1600
prompt: exact official balanced 5-shot int-sci prompt
hidden state: final prompt token
hard regime: |log2(a/b)| < 0.1
```

Primary object:

```text
probe correct
AND
generation wrong
```

## G0 result

**Verdict: `GO_CAUSAL_G1`**

| subset | N | probe accuracy | generation accuracy | gap | invalid |
|---|---:|---:|---:|---:|---:|
| full test | 1600 | 0.996875 | 0.817500 | 0.179375 | 0% |
| hard test | 129 | 0.961240 | 0.682171 | 0.279070 | 0% |

Hard 2×2:

| | generation correct | generation wrong |
|---|---:|---:|
| probe correct | 86 | **38** |
| probe wrong | 2 | 3 |

So:

```text
N_critical = 38 / 129
R_critical = 0.294574
error coverage = 38 / 41 = 0.926829
```

G0 therefore established a real same-prompt representation/behavior dissociation. See [`G0_RESULTS.md`](./G0_RESULTS.md).

---

# Post-G0 discovery

Inspection of the locked seed-0 hard errors revealed a sharper pattern:

> **when generation is wrong, it appears to overwhelmingly choose the operand written in scientific notation.**

Because this was noticed after looking at seed 0, it was exploratory only.

The original G1 protocol therefore preregistered an independent descriptive confirmation on fresh seed `20260824`: among hard errors that exactly equal one of the two operands, scientific-operand choice rate had to be at least `0.80` before any notation mechanism could be considered.

---

# G1 — original rank-causality route

Protocol: [`G1_CAUSAL_ACCESS.md`](./G1_CAUSAL_ACCESS.md)

G1 used a fresh test seed `20260824` and a seed-0-trained ranking probe at the predeclared saturation layer:

```text
L_sat = layer 20
zero-based block = 19
seed-0 validation rank-probe accuracy = 0.990625
```

Before intervention, G1-P0 required the fresh unique hard subset to satisfy all:

```text
N_hard >= 100
frozen rank-probe hard accuracy >= 0.90
unique critical >= 25
critical rate >= 0.20
invalid generation < 5%
```

## G1-P0 result

**Historical verdict: `STOP_G1_NONREPLICATION`**

Fresh seed `20260824`:

```text
raw test                  = 1600
unique test               = 1598
hard                      = 138
full frozen-probe acc     = 0.989987
hard frozen-probe acc     = 0.898551 = 124/138
hard generation acc       = 0.565217
hard critical             = 51/138
invalid                   = 0%
```

Only the `>=0.90` hard-probe point threshold failed, by one correctly classified hard example. Per preregistration, rank reflection and its eight random nulls were not run.

This must remain recorded as a stopped G1. We do **not** change the threshold to rescue rank reflection.

See [`G1_RESULTS.md`](./G1_RESULTS.md).

---

# Independent confirmation of the notation attractor

The same untouched seed `20260824` independently confirmed the post-G0 exploratory observation:

```text
hard generation errors             = 60
exact-operand hard errors           = 60
errors choosing scientific operand  = 55
scientific-operand error rate       = 55/60 = 0.916667
```

This exceeds the preregistered descriptive confirmation threshold of `0.80` by a large margin.

Crucially, this is not the seed paper's known one-shot answer-position bias. The official balanced 5-shot prompt alternates correct answer position precisely to reduce position bias. Here the failure follows **notation form**: whichever side is rendered in scientific notation tends to attract the wrong output.

Thus Topic 20 now has a cleaner empirical sequence:

```text
seed 0        -> exploratory discovery of notation-side attraction
seed 20260824 -> independent confirmation at 91.67%
seed 20260825 -> untouched causal mechanism test
```

---

# G2 — notation competition

Frozen protocol: [`G2_NOTATION_COMPETITION.md`](./G2_NOTATION_COMPETITION.md)

Executable implementation: [`g2_notation_competition.py`](./g2_notation_competition.py)

Frozen runner: [`run_g2.sh`](./run_g2.sh)

## G2 question

> **Does a representation of which operand is written in scientific notation causally compete with an already-correct numerical ranking at the decision stage?**

G2 does not reopen the stopped G1 rank threshold. It uses the independently confirmed notation phenomenon as a new mechanism target.

### Fresh causal seed

```text
seed = 20260825
setting = int_sci_compare
same Qwen3-8B snapshot
same balanced official 5-shot prompt
```

No seed search.

### Object gate

The untouched seed must have:

```text
unique hard >= 100
hard exact-operand generation errors >= 30
scientific-operand error rate >= 0.80
```

The primary causal population then requires at least 25 unique hard cases satisfying:

```text
ranking probe correct
baseline generation wrong
baseline answer exactly equals an input operand
baseline answer is the scientific-notation operand
```

If support is insufficient, stop. No model/prompt/seed rescue.

### Frozen causal coordinates

At the same fixed layer 20:

1. fit the original seed-0 ranking probe `w_rank`;
2. fit a seed-0 classifier for whether the scientific operand is on side A or B;
3. remove the notation direction's projection onto `w_rank`;
4. freeze a one-dimensional notation threshold from seed-0 train only;
5. require seed-0 validation notation-side accuracy `>=0.95` and near-zero cosine with the ranking direction.

### Primary intervention

Neutralize only the notation coordinate:

```text
h_neutral = h - (u_not^T h - tau_not) u_not
```

with `u_not` constructed orthogonal to the ranking direction. Thus the intended ranking projection is preserved while the notation-side signal is removed.

Primary outcome:

```text
wrong scientific operand -> correct ordinary operand
```

Compare against 8 fixed per-example norm-matched random directions orthogonal to both ranking and notation coordinates.

Define:

```text
R_not  = wrong->correct rescue rate under notation neutralization
R_null = mean rescue rate under 8 matched random nulls
DeltaR = R_not - R_null
```

### Positive gate

`NOTATION_COMPETITION_CAUSAL` only if all hold:

```text
notation neutralization manipulation succeeds
ranking-logit preservation succeeds
DeltaR >= 0.20
bootstrap 95% CI lower bound > 0
invalid/neither-operand rate < 0.10
>=80% of changed valid outputs move to the correct ordinary operand
```

Strong null:

```text
NOTATION_READABLE_BUT_NOT_CAUSAL_AT_LSAT
```

if `DeltaR <= 0.05` and bootstrap CI upper bound `<=0.10`, with manipulation checks passing.

Otherwise:

```text
INCONCLUSIVE_DO_NOT_TUNE
```

No layer × token × coefficient × subspace × prompt × model search.

---

# Why G2 is scientifically cleaner than changing the G1 threshold

The `0.898551` fresh hard rank-probe accuracy is practically indistinguishable from the preregistered `0.90` cutoff, but the original G1 contract was explicit, so we preserve its stop.

G2 does not redefine that gate. It follows a different observation that:

1. was discovered on seed 0;
2. had a confirmation criterion frozen before fresh evaluation;
3. independently replicated at `55/60 = 91.67%`;
4. yields a concrete causal intervention with an interpretable behavioral rescue outcome.

That is a much cleaner scientific reason to continue than changing `0.90` to a convenient number after seeing the data.

---

# Novelty boundary

The EACL 2026 seed already establishes:

- internal numerical magnitude and ranking are decodable;
- mixed-notation verbal comparison is difficult;
- one-shot answer-position bias exists in several models;
- probe-aware finetuning improves behavior.

Nearby work also performs activation intervention on ordinal/numeric representations, so `patching numbers` is not itself novel.

The protected Topic-20 direction is now narrower:

> **A model can have the correct numerical ranking yet choose according to a task-irrelevant notation-form signal; is that notation signal a causal competitor in the readout?**

A positive G2 would transform a broad representation/access dissociation into a specific mechanism and a direct method target: suppress or regularize notation-dependent readout while preserving magnitude information.

---

# Resource fit

```text
paid API: 0
new annotation: 0
foundation-model training for mechanism gate: 0
open-weight GPU mechanism analysis: yes
```

Canonical files:

- [`G0_RESULTS.md`](./G0_RESULTS.md)
- [`G1_CAUSAL_ACCESS.md`](./G1_CAUSAL_ACCESS.md)
- [`G1_RESULTS.md`](./G1_RESULTS.md)
- [`G2_NOTATION_COMPETITION.md`](./G2_NOTATION_COMPETITION.md)
- [`post_g0_audit.py`](./post_g0_audit.py)
- [`g1_rank_reflection.py`](./g1_rank_reflection.py)
- [`g2_notation_competition.py`](./g2_notation_competition.py)
- [`run_g0.sh`](./run_g0.sh)
- [`run_g1.sh`](./run_g1.sh)
- [`run_g2.sh`](./run_g2.sh)
