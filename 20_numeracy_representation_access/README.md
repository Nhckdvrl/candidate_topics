# 20 — Representation or Access? Why Can LLMs Encode Numerical Magnitude but Fail to Use It?

**Status: CANDIDATE / G0 PASS / G1 FROZEN**

## Natural question

A system can fail because the relevant information was never represented correctly, or because the information is present internally but the decision process fails to use it. For numerical reasoning this is the old and natural distinction between **representation deficit** and **access deficit**.

The concrete question is:

> **When an LLM fails to compare two numbers written in different notations, is the numerical ordering absent from its representation, or is the ordering already present but not used by generation?**

The project is deliberately not a generic probing paper. The seed already establishes a large representation/behavior gap; our contribution begins at the stronger same-prompt, same-instance dissociation and asks whether the readable ranking state is causally connected to output.

---

## Seed

Fengting Yuchi, Li Du, Jason Eisner. **LLMs Know More About Numbers than They Can Say.** EACL 2026 Oral / Short.

- Paper: https://aclanthology.org/2026.eacl-short.47/
- arXiv: https://arxiv.org/abs/2602.07812
- Official code: https://github.com/VCY019/Numeracy-Probing
- Upstream revision used here: `9e1be04b69965662886c79d543936389c5407d27`

The seed reports on the primary `int-sci` setting for Qwen3-8B roughly:

```text
one-shot verbalization = 70.00%
zero-shot classifier probe = 98.88%
```

But that cross-condition gap is not itself an access result: probing is zero-shot, verbalization is one-shot, and the appendix shows prompt-position effects. Therefore Topic 20 first required the stronger object to exist under the **same balanced five-shot prompt**.

---

# G0 — same-prompt mechanism prerequisite

Frozen model and data:

```text
model: Qwen/Qwen3-8B
HF snapshot: b968826d9c46dd6066d109eabc6255188de91218
dataset: official seed-0 int_sci_compare
train / val / test: 8000 / 1600 / 1600
prompt: exact official balanced 5-shot int-sci prompt
hidden state: final prompt token
hard regime: |log2(a/b)| < 0.1
```

The primary object was the instance-level cell:

```text
probe correct
AND
generation wrong
```

not merely an aggregate accuracy gap.

## Frozen G0 result

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

The G0-selected best decoding layer was layer 36 (zero-based block 35), with validation accuracy `0.999375`.

All preregistered G0 conditions passed. See [`G0_RESULTS.md`](./G0_RESULTS.md) and [`artifacts/g0/`](./artifacts/g0/).

### Important interpretation boundary

G0 proves a real **same-prompt representation/behavior dissociation**: on almost every generation error in the hard set, the correct ordering is still linearly readable immediately before generation.

G0 does **not** prove that the probe direction is the model's native causal decision variable. That is G1.

---

# Post-G0 audit

Manual inspection revealed two facts that must not be silently folded into the original claim:

1. at least one exact displayed hard critical pair is duplicated in the released seed-0 test set;
2. the hard critical errors appear to overwhelmingly choose the scientific-notation operand.

Because both observations were made after inspecting seed-0 test, they are exploratory. `post_g0_audit.py` records duplicates and notation-choice structure, but all new scientific claims must be confirmed on a fresh seed.

This is why G1 does **not** reuse the 38 seed-0 critical cases as confirmatory evidence.

---

# G1 — causal access

The frozen protocol is [`G1_CAUSAL_ACCESS.md`](./G1_CAUSAL_ACCESS.md).

Fresh confirmation:

```text
fresh generator seed = 20260824
same Qwen3-8B snapshot
same int-sci task
same exact five-shot prompt
```

Exact displayed duplicates are counted once for inferential statistics; exact numerical ties are excluded without replacement. Seed-0 train/validation remain the only data used to fit the ranking probe.

## G1-P0: fresh-object replication

Before intervention, the fresh unique hard set must satisfy:

- `N_hard >= 100`;
- frozen seed-0 probe hard accuracy `>= 0.90`;
- at least `25` unique `probe-correct / generation-wrong` examples;
- critical rate `>= 0.20`;
- invalid generation `< 5%`.

Otherwise: `STOP_G1_NONREPLICATION`. No seed/model/prompt rescue.

## G1 causal layer

The G0 maximum was at the final layer, which leaves little downstream computation after intervention. Therefore G1 uses a rule chosen solely from the already-frozen seed-0 validation curve:

> earliest layer with validation ranking-probe accuracy `>= 0.99`.

This fixes:

```text
L_sat = layer 20
zero-based block = 19
seed-0 validation probe accuracy = 0.990625
```

No G1 layer sweep is allowed.

## G1 rank reflection

At `L_sat`, for frozen logistic probe

```text
m(h) = w^T h + b
```

apply the minimum-L2 reflection across its hyperplane:

```text
h_flip = h - 2 m(h) / ||w||^2 * w
```

There is no steering coefficient to tune.

Primary population: fresh unique hard examples that are originally both probe-correct and generation-correct.

Primary outcome: after reflection, does an originally correct output flip to the **opposite original operand**?

Null: eight fixed Gaussian directions (`20260831...20260838`), each orthogonal to the ranking direction and scaled per example to exactly the same L2 perturbation norm.

Define:

```text
F_rank = opposite-operand flip rate under rank reflection
F_null = mean flip rate under 8 norm-matched random nulls
DeltaF = F_rank - F_null
```

### Frozen G1 verdicts

`RANK_DIRECTION_CAUSAL` only if:

- probe sign flip succeeds on `>= 99%`;
- `DeltaF >= 0.20`;
- paired bootstrap 95% CI lower bound `> 0`;
- at least `80%` of changed rank-reflection outputs remain one of the two original operands rather than garbage.

Strong null:

```text
READABLE_BUT_NOT_CAUSALLY_USED_AT_LSAT
```

if `DeltaF <= 0.05` and CI upper bound `<= 0.10`.

Otherwise:

```text
INCONCLUSIVE_DO_NOT_TUNE
```

Executable implementation: [`g1_rank_reflection.py`](./g1_rank_reflection.py). Frozen runner: [`run_g1.sh`](./run_g1.sh).

---

# Conditional notation-competition branch

The seed-0 error inspection suggests a potentially sharper mechanism:

> the model may compute the correct magnitude ordering but generation may follow a competing notation-format route.

This is **not yet a result**. It was discovered on the seed-0 test set.

Only if the fresh seed confirms that at least `80%` of hard exact-operand errors choose the scientific-notation operand may G1-P4 test a notation-side representation at the same frozen layer while preserving the ranking projection. A failure of the main rank-causality test cannot be rescued by searching notation subspaces.

---

# Novelty boundary

The seed already establishes:

- numerical magnitude is linearly recoverable;
- pairwise ranking is highly decodable;
- mixed-notation verbal comparison is worse;
- probe-aware finetuning can improve behavior.

A nearby 2026 mechanistic-interpretability paper studies ordinal/numeric representation geometry with activation patching. Therefore **activation patching numeric representations is not itself the novelty**.

The protected question is narrower:

> **When the same computation contains the correct ranking but generation chooses wrongly, is that ranking coordinate causally used by the generated decision, and if not, what competing readout dominates it?**

---

# Resource fit and stop rule

```text
paid API: 0
new annotation: 0
foundation-model training: 0
open-weight GPU mechanism analysis: yes
```

If fresh G1-P0 fails, stop. If the frozen rank intervention gives a strong null, accept it. Do not search layer × token × strength × prompt × model until something becomes positive.

Canonical files:

- [`G0_RESULTS.md`](./G0_RESULTS.md)
- [`G1_CAUSAL_ACCESS.md`](./G1_CAUSAL_ACCESS.md)
- [`post_g0_audit.py`](./post_g0_audit.py)
- [`g1_rank_reflection.py`](./g1_rank_reflection.py)
- [`run_g0.sh`](./run_g0.sh)
- [`run_g1.sh`](./run_g1.sh)
