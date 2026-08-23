# 20 — Representation or Access? Why Can LLMs Encode Numerical Magnitude but Fail to Use It?

**Status: CANDIDATE / FROZEN G0 READY**

## The natural question

A system can fail at a task for two fundamentally different reasons:

1. the relevant information was never represented correctly; or
2. the information is present internally, but the decision process fails to use it.

For numerical reasoning, this is an old and natural distinction: **representation deficit vs. access deficit**.

Modern LLMs make the distinction unusually testable because we can inspect the hidden state immediately before generation and compare what is linearly available there with what the model actually says.

The concrete question is:

> **When an LLM fails to compare two numbers written in different notations, is the numerical ordering absent from its representation, or is the ordering already present but inaccessible to generation?**

This is not a generic "probe the model" project. The seed paper already establishes a large representation/behavior gap on exactly the open models we can run. The project starts one step later: test whether that gap survives in the **same model, same prompt, same decision regime**, then ask whether the readable ranking state is causally connected to output.

---

## Seed paper

Fengting Yuchi, Li Du, Jason Eisner. **LLMs Know More About Numbers than They Can Say.** EACL 2026 Oral / Short.

- Paper: https://aclanthology.org/2026.eacl-short.47/
- arXiv: https://arxiv.org/abs/2602.07812
- Official repository: https://github.com/VCY019/Numeracy-Probing

The paper reports, on the primary `int-sci` comparison setting:

```text
Qwen3-8B
one-shot verbalization accuracy ≈ 70.00%
zero-shot classifier-probe accuracy ≈ 98.88%
```

and similarly large gaps on several other 7B–8B open models.

The released project contains:

```text
construct_data.py
get_embeds.py
train_probe.py
verbalization.py
finetune.py
```

plus exact scripts for the tested open models.

This is unusually strong experimental inheritance: the phenomenon, target model, data generator, labels, probe recipe and behavioral evaluation are already public.

---

## Why the published gap is not yet enough

A closer audit found an important confound in the headline comparison:

- the paper's probing result uses a **zero-shot** prompt;
- the headline verbalization result uses a **one-shot** prompt;
- the appendix shows that some 7B models imitate the answer position in the one-shot demonstration;
- few-shot prompting reduces several of those positional artifacts.

Therefore we do **not** interpret `98.88% probe vs 70% generation` as direct evidence for an access bottleneck.

The stronger project-level prerequisite is:

```text
same Qwen3-8B
same balanced five-shot prompt
same input instance
same pre-generation hidden state

probe says correct ranking
BUT
that prompt's greedy generation is wrong
```

If this exact `probe-correct / generation-wrong` cell is not dense, the mechanism project dies before activation intervention.

---

## Frozen primary setting

### Model

```text
Qwen/Qwen3-8B
```

No model search in G0.

### Data

Use the official seed-0 `int_sci_compare` dataset only:

```text
8,000 train
1,600 validation
1,600 test
```

A static audit of the released generator found, for the published seed:

- no displayed ties;
- no ordering flips caused by five-significant-digit scientific formatting;
- approximately balanced correct-answer position;
- 129 / 1,600 test items in the seed paper's hard regime `|log2(a/b)| < 0.1`.

`dec_sci_compare` is **not** part of the survival gate. It is reserved for confirmation only after the seed-exact primary G0 passes.

### Prompt

Use the exact five `int-sci` demonstrations already implemented in the official `src/verbalization.py`:

```text
9.9 × 10^2   vs 100
161230        vs 7.182 × 10^5
713           vs 4.78 × 10^2
1.354 × 10^6 vs 4906723
20834         vs 6.5 × 10^3
```

with answer positions alternating `A, B, A, B, A`.

No prompt-template search.

### Hidden-state position

Use only:

```text
last input token immediately before answer generation
```

No token search.

Train one logistic ranking probe per layer on train, choose the best layer on validation, break exact ties toward the earliest layer, then lock the layer before test.

---

## Primary G0 object

The project is not about an aggregate accuracy gap. The project-level object is the instance-level cell:

```text
probe correct
AND
generation wrong
```

Report the 2×2 table:

| | generation correct | generation wrong |
|---|---:|---:|
| probe correct | n11 | **n10 critical** |
| probe wrong | n01 | n00 |

The main subset is fixed in advance:

```text
|log2(a/b)| < 0.1
```

because the seed paper already identifies this as the difficult numerical-comparison regime.

---

## Frozen survival gate

Proceed to causal mechanism work only if the locked Qwen3-8B `int-sci` test satisfies all of the following:

1. full-test probe accuracy `>= 0.90`;
2. hard-subset probe accuracy `>= 0.80`;
3. hard-subset `A_probe - A_generation >= 0.15`;
4. hard subset has at least `30` `probe-correct / generation-wrong` cases;
5. invalid/unparseable generations are `< 5%` of the hard subset.

### GO

`GO_CAUSAL_G1`

Only then study inference-time causal access.

### KILL / DOWNGRADE

`KILL_OR_DOWNGRADE_ACCESS_PROJECT`

Kill the access-mechanism project if balanced five-shot prompting closes the gap, if same-prompt probe accuracy collapses, or if the critical cell is too sparse.

Do **not** rescue by changing prompt, model, token position, hard threshold, nonlinear probe, or test-selected layer.

A failed G0 would not refute the seed paper; it would show that the stronger same-computation access interpretation lacks a sufficiently clean experimental object.

---

## Why this candidate is unusually feasible

This topic was promoted from `advisor_topic_search` because it passes the feasibility-first audit better than the other current NLP candidates:

```text
published phenomenon: yes
same accessible open model: yes
released data generator: yes
exact automatic labels: yes
released probe recipe: yes
released verbalization recipe: yes
paid API for G0: 0
new human annotation: 0
foundation-model training: 0
local GPU usefulness: high
```

The research risk is concentrated in one new scientific question instead of stacked uncertainty about whether the object, model regime, measurement and labels all exist.

---

## If G0 is positive: G1

A positive G0 proves only that correct ranking information is readily decodable while output is wrong. It does **not** prove that the probe direction is the model's native causal channel.

G1 should therefore test causal access with a bounded intervention design.

Preferred route:

1. keep the G0-selected layer/token rule fixed or select one operating point on validation only;
2. construct matched numerical counterfactuals where the semantic magnitudes are unchanged but notation/interface changes;
3. use activation patching / residual-space intervention to move the ranking state toward the correct counterpart;
4. compare against shuffled-label and norm-matched random-direction nulls;
5. evaluate once on locked `probe-correct / generation-wrong` test cases.

The key question is:

> **Can changing the readable ranking state causally change the model's generated choice?**

A clean null is also informative: if ranking is strongly decodable but calibrated intervention does not affect output, the readable state may be epiphenomenal or off the actual generation path.

---

## Method opening

If a specific representation-to-readout bottleneck is identified, the natural follow-up is not another benchmark. It is a method that improves **access to already represented numerical relations**, for example:

- representation-aware readout training;
- routing / access regularization;
- lightweight contrastive finetuning on notation-equivalent pairs;
- intervention-inspired auxiliary objectives that preserve internal ranking information through the generation path.

The seed paper already shows that strengthening numerical representations during training can improve behavior. Our stronger contribution would be to localize whether the remaining failure is genuinely an access/readout problem and identify the causal stage where the information is lost or ignored.

---

## Collision / novelty boundary

The seed already proves:

- numerical magnitude is linearly recoverable;
- pairwise ranking can be highly decodable;
- explicit mixed-notation verbal comparison is much worse;
- a probe-aware training objective can improve verbal behavior.

Therefore this project must **not** claim merely that representation and behavior differ or that representation quality matters.

The protected novelty claim is narrower:

> **Does a same-prompt, same-instance ranking representation remain correct when generation fails, and is that representation causally accessible to the generation decision?**

If a recent paper directly performs inference-time causal intervention on this exact numeracy dissociation, reassess before G1.

---

## Resource fit

This topic matches the project's current resource profile:

- little cash available for closed APIs;
- no budget for large new human annotation;
- strong local GPU availability.

So the expensive part, if G0 survives, is exactly the part we are well equipped to do: repeated hidden-state extraction, activation intervention and open-model mechanism analysis.

---

## Canonical preregistration and implementation

The search-stage audit is retained at:

- [`../advisor_topic_search/ROUND_04_2026-08-23.md`](../advisor_topic_search/ROUND_04_2026-08-23.md)
- [`../advisor_topic_search/ROUND_05_2026-08-23.md`](../advisor_topic_search/ROUND_05_2026-08-23.md)
- [`../advisor_topic_search/g0/NUMERACY_ACCESS_G0.md`](../advisor_topic_search/g0/NUMERACY_ACCESS_G0.md)
- [`../advisor_topic_search/g0/numeracy_data_audit.py`](../advisor_topic_search/g0/numeracy_data_audit.py)
- [`../advisor_topic_search/g0/numeracy_same_prompt_g0.py`](../advisor_topic_search/g0/numeracy_same_prompt_g0.py)

`run_g0.sh` in this directory is the registered-candidate entrypoint and delegates to that frozen implementation so there is only one scientific source of truth.
