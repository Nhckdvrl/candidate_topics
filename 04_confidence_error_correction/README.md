# Topic 04 — Confidence and Error Correction

**Status: READY FOR G-1 / G0 IMPLEMENTED**

> **If two learners are equally far from the correct answer, does being strongly committed to one wrong answer make corrective learning easier or harder?**

This topic studies a natural learning question. The LLM is a controlled learner, not the scientific object of interest.

## Why this question exists independently of LLMs

Human memory research reports the **hypercorrection effect**: an error held with high subjective confidence can be corrected *better* after feedback than a low-confidence error. Two broad explanations have competed for years:

1. **surprise / attention** — violating a strong expectation makes corrective feedback unusually salient;
2. **prior / partial knowledge** — confidence may merely proxy how much relevant knowledge is already available.

The second explanation matters because "high confidence" and "distance from the correct answer" are usually entangled.

Recent LLM work gives the opposite-looking clue. Strong model-internal priors can be sticky under prompt correction, and SFT can fail on samples that conflict with pretrained knowledge. This creates a concrete question:

> **After holding target accessibility fixed, does commitment to a specific wrong hypothesis change the speed or durability of learning the correction?**

## Seed literature

### Human learning

- Butterfield & Metcalfe (2006), **The correction of errors committed with high confidence**.
- Fazio & Marsh (2009), **Surprising feedback improves later memory**.
- Metcalfe & Finn (2011), **People's Hypercorrection of High Confidence Errors: Did They Know it All Along?**
- Sitzman, Rhodes & Tauber (2014), **Prior knowledge is more predictive of error correction than subjective confidence**.
- Butler, Fazio & Marsh (2011), **The hypercorrection effect persists over a week, but high-confidence errors return**.

### AI evidence that motivates, but does not answer, the question

- Xue et al. (ACL 2026), **Why Supervised Fine-Tuning Fails to Learn**: incomplete learning includes conflicts between SFT supervision and pretrained knowledge.
- Casanova et al. (2026), **On the Limits of LLM Adaptability**: high-confidence zero-shot annotation errors are especially resistant to **prompt-level** correction.
- Kumaran et al. (Nature Machine Intelligence 2026), **Competing Biases underlie Overconfidence and Underconfidence in LLMs**: initial confidence predicts reduced flexibility in revising answers.
- Ren & Sutherland, **Learning Dynamics of LLM Finetuning**: SFT probability changes depend on the current predictive distribution and cross-example interactions.

None of these isolates **weight-level corrective learning after matching the initial accessibility of the correct target**.

## The key operational distinction

Do **not** use raw `max_wrong_probability` as the independent variable.

For a multiple-choice item with correct answer \(y^*\), define:

### Target accessibility

\[
a(x)=p(y^*\mid x)
\]

### Wrong-belief concentration

Normalize only over wrong options:

\[
q_j(x)=\frac{p(y_j\mid x)}{1-p(y^*\mid x)}, \qquad y_j\neq y^*
\]

Primary commitment measure:

\[
c_{\max}(x)=\max_j q_j(x)
\]

Robustness measure:

\[
c_H(x)=1-\frac{H(q)}{\log(K-1)}
\]

Intuition:

- same \(a(x)\): both learners are equally far from the correct target;
- different \(c(x)\): one learner strongly commits to one misconception, the other spreads belief over many wrong alternatives.

The main experiment asks whether \(c(x)\) predicts **correction dynamics after conditioning/matching on \(a(x)\)**.

## Why MMLU-Pro is the primary stimulus pool

The primary G-1 pool is the exactly-10-option subset of **MMLU-Pro**.

Reasons:

1. with 10 options, wrong-answer concentration has much more dynamic range than in binary or 4-choice tasks;
2. the dataset is large enough to construct hundreds of tightly matched wrong-item pairs;
3. the published dataset analysis reports lower prompt-format sensitivity than original 4-choice MMLU;
4. each item has a single correct answer and rich domain labels.

The test split is used only as a **pool of supervised learning stimuli**, not to claim benchmark generalization.

External replication can later use original MMLU / ARC-Challenge / OpenBookQA / MedMCQA.

## Validation overview

```text
G-1a  Can we measure stable semantic wrong commitment?
      Score all exact-10-option MMLU-Pro items under balanced option rotations.

G-1b  Can we separate wrong commitment from target accessibility?
      Construct high/low commitment pairs tightly matched on p(correct),
      category, question length, and correct-answer length.

G0-D  Discovery correction experiment
      One identical corrective exposure per item per cycle.
      Track the whole probability trajectory for 10 cycles.

LOCK  Freeze primary direction, effect definition, LR, and analysis.

G0-C  Locked confirmation
      Repeat from the same base model on held-out matched pairs.

G0-R  Robustness
      Independent training seeds; then a second model or second dataset.

G1    Durability / return of the old error
      Only if G0 establishes a reproducible correction-dynamics effect.
```

The exact stop rules and alternative scientifically interesting outcomes are in `VALIDATION.md`.

## Fast path for a many-GPU server

The code is designed so that **scoring shards, discovery/confirmation runs, and seeds can be assigned to separate GPUs/nodes**. Cross-node communication is unnecessary.

Recommended first screen:

```text
base model       Qwen/Qwen2.5-1.5B-Instruct
primary pool     MMLU-Pro exact-10-option test items
G-1 scoring      8 shards in parallel
matched pairs    target >= 600, minimum useful >= 300
G0 cycles        10 (one semantic exposure/item/cycle)
G0 discovery     70% of pairs
G0 confirmation  30% of pairs, not inspected until recipe is frozen
training         full-parameter SFT, bf16
seeds            3 if resources allow; run different seeds on independent GPUs/nodes
```

A 1.5B full-parameter run fits easily on a single large-memory accelerator, so independent single-GPU jobs are preferable to slow cross-node distributed training.

## Four primary scientific outcomes

| Pattern | Interpretation |
|---|---|
| higher wrong commitment → faster correction | **hypercorrection-like** plasticity after accessibility matching |
| higher wrong commitment → slower correction | **entrenchment**: concentrated incorrect priors resist weight revision |
| immediate advantage reverses later / old error returns more often | **fast uptake ≠ durable replacement** |
| after matching accessibility, wrong commitment has reproducibly negligible effect | **accessibility dominance**: the apparent confidence effect is mostly explained by how reachable the target already was |

The last outcome is only scientifically useful if an **equivalence-style null** replicates on locked confirmation / another setting. A single nonsignificant coefficient is not a result.

## Predeclared secondary patterns worth checking

These are allowed because they are specified **before running G0**, not invented after a weak primary result.

1. **Early-vs-late reversal**  
   Compare cycle-1/2 gains with cycle-8/10 gains. A sign reversal suggests different mechanisms for initial uptake and consolidation.

2. **Accessibility × commitment interaction**  
   Commitment may matter only when the correct answer is partially accessible. Test one continuous interaction; do not search arbitrary bins.

3. **Old-error suppression vs correct-target growth**  
   Learning the correct answer and suppressing the original misconception may be separable dynamics.

4. **Domain heterogeneity**  
   Only after the pooled primary test. A domain effect is interesting if it replicates across seeds and has a natural explanation; do not cherry-pick the best subject.

## What kills the topic

Kill or radically reinterpret if any of the following holds:

1. G-1 cannot produce a stable commitment measure after option-position controls.
2. Fewer than ~300 matched pairs survive from the primary pool and no larger clean pool solves this without changing the measurement.
3. Discovery suggests an effect but locked confirmation does not preserve the direction.
4. The apparent effect disappears under the predeclared accessibility adjustment and an equivalence-style null replicates only weakly / inconsistently.
5. The sign depends arbitrarily on prompt wording, label position, or a narrow learning-rate choice.
6. A direct contemporaneous paper is found that already performs accessibility-matched **weight-level** correction curves and durability analysis.

## Novelty boundary

Do not claim:

- "LLMs can be confident";
- "high-confidence LLM errors are sticky";
- "SFT can overwrite knowledge";
- "LLMs reproduce human hypercorrection."

The intended contribution is narrower and more scientific:

> **Separate access to the correct target from commitment to a particular wrong hypothesis, then ask how each variable shapes the time course and durability of corrective learning.**

## Repository layout

```text
04_confidence_error_correction/
├── README.md
├── VALIDATION.md
├── SERVER_RUNBOOK.md
├── requirements.txt
└── code/
    ├── prepare_candidates.py
    ├── mcq_utils.py
    ├── score_mcq.py
    ├── merge_jsonl.py
    ├── audit_prompt_robustness.py
    ├── build_matched_pairs.py
    ├── build_sft_data.py
    ├── train_correction.py
    ├── evaluate_checkpoints.py
    └── analyze_correction.py
```

`SERVER_RUNBOOK.md` contains the exact end-to-end commands.
