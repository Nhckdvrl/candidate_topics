# Topic 04 — Confidence and Error Correction

**Status: CANDIDATE / VALIDATION NOT YET RUN**

> If two learners are equally far from the correct answer, does being strongly committed to one wrong answer make corrective learning easier or harder?

## Why this is a natural question

Human memory research has repeatedly reported the **hypercorrection effect**: errors endorsed with high confidence are often corrected *better* after feedback than low-confidence errors. Competing explanations remain important:

1. **surprise / attention** — a confident error makes corrective feedback unusually salient;
2. **prior / partial knowledge** — confidence may be a proxy for already having more of the relevant knowledge.

Modern language models give us a controlled learning system where we can observe the entire correction curve after each identical exposure.

At the same time, recent LLM work reports the opposite-looking pattern: strong incorrect priors can resist prompt correction and can be hard to override during SFT. The scientific question is therefore not "do LLMs have a human bias?" but:

> **How does prior conviction affect the dynamics and durability of correction, once correct-answer accessibility is held fixed?**

## Seed literature

### Human learning

- Butterfield & Metcalfe (2006), *The correction of errors committed with high confidence*.
- Fazio & Marsh (2009), *Surprising feedback improves later memory*.
- Metcalfe & Finn (2011), *People's Hypercorrection of High Confidence Errors: Did They Know it All Along?*
- Sitzman, Rhodes & Tauber (2014), *Prior knowledge is more predictive of error correction than subjective confidence*.
- Butler, Fazio & Marsh (2011), *The hypercorrection effect persists over a week, but high-confidence errors return*.

### Recent AI evidence

- Xue et al. (ACL 2026), *Why Supervised Fine-Tuning Fails to Learn*: high-confidence conflicts between base-model beliefs and SFT supervision are treated as a distinct source of incomplete learning and are described as resistant / slow to correct.
- Casanova et al. (ICML 2026), *On the Limits of LLM Adaptability*: high-confidence zero-shot annotation errors are especially resistant to **prompt-level** correction.
- Kumaran et al. (Nature Machine Intelligence 2026), *Competing Biases underlie Overconfidence and Underconfidence in LLMs*: initial confidence predicts reduced change-of-mind flexibility.

These papers establish that **prior conviction matters**, but they do not isolate the weight-level learning dynamics asked here.

## The crucial confound

A naive experiment would define confidence as `max_wrong_probability` and compare high vs low confidence errors.

That is not acceptable.

If a wrong answer has probability 0.9, the correct answer will usually have lower initial probability than in a diffuse 0.4-error case. Then a slower correction could simply mean:

> "the correct target started farther away."

The primary experiment therefore separates:

### Correct-answer accessibility

`a(x) = p(y* | x)`

### Wrong-belief concentration

For the normalized distribution over *wrong* options,

`q_j(x) = p(y_j|x) / (1-p(y*|x))`, for wrong `y_j`,

define

`c(x) = max_j q_j(x)`

and, as a robustness measure,

`c_H(x) = 1 - H(q)/log(K-1)`.

`c` asks: **conditional on being wrong, is probability mass concentrated on one specific wrong hypothesis or spread over several?**

The primary high- vs low-conviction comparison is matched on `a(x)`.

## Measurement safeguards

1. **At least 4 options.** Binary questions cannot separate correct accessibility from wrong concentration.
2. **Balanced option permutations.** Each original answer appears in every label position; probabilities are mapped back to option identity and averaged.
3. **Semantic-conviction gate.** The same wrong *content* must remain top-ranked in at least 3/4 balanced permutations; otherwise the item is likely label/position sensitive and is excluded from the primary set.
4. **Match before training.** Matching uses only the frozen base model.
5. **Equal exposures.** Every selected item receives exactly the same number of corrective SFT exposures.
6. **Content targets, not fixed answer letters.** Training permutations vary so the model must learn the answer content rather than memorize `A/B/C/D`.

## Validation pipeline

```text
G-1  Measurement feasibility
     ↓
     Are there enough wrong items with stable semantic conviction and
     matched high/low wrong-concentration at the same p(correct)?

G0   Correction dynamics
     ↓
     Equal corrective exposures; measure item-level p(correct)
     after every exposure cycle.

G1   Durability / return of error
     ↓
     After correction, train on unrelated filler and measure whether
     the original wrong answer returns.

G2   Replication
     ↓
     Another model family + another knowledge domain.
```

### G-1 — measurement feasibility

Initial model: `Qwen/Qwen2.5-1.5B-Instruct` (cheap screen).

Candidate pools should come from several 4-option knowledge/reasoning datasets, e.g. MMLU, ARC-Challenge, OpenBookQA, MedMCQA. The pilot should aim for at least **300 matched pairs** after all filters.

**Stop / redesign the measurement** if:

- fewer than 200 usable pairs survive;
- high/low groups cannot be matched to absolute `p(correct)` difference <= 0.03;
- top-wrong identity is dominated by option-position effects.

### G0 — correction dynamics

Train **one model on both groups together**, shuffled every epoch.

Primary unit: item.

For each exposure cycle `e`, save a checkpoint and compute permutation-averaged `p_e(y*)` and `p_e(y_old-wrong)`.

Primary endpoints:

1. **AUC-correction**: area under the `p(correct)` vs exposure curve.
2. **Time-to-correction**: first exposure where `p(correct) >= 0.5` and stays above 0.5 for the next evaluation.
3. **Old-error suppression**: decline of the original top-wrong probability.

Primary statistical model should include base `p(correct)` continuously, not only matched-group labels.

Do **not** search learning rates or confidence thresholds for a favorable result. Pick one ordinary SFT recipe, verify that learning occurs at all, then freeze it.

### G1 — return of error

Human hypercorrection work reports a subtle dissociation: high-confidence errors may correct well immediately yet be especially likely to reappear when the correction is forgotten.

After G0:

1. keep only items successfully corrected;
2. fine-tune on a fixed unrelated filler corpus for a preregistered number of updates;
3. re-evaluate the original questions.

Measure `P(old wrong returns | was corrected)`.

This separates **fast acquisition** from **durable replacement**.

## Four informative outcome classes

| Outcome | Interpretation |
|---|---|
| Higher conviction → faster correction after accessibility matching | hypercorrection-like learning; concentrated error may create a more targeted / surprising update |
| Higher conviction → slower correction | entrenchment: strong incorrect priors resist weight-level revision |
| Higher conviction → faster immediate correction but more relapse | fast update does not imply durable replacement; close structural parallel to delayed human hypercorrection results |
| Effect vanishes after matching `p(correct)` / prior accessibility | apparent confidence effect is largely a partial-knowledge/accessibility effect |

The fourth result is **not a null with no story**; it directly adjudicates the main human-mechanism dispute.

## What would kill the topic

- no stable semantic-conviction measurement after position controls;
- correction dynamics are almost entirely deterministic from base `p(correct)` and wrong concentration adds no reproducible information across a locked confirmation split/model;
- an exact 2026+ paper is found that already performs accessibility-matched weight-level correction curves and relapse analysis.

## Novelty boundary

Do **not** claim:

- "LLM high-confidence errors are hard to correct" — already adjacent to ACL/ICML 2026 work;
- "LLMs exhibit confidence" — extensively studied;
- "we reproduce a human bias in AI."

The proposed contribution is narrower:

> **separate target accessibility from commitment to a specific wrong hypothesis, then measure how that commitment changes the speed and durability of corrective weight updates.**

## Initial code

```bash
python code/score_mcq.py --model Qwen/Qwen2.5-1.5B-Instruct --input data/candidates.jsonl --output results/base_scores.jsonl
python code/build_matched_pairs.py --input results/base_scores.jsonl --output results/matched.jsonl
python code/analyze_correction.py --input results/correction_scores.jsonl
```

See `VALIDATION.md` for the locked pilot protocol and input schemas.
