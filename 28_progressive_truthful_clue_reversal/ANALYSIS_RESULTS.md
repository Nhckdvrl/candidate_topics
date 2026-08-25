# Topic 28 — Frozen Descriptive Structure Results

## Scientific status

**G0 phenomenon established; descriptive structure analyzed; controlled order
intervention is next.**

This phase used the frozen population and correctness definition from G0. It
did not run model inference, use an LLM judge, train a classifier, select
configs, or make a mechanism claim. The hard reconciliation checks reproduced
exactly `120,353` eligible adjacent correct-state transitions and `8,102`
official-score reversals (`6.7319%`).

## Environment and receipt

- Repository worktree: `candidate_topics_t24`, branch `main`
- Python environment: `/home/xiang/venvs/topic28`
- Responses revision: `a6d18c63e08e6cf9ad56b529ce5b10e217240e36`
- Questions revision: `3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4`
- Response configs loaded: `128/128`
- Analysis seed: `20260825`
- Cluster bootstrap: `2,000` whole-`qid` resamples

Commands:

```bash
cd 28_progressive_truthful_clue_reversal
PATH=/home/xiang/venvs/topic28/bin:$PATH python -m unittest discover -s tests -v
PATH=/home/xiang/venvs/topic28/bin:$PATH HF_HUB_DISABLE_PROGRESS_BARS=1 bash run_analysis1.sh
```

The run completed `15/15` unit tests. Generated analysis tables remain ignored
under `artifacts/analysis1/`; the exact definitions are checked in in
`ANALYSIS_PLAN.md`.

## Population reconciliation and sanity

| Check | Result |
|---|---:|
| raw response rows | 362,120 |
| clean joined non-human rows | 281,620 |
| ambiguous duplicate cells dropped under frozen G0 policy | 372 |
| question join coverage | 1.000000 |
| eligible adjacent correct-state transitions | 120,353 |
| reversal events | 8,102 |
| human rows in eligible/event tables | 0 / 0 |
| non-adjacent rows in eligible/event tables | 0 / 0 |
| empty exact added clues in eligible/event tables | 0 / 0 |

All 8,102 rows in the event dynamics table are the same official-score `1 ->
0` events established in G0. The G0 duplicate, strict-alias (`3,871`
supported events), and deterministic 50/50 added-official-clue audits continue
to apply; this phase never rescored or filtered those events.

## 1. Competitor introduction

The post-reversal prediction supplies an observed wrong competitor. Frozen
surface diagnostics found:

| Diagnostic among 8,102 reversals | Events | Rate |
|---|---:|---:|
| exact normalized wrong prediction occurs in the new clue | 649 | 8.01% |
| exact wrong prediction occurs newly in the new clue | 626 | 7.73% |
| at least one wrong-answer content token is newly introduced | 1,387 | 17.12% |
| wrong content coverage exceeds gold content coverage | 1,413 | 17.44% |

Mean wrong-prediction content coverage in the new clue was `0.1428`, versus
`0.05495` for the best gold alias. These deterministic lexical diagnostics
establish direct competitor introduction for a substantial high-precision
minority, not for the majority. A miss is uninformative about semantic
competition because no semantic judge was introduced after seeing outcomes.

Representative exact-surface events include:

```text
mass -> inertia
added clue: Its rotational analogue is rotational inertia.

affirmative action -> reverse discrimination
added clue: Bakke ... dealt with ... whether it should be viewed as reverse discrimination.

Shiva -> Kartikeya
added clue: His sons are Kartikeya and Ganesha.
```

These examples show the intended object: the new statement remains a truthful
clue for the gold answer while also supplying a locally salient competitor.

## 2. Arrival stage and recovery

Reversal density falls strongly as evidence accumulates, but the object does
not disappear late:

| Added clue index | Eligible transitions | Reversals | Rate |
|---:|---:|---:|---:|
| 2 | 33,613 | 3,516 | 10.46% |
| 3 | 42,979 | 2,622 | 6.10% |
| 4 | 34,337 | 1,570 | 4.57% |
| 5 | 8,590 | 351 | 4.09% |
| 6–7 | 834 | 43 | 5.16% |

The frozen relative-stage bins gave `16.86%` early (`71/421`), `9.93%`
middle (`4,036/40,654`), and `5.04%` late (`3,995/79,278`). The early bin is
small because an eligible boundary requires the model already to be correct.
The `3,995` late-stage events are nevertheless substantial evidence that this
is not only first-clue uncertainty.

Event-level recovery results:

| Outcome after a reversal | Events | Rate |
|---|---:|---:|
| immediate next-clue recovery | 3,659 | 45.16% |
| any observed later recovery | 4,611 | 56.91% |
| final observed state correct | 4,404 | 54.36% |
| never observed to recover | 3,491 | 43.09% |

Median recovery lag among recovered events was one clue. This is a mixed
object: many reversals are transient path-dependent instabilities, while a
large minority persist through the released trajectory.

## 3. Trigger structure

The cleanest descriptive pattern is clue specificity. Frozen corpus-IDF
quartiles rose monotonically from `4.35%` to `6.48%`, `7.74%`, and `8.35%`.
This supports specificity capture as a leading explanation to test causally.

Other preregistered contrasts were more selective:

| Feature (`true - false`) | Rate difference | qid-cluster bootstrap 95% CI |
|---|---:|---:|
| introduced capitalized-name surface | -2.50 pp | [-3.46, -1.60] pp |
| four-digit year | +0.32 pp | [-0.87, +1.63] pp |
| any number | -3.52 pp | [-4.06, -2.99] pp |
| quote/title punctuation | +0.65 pp | [-0.16, +1.47] pp |
| parenthetical | +1.33 pp | [+0.45, +2.20] pp |
| exact gold alias in new clue | -2.26 pp | [-3.84, -0.23] pp |

Long-clue quartiles were non-monotonic (`6.75%`, `6.40%`, `6.20%`, `7.72%`).
Introduced-name counts were inversely associated with reversal (`8.81%` for
zero, `6.86%` for one, `6.11%` for two or more). Thus a generic
"proper nouns/numbers override the answer" account is not supported as
stated. Parentheticals are a narrower candidate; years and quotes are not
resolved by this descriptive run.

## Concentration

The structure analysis preserved the G0 spread:

- all 93 non-human configs with eligible transitions had reversals;
- top config: `311/8,102` events (`3.84%`); top five: `14.93%`;
- config rate quantiles: min `0.00093`, Q1 `0.03402`, median `0.06405`, Q3
  `0.13128`, max `0.47676`;
- 760 questions; top question `40/8,102` (`0.49%`), top five `2.41%`;
- Science supplied `31.15%` of events, matching a high but non-exclusive
  category concentration; History, Literature, Fine Arts, and Geography
  together supplied another `48.42%`.

No single config, question, malformed ID, duplicate policy choice, human
backfill, or canonicalization rule explains the result.

## Interpretation and next gate

The descriptive evidence prioritizes three explanations for a controlled
experiment:

1. **specificity capture** is the strongest population-level correlate;
2. **direct lexical competitor introduction** explains a clean minority;
3. **path-dependent instability** is supported by frequent one-clue recovery,
   alongside a distinct persistent subset.

The broad anchor-override account is weakened because names and numbers are
negatively associated with reversals, while only parentheticals show a clear
positive contrast. These are associations, not causal effects.

The next scientific step is a preregistered order intervention holding the
truthful clue multiset, gold answer, prompt, and model fixed. This result does
not authorize activation patching or a claim that the gold representation was
erased.

## Engineering note

The first full analysis attempt produced no result artifacts and stopped on a
`KeyError` because category/subcategory live inside the released `metadata`
dict rather than as top-level G0-clean columns. The outcome-blind repair
expanded those two fields and added a regression test. The same repair commit
vectorized the already-frozen whole-`qid` bootstrap without changing its seed,
resampling unit, repetitions, or estimator. The complete 128-config analysis
was rerun from scratch and passed all reconciliation assertions.
