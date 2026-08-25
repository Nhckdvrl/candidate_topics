# Shared Truthful-Clue Traps — Artifact-Only Preflight Results

## Final verdict

`STOP_SHARED_TRAP_ROUTE`

Cross-family co-reversal is greater than an agent-marginal-preserving null, but
the frozen population/support requirement for shared semantic traps fails.
This terminal result does not reopen Topic 28 and does not authorize modern
model confirmation.

## Environment and frozen artifact

- Worktree: `candidate_topics_t24`, branch `main`
- Python: `3.13.13`
- Environment: `/home/xiang/venvs/topic28`
- Packages: `numpy 2.5.2`, `pandas 3.0.5`
- Input: Topic 28 G0-cleaned `eligible_transition_features.csv`
- Input SHA-256: `9b001242576a1ca7d15411e7872a9725fef3a1cd740e8eb9b082dd3006ccbc7e`
- Source response revision: `a6d18c63e08e6cf9ad56b529ce5b10e217240e36`
- Source question revision: `3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4`
- No model inference, API call, alternate scorer, alias expansion, or data
  subset was used.

Commands:

```bash
cd advisor_topic_search/g0
/home/xiang/venvs/topic28/bin/python -m unittest -v \
  test_shared_truthful_clue_traps_preflight.py
/home/xiang/venvs/topic28/bin/python \
  shared_truthful_clue_traps_preflight.py --preflight-only --out-dir /tmp/...
/home/xiang/venvs/topic28/bin/python \
  shared_truthful_clue_traps_preflight.py \
  --out-dir artifacts/shared_truthful_clue_traps
```

The null used 1,000 repetitions, seed `20260825`, permuting the complete
reversal/wrong-answer payload within exact
`(config, category, to_clue_idx)` strata.

## Artifact receipt

| Quantity | Observed | Gate |
|---|---:|---:|
| rows | 120,353 | exact PASS |
| reversals | 8,102 | exact PASS |
| configs | 93 | exact PASS |
| frozen families | 19 | exact PASS |
| boundaries | 2,241 | exact PASS |
| questions | 782 | exact PASS |
| risk>=20 and families>=8 boundaries | 2,031 | >=2,000 PASS |

All artifact gates passed.

## Primary results

| Frozen quantity | Observed | Null | Gate |
|---|---:|---:|---:|
| cross-family overlap | 0.007538 | mean 0.004270 | ratio 1.765, PASS |
| overlap permutation p | 0.000999 | — | <=0.001 PASS |
| shared trap boundaries | 17 | mean 0.001 | >=20 **FAIL** |
| unique trap questions | 17 | — | >=20 **FAIL** |
| trap categories | 7 | — | >=5 PASS |
| trap-count permutation p | 0.000999 | — | <=0.001 PASS |
| trap observed/null ratio | 17,000 | — | >=3 PASS |
| exact-consensus events in traps | 165 | — | >=100 PASS |

The config-level overlap was `0.006927` versus null mean `0.005490`. The
family-overlap z-score was `47.85`; this establishes clustering in the released
artifact, but the preregistered mother route also required a sufficiently large
population of shared, same-competitor traps.

## Required artifact-contract correction

The first committed implementation required four families to flip at a
boundary and five exact copies of the top wrong answer, but accidentally did
not require that the exact answer itself span families. Its initial output had
20 apparent traps; audit showed three were answer-consensus clusters entirely
inside one family.

That contradicted the explicit mother criterion. Commit `b9c7e7e` therefore
added the weakest literal cross-family condition: the same exact normalized
wrong answer must be produced by at least two families. No other definition,
threshold, stratum, seed, or gate changed. The complete 1,000-repetition run was
then repeated from scratch and produced the terminal 17-trap result above.

## Concentration and specificity diagnostics

- Exact top-wrong family support among the 17 traps was: two families for 2
  traps, three for 1, four for 5, five for 3, six for 4, and eight for 2.
- The largest single-family contribution was `rag_bm25`: 47/165 exact-consensus
  events (`28.5%`) across 12 traps. It did not alone dominate the corrected
  result, although the three RAG families together contributed 92/165 events.
- Trap categories were Science 6, History 3, Geography 2, Fine Arts 2,
  Religion 2, Literature 1, and Politics 1.
- By frozen specificity quartile, trap counts were Q1=1, Q2=7, Q3=7, Q4=2.
  This diagnostic is not monotonic and cannot rescue the failed support gates.

## Representative cases and measurement caution

Clear semantic competitors include:

- gold Sparta, exact consensus Athens (`14/19` reversals; 8 families);
- gold Tolstoy, exact consensus *Anna Karenina* (`14/17`; 6 families);
- gold temperature, exact consensus entropy (`8/16`; 5 families);
- gold multiplication, exact consensus cross product (`8/15`; 5 families).

Several other exact released-score errors are answer-granularity neighbors,
including static -> static electricity, polls -> opinion poll, and soprano ->
coloratura soprano. The frozen strict alias diagnostic intentionally does not
invent post-outcome aliases or replace released `score`; these examples are
reported as a measurement limitation and further weaken a broad “shared wrong
competitor” interpretation.

## Interpretation and stop rule

Released reversals are not fully independent across model families, but only
17 boundaries satisfy the complete frozen shared-trap definition. Because the
20-boundary and 20-question gates fail, the route stops. Do not run untouched
modern models, natural clue substitution, aggregation-law fitting, hidden
states, or another scorer/config/panel search.
