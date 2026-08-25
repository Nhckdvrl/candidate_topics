# Topic 28 — Frozen Descriptive Analysis Plan

**Status: FROZEN BEFORE ANALYSIS OUTPUTS.**

## Scope

This phase characterizes the established G0 object. It does not run model
inference, train a classifier, use an LLM judge, change correctness, or make a
mechanistic claim.

Population:

```text
all released non-human response configs
same G0 cleaning and duplicate policy
same trajectory key: (config, agent_type, qid)
same eligible boundary: adjacent state with released score_before = 1
reversal: released score_after = 0
stable control: released score_after = 1
```

All 1->0 events and all eligible 1->1 controls are retained. No config,
question, category, or model is selected by outcome.

## Question 1 — Competitor introduction

The post-reversal wrong prediction is treated as an observed competitor. The
following high-precision surface diagnostics are frozen:

1. exact normalized wrong-answer token sequence appears in the new clue;
2. it appears in the new clue but not the cumulative text before it;
3. at least one non-stopword wrong-answer token appears newly in the clue;
4. wrong-answer content-token coverage in the new clue;
5. maximum gold-alias content-token coverage in the new clue;
6. wrong-minus-gold coverage advantage.

These are lexical lower bounds on competitor introduction. Failure to detect a
surface relation is not evidence that no semantic competitor was activated.

## Question 2 — Evidence order and recovery dynamics

For every reversal event, report:

- from/to clue index and relative arrival stage;
- whether the next observed official clue immediately restores correctness;
- whether any later observed state restores correctness;
- lag in clue indices to the first observed recovery;
- whether the final observed state is correct;
- reversal rates by clue index, relative stage, config, and category.

Immediate recovery requires the exact next clue index. Eventual recovery follows
the G0 diagnostic convention and allows a later observed correct state even if
the released trajectory has a gap.

## Question 3 — Reversal-trigger structure

Structural features are computed on the exact newly added official clue for all
eligible reversal and stable-control transitions:

- clue word count;
- corpus-IDF specificity, computed only from released official atomic clues;
- capitalized-name surface spans introduced relative to the previous cumulative
  text (a deterministic proper-noun proxy);
- four-digit year mention;
- any numeric mention;
- quotation/title punctuation;
- parenthetical text;
- exact normalized gold-alias mention.

Report reversal rates by binary feature and by frozen outcome-blind bins:

- word-count quartiles over all eligible transitions;
- specificity quartiles over all eligible transitions;
- introduced-name count: 0, 1, 2+;
- relative arrival stage: early, middle, late by `to_clue / total_clues` thirds.

For the primary binary contrasts, report a 95% cluster bootstrap interval for
the reversal-rate difference, resampling whole original `qid` clusters with
seed `20260825` and 2,000 repetitions.

## Four competing explanations carried forward

1. **Direct competitor introduction:** the added clue names or lexically evokes
   the answer the model switches to.
2. **Specificity capture:** rare named details disproportionately redirect the
   current hypothesis, even when they are truthful for the gold answer.
3. **Anchor override:** years, numbers, titles, or parentheticals create a strong
   local retrieval cue that overrides accumulated evidence.
4. **Path-dependent instability:** reversals depend on when evidence arrives and
   often recover when later clues reweight the hypothesis.

This phase ranks and sharpens these explanations descriptively. It does not
declare any one causal and does not authorize hidden-state work.

## Outputs

```text
artifacts/analysis1/
  analysis_receipt.json
  analysis_summary.json
  eligible_transition_features.csv
  reversal_event_dynamics.csv
  trigger_feature_summary.csv
  rates_by_to_clue.csv
  rates_by_relative_stage.csv
  rates_by_config.csv
  rates_by_category.csv
```

Generated artifacts remain uncommitted. The checked-in result document records
support, exact revisions, fixed definitions, and limitations.

