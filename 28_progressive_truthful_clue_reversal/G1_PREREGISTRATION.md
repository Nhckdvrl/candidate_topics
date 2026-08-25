# Topic 28 — Frozen G1 Adjacent-Order Intervention

**Status: FROZEN BEFORE MODEL OUTPUTS.**

## Question

Does a truthful clue's ability to destabilize a correct answer depend on when
it arrives in the evidence history, or is the effect explained by the clue or
clue set independent of order?

G1 changes only the order of two adjacent official clues:

```text
original: prefix, C_t, C_(t+1)
swapped:  prefix, C_(t+1), C_t
```

The question, gold answer, clue multiset, clue count, prompt, model, decoding,
and scorer are identical. G1 does not use hidden states.

## Outcome-blind panel

The panel is built only from frozen question artifact revision
`3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4`. Released response rows, G0
scores, G0 reversal labels, model predictions, and `reversal_events.csv` are
not read during panel construction.

A boundary is included iff:

1. all official clue states `1..t+1` exist and are contract-valid;
2. `t >= 2`, so at least one clue precedes the swapped pair;
3. question category metadata and strict `clean_answers` aliases are present;
4. both adjacent atomic clues are non-empty;
5. `C_(t+1)` specificity is at or above the median specificity over all 3,042
   released official atomic clues.

Specificity is the already-frozen corpus-IDF mean over non-stopword clue
tokens. The outcome-blind preflight fixes the expected full panel at 498
boundaries from 415 questions. Multiple qualifying boundaries from one
question remain; uncertainty resamples whole question clusters.

No boundary is selected because any released or newly run model was correct or
reversed there.

## Frozen model and inference

```text
model: Qwen/Qwen2.5-7B-Instruct
revision: a09a35458c702b33eeacc393d103063234e8bc28
precision: bfloat16
prompt: fixed chat template below
decoding: greedy, do_sample=false, max_new_tokens=24
correctness: exact frozen SQuAD-normalized match to released clean_answers
```

System message:

```text
You answer Quiz Bowl questions. Return only the short answer, with no explanation.
```

User message:

```text
Identify the answer described by these clues.

Clues:
1. <clue>
2. <clue>
...

Answer:
```

The model's complete decoded continuation is scored. No post-result answer
extraction, fuzzy matching, alias extension, or prompt repair is allowed.

For every panel boundary exactly four states are run:

```text
O1 = prefix + C_t
O2 = prefix + C_t + C_(t+1)
S1 = prefix + C_(t+1)
S2 = prefix + C_(t+1) + C_t
```

## Primary paired estimand

```text
R_original = 1[O1 correct and O2 wrong]
R_swap     = 1[S1 correct and S2 wrong]

delta_order = mean(R_original - R_swap)
```

This is estimated over the full outcome-blind panel, never over a panel
selected for original-order reversal. A 95% interval uses 2,000 whole-question
cluster bootstrap resamples with seed `20260825`.

Because `C_t` and `C_(t+1)` can differ in standalone difficulty, a path
dependence verdict additionally requires the clean same-multiset final-state
contrast:

```text
delta_final_error = P(O2 wrong) - P(S2 wrong)
```

This contrast compares prompts containing exactly the same clues and differing
only in their order. It uses the same clustered bootstrap.

## Frozen diagnostics for competing explanations

These decompose the primary result but do not replace it:

- first-state correctness `P(O1 correct)` and `P(S1 correct)`;
- common-belief subset where both O1 and S1 are correct;
- original-only final harm: `O2 wrong, S2 correct`;
- swap-only final harm: `O2 correct, S2 wrong`;
- order-independent conflict: `O2 wrong, S2 wrong`;
- exact four-state patterns `(O1,O2,S1,S2)`;
- results by frozen trigger-specificity half (Q3 versus Q4), category, and `t`;
- prompt compliance, truncation, and missing-output audits.

No random shuffle, full reversal, alternate model, alternate prompt, semantic
judge, scorer change, or mechanism experiment is part of G1.

## Frozen gates and verdicts

Artifact/measurement gates:

```text
full panel boundaries                         == 498
unique questions                              == 415
valid four-output boundaries                  >= 0.98
O1-correct support                            >= 100
S1-correct support                            >= 100
common-belief support (O1=S1=correct)         >= 75
```

Scientific path-dependence gates:

```text
original reversal events                      >= 20
delta_order                                   >= 0.02
cluster-bootstrap lower 95% bound delta_order > 0
delta_final_error                             >= 0.01
cluster-bootstrap lower bound final error     > 0
```

Verdict logic:

```text
STOP_G1_MEASUREMENT
    any artifact/measurement gate fails

GO_PATH_DEPENDENT_REVERSAL
    all artifact and all scientific gates pass

GO_ORDER_EFFECT_ONLY
    artifact gates pass and delta_final_error >= 0.01 with lower bound > 0,
    but the full reversal gate set does not pass

STOP_ORDER_DEPENDENCE
    artifact gates pass and neither positive verdict applies
```

These thresholds, panel rules, model, scorer, and verdict names may not be
changed after observing G1 outputs. A stop does not authorize another model,
prompt, threshold, panel, or permutation search.
