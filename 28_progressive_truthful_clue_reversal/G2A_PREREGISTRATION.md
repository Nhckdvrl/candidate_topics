# Topic 28 — Frozen G2a Destructive-Conjunction Screen

**Status: FROZEN BEFORE ANY `C-alone` MODEL OUTPUT.**

## Scientific question

Can two naturally written, individually sufficient pieces of truthful evidence
for the same answer become harmful when composed?

For every frozen boundary:

```text
P     = the original cumulative prefix through clue t
C     = the next official truthful clue, presented alone
P + C = the original cumulative state through clue t+1
```

The primary cell is:

```text
P correct AND C correct AND (P+C) wrong
```

This is called **destructive conjunction**. It is stronger than an ordinary
correct-to-wrong transition because both constituent evidence inputs must
independently elicit the gold answer.

G2a does not test clue order, natural clue substitution, an aggregation-law
model, mitigation, or hidden states.

## Frozen outcome-blind panel and reused outputs

G2a uses the complete G1 panel exactly as frozen before G1 outcomes:

```text
498 boundaries
415 original questions
t >= 2
official trigger-clue specificity >= the all-clue median
```

Panel construction used only question metadata and clue text. It never used G0
or G1 correctness/reversal labels. No row is selected based on P, C, or P+C
outcomes.

G2a reuses the already released G1 outputs for P (`O1`) and P+C (`O2`) and
runs only the 498 missing C-alone states. Frozen G1 artifact hashes are:

```text
panel.csv
427deda33f45bb2a6c2d2caa1e64cead2b920b579412d0f7dbae1fe92f7b6f92

state_outputs.csv
ddf8eeb10e6dd4b69d1e1c09b2998e8bd6cb1a67cb1bf83f6ee2c348e4e4d11a

paired_results.csv
f3a05d65b07dc68977161fe77eb4593d9bdaf612bee4e575787a2e249750bc97
```

The full run aborts if any hash, row count, boundary key, model receipt, or
validity contract differs.

## Frozen inference and scorer

The C-alone run uses the exact G1 contract:

```text
model: Qwen/Qwen2.5-7B-Instruct
revision: a09a35458c702b33eeacc393d103063234e8bc28
prompt: same system/user template; one numbered clue
precision: bfloat16
decoding: greedy, do_sample=false, max_new_tokens=24
correctness: exact frozen SQuAD-normalized clean_answers alias match
```

The complete decoded continuation is scored. No answer extraction, fuzzy
matching, LLM judge, post-result alias addition, or scorer repair is allowed.

## Primary statistic

Let:

```text
jointly_sufficient = 1[P correct AND C correct]
destructive_exact  = 1[P correct AND C correct AND (P+C) wrong]
```

The primary statistic is:

```text
R_destructive = sum(destructive_exact) / sum(jointly_sufficient)
```

A 95% interval resamples whole original-question clusters 2,000 times with
seed `20260825`.

## Frozen high-precision wrong-answer support

G1 exposed low recall in the strict alias list: outputs such as `Mozart` for
`Wolfgang Amadeus Mozart` can be marked wrong. G1 remains terminal and is not
rescored. To prevent the same known undercoverage from manufacturing a positive
G2a, a destructive event receives `clear_wrong` support only if its P+C
prediction:

1. is not an exact normalized gold alias;
2. contains no normalized gold-alias phrase and is not contained by one;
3. shares no non-stopword content token with any gold alias;
4. has no prediction/alias content-token pair that becomes equal after frozen
   singular-suffix stripping or has character similarity at least 0.85.

This conservative diagnostic can reject genuine wrong answers that share gold
tokens. That is acceptable: it is a high-precision lower bound required to
support a positive claim. It never changes primary correctness labels.

## Frozen diagnostics

Report without changing the primary statistic:

- all eight `(P, C, P+C)` correctness cells;
- P, C, and P+C marginal correctness;
- exact and clear destructive counts/rates;
- unique questions with exact and clear destructive conjunction;
- counts by Q3/Q4 trigger specificity, category, and t;
- whether the C prediction equals the P prediction or the P+C prediction;
- all destructive-event predictions for manual scientific audit.

## Frozen gates and verdicts

Artifact/measurement gates:

```text
G1 artifact hashes                            exact match
panel boundaries                              == 498
unique questions                              == 415
valid C-alone outputs                         >= 0.98
jointly sufficient P-and-C support            >= 100
```

Scientific gates:

```text
exact destructive events                      >= 10
exact destructive rate                        >= 0.03
cluster-bootstrap lower 95% bound              > 0.01
clear-wrong destructive events                 >= 5
unique clear-wrong questions                   >= 5
clear-wrong destructive rate                   >= 0.01
```

Verdicts:

```text
STOP_G2A_MEASUREMENT
    any artifact/measurement gate fails

GO_DESTRUCTIVE_CONJUNCTION_OBJECT
    all artifact and scientific gates pass

STOP_DESTRUCTIVE_CONJUNCTION
    artifact gates pass but any scientific gate fails
```

A debug subset is always labeled `DEBUG_NO_VERDICT`. A stop does not authorize
another model, prompt, panel, scorer, threshold, or outcome-selected subset.

## Literature boundary

The claim is deliberately narrower than adjacent work:

- BayesBench studies Bayesian belief trajectories and downstream prediction;
- *Lying with Truths* studies adversarial truthful-fragment montage;
- CUE-R studies intervention-based RAG evidence utility and positive
  non-additive multi-support effects;
- G2a asks whether two naturally convergent, individually sufficient clues for
  one gold answer become incorrect specifically when combined.
