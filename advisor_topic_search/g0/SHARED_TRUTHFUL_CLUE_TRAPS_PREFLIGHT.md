# Shared Truthful-Clue Traps — Frozen Artifact-Only Preflight

**Status: FROZEN BEFORE CO-REVERSAL / CONSENSUS OUTPUTS.**

This is not a continuation or rescue of archived Topic 28's order-dependence or
destructive-conjunction explanations. It is a new candidate mother question:

> Do naturally authored, non-adversarial truthful clues contain transferable
> semantic traps that make independently trained QA system families leave the
> correct answer together and converge on the same wrong competitor?

No model inference, prompt, new dataset, answer judge, hidden state, or
mechanism is used. A failed preflight permanently stops this route. A pass only
authorizes separate registration and untouched modern-model confirmation.

## Literature boundary

The intended claim is narrower than nearby 2026 work:

- *Who Flips?* challenges correct answers with generated arguments for an
  incorrect option;
- *Lying with Truths* adversarially montages truthful fragments to manipulate
  belief;
- *LLMs as a Jury* characterizes cross-model consensus and shared-error floors;
- this preflight asks whether one ordinary, naturally authored QuizBowl clue
  creates a shared hypothesis switch under incremental revelation.

It cannot claim that truthful evidence can mislead models in general, or that
models can share errors in general.

## Frozen input object

Use only the complete G0-cleaned eligible transition table:

```text
28_progressive_truthful_clue_reversal/
  artifacts/analysis1/eligible_transition_features.csv
```

Frozen SHA-256:

```text
9b001242576a1ca7d15411e7872a9725fef3a1cd740e8eb9b082dd3006ccbc7e
```

Hard reconciliation:

```text
rows                  120,353
released reversals      8,102
configs                    93
families                   19
boundaries               2,241
questions                  782
```

One boundary is `(qid, clue_t, clue_t+1)`. Its risk set contains every released
AI config that was correct at `t` and had the exact adjacent `t+1` state.

## Frozen system-family taxonomy

Family assignment uses only config names and is frozen before shared-boundary
results. Prompt variants, sizes, and instruction variants from one lineage do
not count as independent families.

| Family | Config-name rule | Configs |
|---|---|---:|
| `retrieval_bm25` | starts `bm25_` | 8 |
| `retrieval_contriever` | starts `contriever_` | 10 |
| `retrieval_grit` | starts `grit_` | 3 |
| `rag_bm25` | starts `rag-bm25` | 12 |
| `rag_contriever` | starts `rag-contriever` | 4 |
| `rag_grit` | starts `rag-grit` | 6 |
| `t5_t0_ul2` | starts `T0`, `flan-`, or `ul2_` | 11 |
| `falcon` | starts `falcon` | 5 |
| `gemma` | starts `gemma-` | 2 |
| `gemini` | starts `google-gemini` | 2 |
| `cohere` | starts `cohere-` | 2 |
| `gpt_neo` | starts `gpt-neo` | 2 |
| `llama` | starts `llama-` or `meta-llama` | 6 |
| `mistral_mixtral` | starts `mistral` or `mixtral` | 4 |
| `openai` | starts `openai-` | 4 |
| `opt` | starts `opt-` | 4 |
| `phi` | starts `phi-` | 1 |
| `pythia` | starts `pythia-` | 6 |
| `vicuna` | starts `vicuna-` | 1 |

Unknown or multiply matched configs abort the run. RAG systems are grouped by
retrieval architecture rather than counted as independent copies of their
generator.

## Agent-marginal-preserving null

Within every exact:

```text
(config, category, to_clue_idx)
```

stratum, randomly permute the complete reversal payload across that config's
eligible boundaries. A payload contains:

```text
released reversal indicator
strict-alias-after support flag
normalized post-reversal prediction
```

Thus every permutation exactly preserves for each config and category/stage:

- eligible transition count;
- reversal count;
- strict-supported wrong-answer count;
- wrong-prediction multiset.

It destroys only which models co-reverse on which natural boundary and which
wrong answers align there. Use 1,000 permutations, seed `20260825`. The
plus-one permutation p-value has minimum `1/1001`.

## Primary co-reversal statistic

For boundary `b` and family `f`:

```text
x_bf = reversals in family f at b / eligible configs in family f at b
```

The cross-family overlap statistic averages `x_bf * x_bg` over all unordered
family pairs jointly represented at all boundaries. This prevents a family
with many near-duplicate configs from counting as many independent systems.

Report observed/null ratio, null z-score, and plus-one permutation p-value.
Raw config-pair overlap is diagnostic only.

## Frozen shared semantic-trap definition

A boundary is a shared trap only if all conditions hold:

```text
eligible configs                         >= 20
eligible families                        >= 8
released reversals                       >= 8
boundary hazard                          >= 0.20
family hits                              >= 4
top exact normalized wrong answer count  >= 5
top wrong-answer share of all reversals  >= 0.50
```

A family hit requires at least 25% of its eligible configs at that boundary to
reverse. Wrong-answer consensus uses only post-reversal predictions that fail
the frozen strict gold-alias check; the denominator remains all released
reversals, making consensus conservative. No semantic clustering, stemming,
alias addition, or LLM judging is allowed.

The same trap definition is applied to every null permutation.

## Frozen diagnostics

Report:

1. cross-family and raw-config co-reversal overlap versus null;
2. observed and null shared-trap counts;
3. trap boundaries, risk, hazard, family support, exact top competitor, and
   consensus share;
4. total exact-consensus reversal events within traps;
5. trap questions and categories;
6. trap incidence and mean family overlap by outcome-blind boundary-level
   specificity quartile;
7. representative natural clue traps.

Specificity is descriptive in this preflight and is not allowed to rescue a
failed core shared-trap gate.

## Frozen gates and verdicts

Artifact gates:

```text
input SHA / rows / reversals / configs      exact frozen values
families                                    == 19
boundaries                                  == 2,241
questions                                   == 782
risk>=20 and eligible-families>=8 boundaries >= 2,000
```

Scientific gates:

```text
cross-family overlap observed/null ratio     >= 1.25
cross-family overlap permutation p           <= 0.001
shared trap boundaries                       >= 20
unique shared-trap questions                 >= 20
shared trap categories                       >= 5
shared-trap count permutation p              <= 0.001
shared-trap count observed/null-mean ratio    >= 3.0
exact-consensus reversal events in traps      >= 100
```

Verdicts:

```text
STOP_SHARED_TRAP_ARTIFACT
    any artifact gate fails

GO_SHARED_TRUTHFUL_CLUE_TRAPS
    all artifact and scientific gates pass

STOP_SHARED_TRAP_ROUTE
    artifact gates pass but any scientific gate fails
```

A preflight-only or reduced-permutation run is always `DEBUG_NO_VERDICT`. Gates,
family mapping, strata, normalization, and trap thresholds cannot change after
observing full outputs.
