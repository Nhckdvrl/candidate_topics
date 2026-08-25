# 28 — Information Non-Monotonicity under Progressive Truthful Clues

**Status: REGISTERED / FROZEN ARTIFACT-ONLY G0 IMPLEMENTED / LOGIC TESTS PASS / FULL DATA RUN NEXT.**

## Natural scientific question

> **If each new clue is truthful and relevant to the same target, can strictly more information make a QA system abandon a correct answer for a wrong one?**

The project studies **truthful-evidence reversal** / **information non-monotonicity**, not generic long-context degradation.

The exact event is:

```text
same question
same released AI agent
clues 1..t       -> correct
clues 1..t+1     -> wrong
new information   = the next official QuizBowl clue for the same gold answer
```

## External anchor and artifact

Seed: EMNLP 2024 Main, *Do great minds think alike? Investigating Human-AI Complementarity in Question Answering with CAIMIRA*.

Released Hugging Face artifacts:

```text
mgor/protobowl-11-13
    progressive-clues / eval
    ~3.8k cumulative QuizBowl questions

mgor/protobowl-11-13-agent-responses
    128 configs / train
    ~304k released response rows
```

The original work studies human/AI proficiency and complementarity. It does not use adjacent `correct -> wrong` transitions under cumulative truthful clues as the scientific object.

### Critical artifact caveat

Human responses are **not** valid evidence for this project: the CAIMIRA preprocessing explicitly backfills later human clue states after a correct response under a monotonicity assumption. Human rows are therefore excluded from the scientific G0 by construction.

The primary G0 uses only released non-human AI response rows.

## Why registration does not require an upstream phenomenon receipt

There is no published `correct -> wrong` seed effect being assumed. The released response trajectories themselves are the experimental object, and G0 directly measures the new mother phenomenon. Thus the first run is simultaneously the exact artifact receipt and the scientific density screen.

No model inference, API call, prompt recreation, or hidden-state work occurs before this screen.

# Frozen G0

## Unit of analysis

One trajectory is:

```text
(response config, released agent_type, original question qid)
```

Each cumulative state is indexed by official `qc_id = q<id>_<clue_idx>`.

Only **consecutive** cumulative states count for the primary transition:

```text
q123_2 -> q123_3    eligible
q123_2 -> q123_5    not primary; reported as a gap transition only
```

## Official primary correctness

Use the released binary `score` field as the primary correctness signal. Do not replace the official scorer with a new fuzzy matcher after seeing results.

As a canonicalization sanity check, join the released progressive-clue dataset and independently test whether predictions exact-match one of `clean_answers` after a frozen SQuAD-style normalization. This strict alias check is diagnostic/high-precision support; it does **not** replace the official score.

## Duplicate handling

A `(config, agent_type, qid, clue_idx)` cell must be unique.

- identical duplicates collapse;
- if duplicate rows disagree on correctness **or normalized prediction**, drop that whole ambiguous cell;
- never choose one conflicting row post hoc.

## Primary transition table

For every consecutive pair:

```text
wrong -> wrong
wrong -> correct
correct -> correct
correct -> wrong   <-- critical reversal
```

Primary reversal rate:

```text
R = N(correct -> wrong) / N(current state correct and next clue exists consecutively)
```

## Frozen viability gates

These are **paper-worthiness / critical-cell density** gates, not hypothesis p-value thresholds:

```text
question-metadata join coverage               >= 0.98
eligible consecutive transitions from correct >= 500
official correct->wrong events                >= 100
reversal rate R                               >= 0.02
unique questions with a reversal              >= 50
unique non-human configs with a reversal      >= 5
strict alias-stable reversal events           >= 30
```

Verdict:

```text
GO_REVERSAL_OBJECT
STOP_REVERSAL_OBJECT
STOP_ARTIFACT_CONTRACT
```

Do not lower these gates, change aliases, selectively remove configs/questions, or choose a different scoring rule after observing the result.

## Outputs

The G0 writes:

```text
artifacts/g0/
  dataset_receipt.json
  agent_type_inventory.csv
  transition_summary.json
  summary_by_config.csv
  summary_by_agent_type.csv
  trajectory_flags.csv
  reversal_events.csv
```

`reversal_events.csv` includes the before/after predictions and the exact newly added official clue text so high-value cases can be inspected without reconstructing the dataset manually.

# What a positive G0 establishes

A positive G0 establishes only:

> systematic released AI trajectories exist where adding the next truthful QuizBowl clue changes an already-correct answer to an incorrect one at nontrivial density.

It does **not** yet establish:

- that modern open-weight LLMs reproduce the same rate;
- why the new clue causes the reversal;
- that the correct answer representation disappeared;
- that clue order is causal;
- any hidden-state mechanism.

# Branch map after a positive G0

1. **Competitor introduction** — does the added clue activate a plausible competing entity/category?
2. **Clue specificity / ambiguity** — characterize which clue properties predict reversal.
3. **Order intervention** — hold the clue multiset fixed and permute order on a frozen local-model panel.
4. **Local open-model reproduction** — replay a frozen subset on Qwen/Llama/Gemma with deterministic prompting.
5. **Boundary mechanism** — only then compare correct-before vs wrong-after states.
6. **Causal rescue** — intervention only after the behavioral transition object is locked.

# Kill lines

Stop or demote if:

- response rows cannot be reliably ordered into exact same-agent trajectories;
- question join coverage is insufficient;
- reversals are too sparse;
- the events collapse under the frozen strict alias sanity check;
- a few corrupted questions/configs dominate the effect;
- released AI rows themselves were generated/backfilled under a monotonicity assumption;
- a direct recent paper already occupies adjacent truthful-clue `correct -> wrong` reversals.

# Implementation

Committed files:

```text
README.md
g0_progressive_reversal.py
run_g0.sh
requirements.txt
tests/test_g0_progressive_reversal.py
```

Local logic validation completed with Hugging Face network calls stubbed: **8/8 unit tests pass**. The tests cover ID parsing, normalization, exact added-clue extraction, human exclusion, conflicting duplicate removal, gap-transition exclusion, reversal detection, strict-alias support, and frozen verdict gates.

The current execution environment could not install `datasets` because outbound package access was unavailable, so the full public-data run has **not** been claimed as completed.

# Run

```bash
cd 28_progressive_truthful_clue_reversal
pip install -r requirements.txt
bash run_g0.sh
```

The first shot downloads only public Hugging Face metadata/data and performs no model inference.
