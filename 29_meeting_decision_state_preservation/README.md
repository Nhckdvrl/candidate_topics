# 29 — Decision-State Preservation in Meeting Summaries

**Status: REGISTERED / G0 IMPLEMENTED / PUBLIC-ARTIFACT PREFLIGHT NEXT.**

## Mother question

> When a meeting transcript discusses, proposes, tentatively accepts, conditions, rejects, revises, and finally decides the same proposition, does summarization preserve that proposition's decision state—or silently upgrade/downgrade it while keeping the surface fact roughly correct?

The target variable is an external, pre-model construct: the action/commitment state of a proposition in a collaborative decision process. This is not generic meeting hallucination.

## Scientific object

Primary states: `OPEN -> PROPOSED -> TENTATIVE / CONDITIONAL -> DECIDED`, with `REJECTED` and revision/reversal branches.

Primary failure classes: proposal→decision, tentative→final, conditional→unconditional, open→resolved, rejected→accepted/decided, and stale earlier decisions retained after revision.

## Why executable

AMI's released annotation stack contains ordered dialogue acts with timestamps, abstractive summaries with a dedicated `decisions` type, and summlink links from a decision abstract to supporting utterances. The public `guokan-shang/ami-and-icsi-corpora` converter documents this exact JSON contract. G0 therefore starts from independently annotated meeting structure rather than LLM-created labels.

QMSum can be added later for free-form transfer, but is not required for the first test.

## Frozen P0 artifact audit

Run `audit_ami.py` over processed AMI annotations. Gates:

- >=200 decision abstracts;
- >=100 decision abstracts linked to >=2 source utterances;
- >=75 linked chains spanning >=15 seconds;
- >=100 chains with an explicit conservative state cue.

This is outcome-blind and makes no model call.

## G0a — natural summary drift

For each eligible decision chain, construct a bounded transcript window containing the linked decision-support utterances. Generate a short minutes-style summary with one fixed open model/prompt. Score source→summary change along:

1. decision-state level;
2. conditionality;
3. rejection/negation;
4. revision recency.

Primary headline metric: **unsupported state-upgrade rate** among chains whose licensed source state is not unconditionally decided.

`decision_state.py` is intentionally conservative and deterministic; it is a high-precision first-pass scorer, not the final paper annotation scheme.

## G0b — matched contrast

Only after G0a establishes the object, build proposition-matched state contrasts such as `propose` vs `decided`, `agreed if X` vs `agreed`, and `rejected X` vs `decided X`. This isolates decision state from topic content.

## Method runway

**Decision-State-Preserving Summarization**: maintain a ledger

`proposition -> speaker/owner -> state -> condition -> revision history -> finality`

before generation, then constrain the summary to the latest licensed state.

## Collision position

Prior meeting work already studies decision-focused summarization, consensus, omission, hallucination and factuality. The distinct narrative here is proposition-level **state transmutation under compression**, its boundary conditions, and a structured preservation remedy.

## Validation receipt

- AMI public converter/readme schema independently checked: decision abstracts and extractive links are present.
- Local scorer and end-to-end preflight fixtures passed; 4 total Topic29/30 unit tests pass.
- No scientific model outcome was inspected during implementation.
- Next action: exact corpus-level AMI support count, then G0a immediately if support passes.
