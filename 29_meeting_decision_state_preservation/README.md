# 29 — Decision-State Preservation in Meeting Summaries

**Status: REGISTERED / PUBLIC-ARTIFACT PREFLIGHT PASSED / TEMPORAL-PREFIX G0 PASSED.**

## Mother question

> When a meeting transcript discusses, proposes, tentatively accepts, conditions, rejects, revises, and finally decides the same proposition, does summarization preserve that proposition's decision state—or silently upgrade/downgrade it while keeping the surface fact roughly correct?

The target variable is an external, pre-model construct: the action/commitment state of a proposition in a collaborative decision process. This is not generic meeting hallucination.

## Scientific object

Primary states: `OPEN -> PROPOSED -> TENTATIVE / CONDITIONAL -> DECIDED`, with `REJECTED` and revision/reversal branches.

Primary failure classes: proposal→decision, tentative→final, conditional→unconditional, open→resolved, rejected→accepted/decided, and stale earlier decisions retained after revision.

## Why executable

AMI's released annotation stack contains ordered dialogue acts with timestamps, abstractive summaries with a dedicated `decisions` type, and summlink links from a decision abstract to supporting utterances. The public `guokan-shang/ami-and-icsi-corpora` converter documents this exact JSON contract. This independently identifies decision propositions and their evidence chains, but it does **not** independently label every intermediate source state. In particular, selecting `decisions` conditions the pool on an eventual decision; a proposal-looking support chain may encode implicit consensus or incomplete extractive linking rather than a true unsupported upgrade.

QMSum can be added later for free-form transfer, but is not required for the first test.

## Frozen P0 artifact audit

Run `audit_ami.py` over processed AMI annotations. Gates:

- >=200 decision abstracts;
- >=100 decision abstracts linked to >=2 source utterances;
- >=75 linked chains spanning >=15 seconds;
- >=100 chains with an explicit conservative state cue.

This is outcome-blind and makes no model call.

## G0a — natural summary drift

For each eligible decision chain, construct a bounded transcript window containing the linked decision-support utterances. Before treating source→summary differences as scientific errors, validate source-state labels on a small blinded sample or use temporally truncated prefixes with independently licensed states. Then generate a short minutes-style summary with one fixed open model/prompt and score:

1. decision-state level;
2. conditionality;
3. rejection/negation;
4. revision recency.

Primary headline metric: **unsupported state-upgrade rate** among chains whose licensed source state is not unconditionally decided.

`decision_state.py` is intentionally conservative and deterministic; it distinguishes uncued `UNKNOWN` from explicit `OPEN`, avoids treating rejection as the bottom of a false ordinal scale, and uses genre-specific rules for transcript versus minutes text. It is a triage scorer, not source-state ground truth or the final paper annotation scheme.

## G0b — matched contrast

Only after G0a establishes the object, build proposition-matched state contrasts such as `propose` vs `decided`, `agreed if X` vs `agreed`, and `rejected X` vs `decided X`. This isolates decision state from topic content.

## Method runway

**Decision-State-Preserving Summarization**: maintain a ledger

`proposition -> speaker/owner -> state -> condition -> revision history -> finality`

before generation, then constrain the summary to the latest licensed state.

## Collision position

Prior meeting work already studies decision-focused summarization, consensus, omission, hallucination and factuality. The distinct narrative here is proposition-level **state transmutation under compression**, its boundary conditions, and a structured preservation remedy.

## Public-data preflight receipt

- Official AMI manual annotations v1.6.2 were downloaded and converted with public converter commit `81716f66`.
- Exact counts: 624 linked decision abstracts, 366 multi-turn chains, 281 chains spanning at least 15 seconds, and 180 chains with a conservative explicit lexical state cue. All four frozen artifact-support conditions pass.
- A fixed-model temporal-prefix G0 now provides direct phenomenon evidence: on 52 content-grounded, explicitly non-final prefixes, Qwen2.5-7B-Instruct upgraded 39 to an unconditional decision (75.0%). On the identical inputs, one explicit state-preservation instruction reduced this to 0/52.
- Code review and full interpretation are recorded in `REVIEW_AND_PREFLIGHT_RESULTS.md`.
- Full G0 receipt and scope limits are recorded in `G0_RESULTS.md`.
- Next paper step: human-adjudicate the prefix state and proposition alignment, then expand across models and a transfer corpus. These are publication-validity steps, not additional gates on whether the topic is worth pursuing.
