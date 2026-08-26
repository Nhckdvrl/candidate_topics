# 29 — Decision-State Preservation in Meeting Summaries

**Status: ARCHIVED / TEMPORAL-PREFIX PHENOMENON REAL / STANDALONE QUESTION TOO NARROW.**

> Final decision: do not continue this topic as a standalone candidate. The three-model temporal-prefix effect is preserved as a real empirical result, but the meeting-specific mother question is too small, the strongest G0 depends on artificial pre-decision truncation, and a one-line preservation instruction nearly removes the failure. See [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md).

## Mother question

> When a meeting transcript discusses, proposes, tentatively accepts, conditions, rejects, revises, and finally decides the same proposition, does summarization preserve that proposition's decision state—or silently upgrade/downgrade it while keeping the surface fact roughly correct?

The target variable is an external, pre-model construct: the action/commitment state of a proposition in a collaborative decision process. This is not generic meeting hallucination.

## Scientific object

Primary states: `OPEN -> PROPOSED -> TENTATIVE / CONDITIONAL -> DECIDED`, with `REJECTED` and revision/reversal branches.

Primary failure classes: proposal→decision, tentative→final, conditional→unconditional, open→resolved, rejected→accepted/decided, and stale earlier decisions retained after revision.

## Why executable

AMI's released annotation stack contains ordered dialogue acts with timestamps, abstractive summaries with a dedicated `decisions` type, and summlink links from a decision abstract to supporting utterances. The public `guokan-shang/ami-and-icsi-corpora` converter documents this exact JSON contract. This independently identifies decision propositions and their evidence chains, but it does **not** independently label every intermediate source state. In particular, selecting `decisions` conditions the pool on an eventual decision; a proposal-looking support chain may encode implicit consensus or incomplete extractive linking rather than a true unsupported upgrade.

QMSum could have been added for free-form transfer, but no further expansion is authorized for this archived topic.

## Frozen P0 artifact audit

Run `audit_ami.py` over processed AMI annotations. Gates:

- >=200 decision abstracts;
- >=100 decision abstracts linked to >=2 source utterances;
- >=75 linked chains spanning >=15 seconds;
- >=100 chains with an explicit conservative state cue.

All four gates passed on the public artifact.

## G0 result

A repaired temporal-prefix G0 withheld the final linked decision turn and retained only explicitly non-final prefixes. Unsupported unconditional-decision rates under a neutral minutes prompt were:

- Qwen2.5-7B-Instruct: `39/52 = 75.0%`;
- Qwen3-8B: `34/53 = 64.2%`;
- Gemma-3-12B-IT: `22/50 = 44.0%`.

On a common 49-example grounded intersection the same directional effect remained. A matched state-preservation instruction reduced the corresponding errors to `0/52`, `1/53`, and `0/50`.

The result establishes premature finalization in the controlled temporal-prefix task. It does **not** establish that ordinary full-meeting summarization has a comparably large decision-state problem.

## Why no further method work

The originally proposed Decision-State-Preserving Summarizer would maintain

`proposition -> speaker/owner -> state -> condition -> revision history -> finality`

before generation. After G0, however, a simple explicit preservation instruction already removes almost all of the measured failure. Combined with the narrow meeting-specific scope and artificial truncation used by the strongest test, this leaves insufficient standalone method runway.

Do not rescue this topic by model scaling, human annotation, mechanism work, or post-hoc expansion into a broad “commitment inflation” umbrella.

## Final archive pointer

See [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) for the authoritative stop rationale and transferable lessons.
