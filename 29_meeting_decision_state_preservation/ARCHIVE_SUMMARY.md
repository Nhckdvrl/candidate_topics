# Archive Summary — Topic 29: Decision-State Preservation in Meeting Summaries

**Final status: ARCHIVED / TEMPORAL-PREFIX PHENOMENON REAL / STANDALONE QUESTION TOO NARROW**

## Original question

> When a meeting transcript proposes, tentatively accepts, conditions, rejects, revises, and finally decides the same proposition, does summarization preserve that proposition's decision state—or silently upgrade it while keeping the surface content roughly correct?

The intended object was proposition-level decision state under meeting summarization: `PROPOSED / TENTATIVE / CONDITIONAL / REJECTED / DECIDED`, plus revision history.

## What was actually established

The public AMI artifact was large enough for a clean feasibility test:

- 624 linked decision abstracts;
- 366 multi-turn linked chains;
- 281 chains spanning at least 15 seconds;
- 180 chains with a conservative explicit lexical state cue.

A repaired temporal-prefix G0 withheld the final linked decision turn and retained only prefixes with an explicit non-final cue. On content-grounded candidates, a neutral minutes prompt produced unsupported unconditional-decision language at the following rates:

- Qwen2.5-7B-Instruct: `39/52 = 75.0%`;
- Qwen3-8B: `34/53 = 64.2%`;
- Gemma-3-12B-IT: `22/50 = 44.0%`.

On the common 49-example grounded intersection, the rates remained `79.6% / 65.3% / 42.9%`.

A matched state-preservation instruction reduced the corresponding errors to `0/52`, `1/53`, and `0/50`.

Therefore the local behavioral phenomenon is real in the controlled temporal-prefix task. This archive must not rewrite the result as a failed or null experiment.

## Why the topic is archived anyway

The stop is about **research-question scale and longevity**, not phenomenon existence.

1. **The mother question is too small.** Decision-state transmutation is one narrow error subtype inside meeting summarization. Expanding the taxonomy to proposal/tentative/conditional/rejected/revised does not change that basic scale.
2. **The strongest G0 depends on an artificial truncation.** The experiment intentionally stops an eventual-decision chain before its final decision. This cleanly tests premature finalization, but it is not the ordinary full-meeting summarization setting. On the complete chain, reporting the eventual decision can be correct.
3. **The error is almost eliminated by one explicit instruction.** The matched preservation prompt reduces error from `44–75%` to `0–1.9%`. That is useful diagnostically but leaves a weak standalone method runway: a heavy state-ledger system is hard to justify when the failure nearly disappears under a simple instruction.
4. **Strong-model longevity is uncertain.** The topic risks becoming a transient default-prompt weakness rather than a durable NLP problem. A paper whose scientific importance collapses when a stronger model follows the instruction better is not a good standalone target.
5. **Broadening to “commitment inflation” would be a rescue by umbrella.** Epistemic uncertainty, evidentiality, deontic modality, intention, and meeting decisions are different semantic systems. Combining them after the fact would create an artificial construct and collide with existing uncertainty/factuality work rather than reveal a natural shared mother question.

The correct conclusion is therefore:

**ARCHIVE. Keep the empirical result as a possible supporting phenomenon for a future independently motivated project, but do not continue human annotation, model scaling, mechanism analysis, or umbrella expansion for Topic 29 itself.**

## Failure / stop type

**Layer A/P2/P4 — phenomenon real, but standalone scientific scale and method runway are insufficient.**

This is not an implementation failure, artifact failure, or falsification of the observed temporal-prefix behavior.

## Main lessons

1. **A large G0 effect is not sufficient evidence that the research question is good.** `75% -> 0%` can be experimentally striking while still belonging to a small application corner.
2. **Ecological scope matters separately from identification.** Artificial truncation improved causal clarity but narrowed the task away from ordinary full-meeting summarization.
3. **A trivial remedy can be a negative signal for paper scale.** If one sentence of instruction nearly removes the failure, the proposed structured method needs a broader independently justified problem before it is worth building.
4. **Do not rescue narrow phenomena by post-hoc generalization.** A new umbrella is valid only if the construct and operationalization are natural before seeing the local result.
5. **Ask the longevity question early:** if a stronger model reduces the error to a few percent, does the scientific question remain important? For Topic 29, the answer is not strong enough.

## Preserved artifacts

- `G0_RESULTS.md`
- `REVIEW_AND_PREFLIGHT_RESULTS.md`
- `audit_ami.py`
- `decision_state.py`
- `run_g0_temporal_prefix.py`
- `score_summaries.py`
- tests and public-artifact receipts

These artifacts remain valid evidence of the controlled phenomenon, but Topic 29 is closed as a standalone candidate.
