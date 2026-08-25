# Topic 26 — G0 Preregistration / Validation Contract

Date frozen: 2026-08-25

## Claim boundary

G0 can identify **behavioral sensitivity of historical temporal scope to controlled intervening content**. It cannot by itself identify where temporal scope is stored, whether a representation decays, or whether a successful cue proves latent retention.

## Identification logic

The experiment is deliberately within-item. For every selected ChronoScope target, the anchor question, Gold anchor answer, final implicit probe, historical final answer, and released 2025 final answer are identical across conditions.

The primary three-way manipulation uses a temporally stable non-target fact:

- `neutral_1`: same property class, stable fact, another entity;
- `same_entity_semantic`: same stable fact structure, target entity;
- `bounded_present`: exact same target-entity property/value as semantic, but with an explicit `As of 2025` cue.

Because the stable fact's value is identical historically and in 2025, `same_entity_semantic -> bounded_present` does not inject a newer contradictory value. It tests whether a bounded present temporal cue itself attracts the subsequent ambiguous probe toward now.

The filler preamble explicitly states that the aside does not change the main discussion's time frame. Therefore the present condition is not a normative scope-switch instruction.

`bounded_present_reinstate` then adds a weak relational cue that does not repeat the year, entity answer, or target property.

## Eligibility

A target is eligible iff:

1. chain family is exactly `carryover`;
2. chain truth type is exactly `temporal`;
3. chain has exactly two turns;
4. final turn has a nonempty turn-level `present_day_answer`;
5. normalized historical final answer != normalized present-day final answer;
6. subject, target PID, and integer anchor year exist;
7. the target subject has another PID with a released present answer equal to its historical answer;
8. at least four other subjects have a stable fact for that donor PID.

Donors are chosen without model outputs. For same-entity stable facts, deterministic lexical/PID order is used. Other-entity controls are selected by closest subject+answer character length, then lexical order. The candidate list is shuffled exactly once with seed `20260825`, and the first 512 are frozen.

## Model / decoding

```text
Qwen/Qwen2.5-7B-Instruct
greedy generation
max_new_tokens = 24
system prompt = released ChronoScope system prompt
prior factual assistant turn = released Gold answer
```

No answer sampling, reranking, semantic judge, or model-specific prompt rewrite is allowed in G0.

## Metrics

`relaxed_match` follows the released evaluator's intended short-answer normalization: lowercase, first line, punctuation removal, exact/contained normalized string match.

For each condition:

- historical accuracy;
- present-day drift rate over the exact drift-eligible panel;
- prompt token count.

Paired item bootstrap: 5,000 resamples, seed derived deterministically from `20260825` and contrast name.

## Branch-level interpretation

For delta `D = Acc(A) - Acc(B)`, call a directional effect **supported** only if:

```text
D >= 0.05
and bootstrap 95% lower bound > 0
```

The four frozen deltas are:

```text
D_decay       = Acc(neutral_1) - Acc(neutral_4)
D_semantic    = Acc(neutral_1) - Acc(same_entity_semantic)
D_present     = Acc(same_entity_semantic) - Acc(bounded_present)
D_reinstate   = Acc(bounded_present_reinstate) - Acc(bounded_present)
```

These are independent branch tests, not a conjunctive gate. Examples:

- only `D_decay`: decay-like distance sensitivity deserves follow-up;
- `D_semantic` without `D_decay`: content-specific interference is favored over pure turn-distance decay;
- `D_present`: bounded present cue has an additional attraction effect beyond same-entity content;
- `D_reinstate`: weak contextual reinstatement can behaviorally reverse part of the present-cue damage;
- none: `NO_LARGE_CONTROLLED_EFFECT`, stop this exact G0 story rather than prompt-shopping.

## Measurement gate

After tokenization but before scientific interpretation, compute per-item token counts for `neutral_1`, `same_entity_semantic`, and `bounded_present`. If any item's max-minus-min spread exceeds 16 tokens, return `MEASUREMENT STOP`; do not interpret scientific deltas.

The reinstate condition is intentionally longer because cue insertion is the treatment itself. Neutral distance conditions are intentionally longer by design.

## Artifact-interface discrepancy

At the public code state audited on 2026-08-25, the Stage-3 writer places `present_day_answer` inside each turn, while `hf_scope_benchmark.py` accesses a chain-level `present_day_answer` when computing drift. Our code follows the Stage-3 schema and final-turn field. This choice is frozen before any G0 model output.
