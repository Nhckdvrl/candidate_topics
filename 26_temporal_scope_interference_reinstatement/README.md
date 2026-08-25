# 26 — Temporal Scope Interference & Reinstatement

**Status: ARCHIVED AT PREFLIGHT / STOP_INSUFFICIENT_EXACT_SUPPORT**

The frozen 2026-08-25 preflight found `0/512` exact eligible items in the
official downloaded artifact. The released file has no turn-level
`present_day_answer` on any turn, so the registered target and stable-donor
objects cannot be constructed without changing the frozen metadata contract.
No tokenizer measurement or model inference was run. See `G0_RESULTS.md`.

## Natural scientific question

> When a conversation has established a historical time frame, does that temporal scope disappear mainly because it becomes distant, because semantically related material interferes with it, or because present-day cues pull the model toward a default "now" interpretation? If the scope is disrupted, can a weak contextual cue restore correct use without restating the year or answer?

The seed phenomenon is not ours to rediscover. Atri, Johnson & Hartvigsen, **ChronoScope: Evaluating Temporal Consistency in Multi-Turn Language Models** (ACL 2026 Main), already shows that models abandon historical scope across turns, often substitute present-day facts, and still drift under Gold/Oracle conversational context.

Our question is the explanatory layer ChronoScope leaves open: **what kind of conversational state is temporal scope, what disrupts it, and what can behaviorally reinstate it?**

## Why this is not a re-registration of Topic 05

The closest internal archive is Topic 05, *Temporal Forgetting: Lost Skill or Lost Entry Point?* It was stopped because supplying an earlier correct reasoning prefix changes the task and cannot identify retained uncued competence.

Topic 26 does **not** infer hidden retention from rescue. It manipulates an observable conversational binding while keeping the final question and historical answer fixed. A reinstatement gain is interpreted only as:

> a weak temporal-context cue causally improves historical-scope behavior after a matched distractor.

It is **not** evidence that a latent scope representation remained stored, nor that the model "remembered but could not access" it. Hidden-state retention/decodability is a later question only if the behavioral object survives G0.

Topic 07 is also adjacent but non-colliding: it studied PI-vs-RI memory interference across sequence-model update architectures, not conversational temporal scope.

## External collision audit in one paragraph

ChronoScope already includes `Carryover-Then`, so merely showing that the word *then* helps is not a contribution. Recent work on multi-turn context interference studies irrelevant retrieved documents in search agents, while recent temporal-context reinstatement work studies episodic/order memory in long contexts. Neither occupies the matched conversational experiment here: **same final temporal probe, same Gold history, randomized intervening content, other-entity vs same-entity semantic content, a bounded present cue on the exact same stable fact, and a weak post-interference return cue.** See `LITERATURE.md`.

# Experimental object

G0 uses only two-turn **ChronoScope `carryover` chains** with temporal truth and a drift-identifiable final turn:

```text
Turn 1: explicit historical year + factual question
        -> Gold assistant answer
Turn 2: implicit follow-up whose historical answer differs from its 2025 answer
```

The final probe is copied byte-for-byte into every condition. Prior assistant answers and filler acknowledgements are deterministic; there is no self-conditioned error propagation.

Each target must also have a different-property **temporally stable fact** about the same entity: its historical and 2025 values must be identical. That stable fact is deliberate. It lets the present-cue contrast change the temporal cue without simultaneously changing the proposition's truth value.

# Frozen G0 conditions

For one target item, let `F(X)` be a temporally stable non-target fact about target entity `X`, and let `F(Y)` be a matched stable fact with the same property about another entity `Y`.

All filler messages explicitly say they are a separate aside that does not change the main discussion's time frame. This makes any induced drift a failure of scope maintenance rather than a legitimate conversational scope switch.

```text
baseline
    anchor -> Gold answer -> identical probe

neutral_1
    anchor -> Gold -> unrelated-entity stable fact F(Y) -> probe

neutral_2 / neutral_4
    anchor -> Gold -> 2 or 4 unrelated stable facts -> probe

same_entity_semantic
    anchor -> Gold -> stable fact F(X) -> probe

bounded_present
    anchor -> Gold -> "As of 2025, F(X)" -> probe

bounded_present_reinstate
    anchor -> Gold -> "As of 2025, F(X)"
    -> "Return to the earlier time frame from the original question."
    -> probe
```

The semantic/present pair uses the **same entity, same property, same value**. The fact is stable across time. The intended change is the present-time cue, not a new conflicting answer.

## Primary contrasts

All contrasts are paired by item and reported with a paired bootstrap 95% interval:

```text
neutral decay       Acc(neutral_1) - Acc(neutral_4)
same-entity penalty Acc(neutral_1) - Acc(same_entity_semantic)
present-cue penalty Acc(same_entity_semantic) - Acc(bounded_present)
reinstatement gain  Acc(bounded_present_reinstate) - Acc(bounded_present)
```

We also report present-day drift directly: an answer counts as present drift only when it is wrong under the historical gold and matches the released 2025 answer for that exact target turn.

A branch is called supported in the G0 summary only when its paired accuracy delta is at least `+0.05` and the paired bootstrap 95% lower bound is above zero. Failure of one branch does not silently kill the others.

# Frozen panel and stop rules

```text
model                 Qwen/Qwen2.5-7B-Instruct
source                 yashkumaratri/ChronoScope merged_scope_benchmark.jsonl
family                 carryover only
truth type             temporal only
selection seed         20260825
G0 items               512 exact eligible targets
decoding               greedy
temperature            0
max new tokens         24
primary prompt-gap cap <= 16 tokenizer tokens per item
```

Hard stops before interpretation:

1. fewer than 512 exact eligible target+donor objects;
2. final probe differs across conditions;
3. historical and present target answers are equal;
4. the same-entity donor is not stable across historical/present truth;
5. same-entity donor PID equals target PID;
6. primary neutral/semantic/present prompt token spread exceeds 16 tokens for any item.

Do not repair a failed support gate by changing model, seed, family, threshold, donor definition, cue wording, or sample subset after seeing outcomes.

# Public-artifact audit note

The current public Stage-3 writer annotates `present_day_answer` on each **turn**. The current public `hf_scope_benchmark.py` drift scorer reads `chain.get("present_day_answer")`. Topic 26 therefore reads the turn-level field directly and does not copy that chain-level lookup. Accuracy and message construction otherwise follow the released artifact conventions.

This is treated as an artifact-interface discrepancy, not as evidence against the published paper.

# Run

```bash
cd 26_temporal_scope_interference_reinstatement
pip install -r requirements.txt
./download_data.sh
./run_g0.sh data/merged_scope_benchmark.jsonl
```

Panel construction is outcome-blind and happens first. If exact support is below the frozen 512-item requirement, `prepare` exits before model inference.

Local logic tests:

```bash
python -m unittest discover -s tests -v
```

# If G0 is positive

Only then expand in this order:

1. locked confirmation on a disjoint deterministic holdout;
2. cue-dose curve: exact year -> weak relational cue -> no cue;
3. interference law across entity/property/time similarity;
4. cross-model generalization;
5. hidden-state trajectory / year decoding as characterization, not as proof of storage;
6. causal or external-state mitigation if a stable behavioral boundary exists.

The paper should remain phenomenon-first. Mechanism work is not allowed to manufacture a story after a weak behavioral G0.
