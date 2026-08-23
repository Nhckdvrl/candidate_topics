# Master Topic Search Log

This file is the lightweight index for the breadth-first search rounds under `master_topic_search`.

The detailed notes live in per-round files. This index should stay short and make it easy to see what was searched, what survived, and what was killed.

**Current selection rules:** [`SELECTION_BAR_V2.md`](./SELECTION_BAR_V2.md). The active shortlist is intentionally allowed to shrink after re-screening.

| Round | Date | Main search baskets | Strongest survivors | Important kills / lessons |
| --- | --- | --- | --- | --- |
| [Round 01](./round01_2026-08-23.md) | 2026-08-23 | observed anomalies; old problems whose study may change in the LLM era; lab-adjacent special tasks | scientific-document/meta-research at scale; misconception/question-design science; human–AI disagreement confidence anomaly; non-toy learning-dynamics anomalies | synthetic students and synthetic populations are already becoming crowded; do not mistake a clean old-task automation story for a new scientific question |
| [Round 02](./round02_2026-08-23.md) | 2026-08-23 | labmate-style `seed paper → one meaningful axis`; factual acquisition/access; implicit behavioral adaptation; citation certainty; peer-review/meta-science | fact encoding→recall trajectory; matched positive vs negative adaptation; claim-preserving citation certainty drift | contextual-entrainment training dynamics downgraded after copying/in-context-learning collisions; reviewer-overlap and multiple-discovery questions already occupied; generic Agentic-RL feedback internalization remains too crowded |
| [Round 03](./round03_2026-08-23.md) | 2026-08-23 | complete top-five without lowering bar; old method-reporting bottlenecks; recent representation→behavior rotations | shortcut-citation method-information decay; memory age→overwriteability | source-label alignment origin killed by nearby controlled alignment-bias studies; spacing collided with Aug-2026 continual-pretraining work; preprint→publication claim drift already scaled to 72k+ studies; evaluator-vs-generator asymmetry crowded |
| [Round 04](./round04_2026-08-23.md) | 2026-08-23 | re-screen all five for external support, positive-result excitement, explicit method opening, archive failure patterns and latest collisions | **citation transmutation at scale** (`STRONG_KEEP`); **shortcut-citation method reconstructability / silent divergence** (`KEEP_REFRAME`) | fact encoding→recall temporal lag killed after new Jul-2026 continual-fact-learning collision and weak `then what?`; memory-age→overwrite moved to `HOLD_FOR_EXTERNAL_EVIDENCE`; inhibition downgraded to `CHEAP_G0_ONLY` because the causal sign axis is still inferred from confounded task families |

## Current active shortlist

### Serious candidates

1. **[Do Scientific Claims Become More Certain as They Are Cited?](./candidates/do_scientific_claims_become_more_certain_as_they_are_cited.md)** — `STRONG_KEEP`
   - exciting positive: citation itself systematically inflates the epistemic status of unchanged claims without new evidence;
   - method opening: evidence-conditioned citation verifier / certainty-preserving scholarly-writing guard.

2. **[Does Methodological Information Decay Along Shortcut-Citation Chains?](./candidates/does_methodological_information_decay_along_shortcut_citation_chains.md)** — `KEEP_REFRAME`
   - current mother question should be reframed as **method reconstructability / silent implementation divergence**, not a guessed monotonic depth law;
   - method opening: protocol-lineage resolver / shortcut-citation linter / method-diff repair system.

### Cheap prerequisite only

3. **[Is Negative Behavioral Adaptation Intrinsically Harder?](./candidates/is_negative_behavioral_adaptation_intrinsically_harder.md)** — `CHEAP_G0_ONLY`
   - run only the fully matched positive-acquisition vs negative-inhibition experiment;
   - do not treat as a project unless the sign effect remains large and cross-model stable;
   - if alive, method opening is inhibition-aware implicit memory / negative-experience post-training.

## Removed from active shortlist

4. **[When Does a Fact Become Recallable?](./candidates/when_does_a_fact_become_recallable.md)** — `KILL CURRENT FORMULATION`
   - temporal lag is an unsupported axis rotation;
   - Jul-2026 continual fact-learning work already shows diverse restatements directly reduce the recitation→use gap and improve later retention, occupying the most obvious intervention path.

5. **[Do Language-Model Memories Consolidate With Age?](./candidates/do_language_model_memories_consolidate_with_age.md)** — `HOLD_FOR_EXTERNAL_EVIDENCE`
   - `Fresh in Memory` proves acquisition recency is represented, not that it controls overwriteability;
   - do not repeat the repository's representation→behavior bridge failures without a direct behavioral anomaly.

See [`candidates/README.md`](./candidates/README.md) for the detailed current status.

## Logging rule

Every future round should record both survivors and rejected ideas.

A useful failed lead is not wasted work if the log makes clear:

- what initially looked promising;
- what exact collision or conceptual weakness killed it;
- whether the positive result would actually be exciting;
- what method / intervention opening would follow if the phenomenon were real;
- what broader search lesson transfers to later rounds.

Do not silently drop candidates and rediscover them later.