# Master Topic Search Log

This file is the lightweight index for the breadth-first search rounds under `master_topic_search`.

The detailed notes live in per-round files. This index should stay short and make it easy to see what was searched, what survived, and what was killed.

| Round | Date | Main search baskets | Strongest survivors | Important kills / lessons |
| --- | --- | --- | --- | --- |
| [Round 01](./round01_2026-08-23.md) | 2026-08-23 | observed anomalies; old problems whose study may change in the LLM era; lab-adjacent special tasks | scientific-document/meta-research at scale; misconception/question-design science; human–AI disagreement confidence anomaly; non-toy learning-dynamics anomalies | synthetic students and synthetic populations are already becoming crowded; do not mistake a clean old-task automation story for a new scientific question |
| [Round 02](./round02_2026-08-23.md) | 2026-08-23 | labmate-style `seed paper → one meaningful axis`; factual acquisition/access; implicit behavioral adaptation; citation certainty; peer-review/meta-science | [fact encoding→recall trajectory](./candidates/when_does_a_fact_become_recallable.md); [matched positive vs negative adaptation](./candidates/is_negative_behavioral_adaptation_intrinsically_harder.md); [claim-preserving citation certainty drift](./candidates/do_scientific_claims_become_more_certain_as_they_are_cited.md) | contextual-entrainment training dynamics downgraded after copying/in-context-learning collisions; reviewer-overlap and multiple-discovery questions already occupied; generic Agentic-RL feedback internalization remains too crowded |
| [Round 03](./round03_2026-08-23.md) | 2026-08-23 | complete top-five without lowering bar; old method-reporting bottlenecks; recent representation→behavior rotations | [shortcut-citation method-information decay](./candidates/does_methodological_information_decay_along_shortcut_citation_chains.md); [memory age→overwriteability](./candidates/do_language_model_memories_consolidate_with_age.md) | source-label alignment origin killed by nearby controlled alignment-bias studies; spacing collided with Aug-2026 continual-pretraining work; preprint→publication claim drift already scaled to 72k+ studies; evaluator-vs-generator asymmetry crowded |

## Current provisional top five

1. [When Does a Fact Become Recallable?](./candidates/when_does_a_fact_become_recallable.md)
2. [Is Negative Behavioral Adaptation Intrinsically Harder?](./candidates/is_negative_behavioral_adaptation_intrinsically_harder.md)
3. [Do Scientific Claims Become More Certain as They Are Cited?](./candidates/do_scientific_claims_become_more_certain_as_they_are_cited.md)
4. [Does Methodological Information Decay Along Shortcut-Citation Chains?](./candidates/does_methodological_information_decay_along_shortcut_citation_chains.md)
5. [Do Language-Model Memories Consolidate With Age?](./candidates/do_language_model_memories_consolidate_with_age.md)

These are provisional survivors only. They have **not** been promoted to numbered root-level Topics.

## Logging rule

Every future round should record both survivors and rejected ideas.

A useful failed lead is not wasted work if the log makes clear:

- what initially looked promising;
- what exact collision or conceptual weakness killed it;
- what broader search lesson transfers to later rounds.

Do not silently drop candidates and rediscover them later.