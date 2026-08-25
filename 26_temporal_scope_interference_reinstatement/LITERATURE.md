# Topic 26 — Literature and Collision Audit

Audit date: 2026-08-25

## Seed: ChronoScope (ACL 2026 Main)

Yash Kumar Atri, Steven L. Johnson, Thomas Hartvigsen. **Evaluating Temporal Consistency in Multi-Turn Language Models.** ACL 2026 Long.

- ACL: https://aclanthology.org/2026.acl-long.2133/
- code: https://github.com/yashkumaratri/ChronoScope
- dataset: `yashkumaratri/ChronoScope`

What it already establishes:

- temporal scope can be treated as a conversational state that must persist, switch, or transfer;
- more than one million deterministic Wikidata chains;
- historical scope is often abandoned in favor of present-day facts;
- failure persists under Gold/Oracle assistant history;
- follow-up depth increases drift;
- `Carryover-Then` already tests an explicit linguistic marker such as "then".

What it does **not** isolate:

- pure distance/turn decay under content-matched intervention;
- same-entity semantic interference versus other-entity filler;
- a bounded present cue on the same nonchanging factual proposition;
- recovery after interference from a weak cue that does not repeat the exact year.

Therefore Topic 26 is not "ChronoScope + one more model". It randomizes the intervening conversational content while keeping the final temporal probe fixed.

## General multi-turn degradation

**LLMs Get Lost In Multi-Turn Conversation** (ICLR 2026 Best Paper; Microsoft Research) establishes broad multi-turn unreliability and poor recovery after early mistakes. It is important background for sequential degradation, but it does not identify temporal-scope interference or present-default attraction on matched factual chains.

## Context interference in agents

**Mitigating Context Interference for Reliable and Efficient Search Agents** (ACL 2026) studies interference from retrieved documents in multi-turn search agents and proposes mitigation. Its experimental object is document/retrieval interference during search, not a latent historical conversational frame with a fixed final QA probe.

This is neighboring evidence that "context interference" is a meaningful systems problem, not a title-level collision.

## Temporal-context reinstatement in long-context memory

**Temporal Context Reinstatement Drives Episodic-Like Order Memory in Long-Context Language Models** (ICML 2026 / arXiv:2607.22575) studies episodic/order retrieval and mechanistic temporal reinstatement in long contexts. It makes the phrase *temporal context reinstatement* nonempty prior art.

Topic 26 therefore avoids claiming generic novelty for "reinstatement". Its independent object is **conversational historical scope after a controlled interference manipulation**. If mechanism work later finds a related trajectory, that paper must be treated as a direct mechanistic neighbor.

## Agentic memory reinstatement

**RaMem: Contextual Reinstatement for Long-term Agentic Memory** concerns external/long-term agent memory retrieval across sessions. It is conceptually adjacent but not the same observational unit or causal contrast.

## Internal collision card

### Closest numbered topic: 05 — Temporal Forgetting: Lost Skill or Lost Entry Point?

Failure: conceptual identification. Correct-prefix rescue changed the task and could not prove uncued skill retention.

Same scientific claim? **No.** Topic 05 was about training-time competence loss/access. Topic 26 is about an explicitly established conversational temporal binding under fixed weights.

Same identification route? **No.** Topic 26 does not supply part of the target solution. It changes only intervening non-target context, keeps the final probe identical, and interprets cue rescue behaviorally.

Transferred prohibition: **do not claim that successful reinstatement proves the temporal state remained internally stored.**

### Closest numbered topic: 07 — Old Blocks New, or New Erases Old?

Failure: the motivating PI>RI effect replicated, but the frozen cross-architecture explanatory separation was too weak/unstable.

Same scientific claim? **No.** Topic 07 asked whether memory-update architecture changes AB-AC PI/RI asymmetry. Topic 26 asks what kinds of intervening conversational content disrupt an already established temporal scope.

Transferred lesson: a robust seed phenomenon does not imply the proposed explanatory axis is large. Each interference branch has an explicit G0 effect-size gate and can fail independently.

## Collision verdict

**REGISTER.** No retrieved 2025–2026 neighbor occupies the full title-level question plus decisive matched contrast. ChronoScope supplies the phenomenon and artifact; it does not already perform the intervention matrix. The main novelty risk is overstating trivial "then/reminder" effects, which is why G0 specifically requires post-interference weak reinstatement and excludes `Carryover-Then` as the novel claim.
