# User-Interest Topic Search

This is the **personal-interest research search track**. It is intentionally separated from advisor-facing topic search.

Primary domains:

- Agentic RL / interactive post-training
- reasoning RL and training dynamics
- continual/self-improving learning
- embodied AI / VLA / world-action models
- direct behavioral/mechanistic phenomena in modern open models

The existing `phenomenon_mining/` rounds (2026-08-23) belong conceptually to this track and should be treated as its first mining history. They focused on RL checkpoint dynamics, recovery, feedback use, action-chunk staleness, and VLA closed-loop behavior.

## Selection standard

This track may optimize for frontier-AI significance and the user's own long-term research taste. It does **not** need to look like a conventional Sasano-lab NLP project.

However, it still obeys the repository's failure lessons:

1. phenomenon first, mechanism second;
2. no bridge hypothesis from two unrelated papers;
3. no LLM judge as the core observable when a programmatic one is possible;
4. meaningful non-toy regime required;
5. no rescue by model/layer/threshold fishing;
6. positive result must be exciting and expose a method lever.

## Current strongest mining objects

- checkpoint-wise recovery dynamics in ordinary Agentic RL;
- behavior lost between peak and final RL checkpoints;
- event-conditioned stale control inside VLA action chunks;
- mechanically identifiable reasoning-strategy transitions;
- direct capability redistribution during post-training.

See `../phenomenon_mining/` for the current source ledger and first three rounds.
