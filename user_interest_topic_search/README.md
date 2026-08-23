# User-Interest Topic Search

This directory is the **personal-interest / frontier-AI research-search track**.

It is intentionally separate from [`advisor_topic_search`](../advisor_topic_search/). The purpose here is to search for research problems that match the user's own long-term technical interests and target venues, even when those problems would not naturally look like a conventional Sasano-lab NLP project.

The main goal is not to maximize short-term advisor fit. It is to discover research directions that are genuinely worth spending months on because they are scientifically interesting, technically deep, aligned with current frontier AI, and have a plausible path to strong AI/robotics venues.

The existing [`phenomenon_mining`](../phenomenon_mining/) directory is the first historical mining record for this track. Its agentic-RL / post-training / VLA rounds should be treated as prior search history, not as a separate third topic stream.

---

# 1. What this track is for

This track searches for questions in areas the user is personally willing to pursue deeply:

- modern LLM post-training;
- RL for reasoning and agents;
- Agentic RL / tool-interaction learning;
- continual / self-improving learning;
- reasoning behavior and training dynamics;
- VLA / embodied AI / world-action modeling;
- robot-policy mechanisms and closed-loop behavior;
- mechanistic analysis of modern open AI systems when grounded in a real behavioral phenomenon.

The intended publication bar is primarily:

- ICLR;
- ICML;
- NeurIPS;
- AAAI when appropriate;
- CoRL;
- RSS;
- ICRA;
- IROS;
- occasionally strong domain-specific venues if the scientific question fits better there.

This directory is not required to look like a classic NLP project and should not distort a strong frontier-AI problem merely to make it sound linguistically adjacent.

---

# 2. Core research philosophy

The strongest lesson from the archived numbered topics is that **elegant hypotheses have a low base rate**.

The failed pattern is:

```text
paper A shows phenomenon X
paper B exposes variable Y
therefore maybe Y explains X
-> build a careful experiment
-> relation is tiny / absent / opposite / unidentifiable
```

This track therefore follows a strict **phenomenon-first** process.

The order should be:

```text
real observed phenomenon
    ↓
independent reproduction / direct inspection
    ↓
characterize what actually changes
    ↓
identify one concrete bottleneck / mechanism candidate
    ↓
one clean intervention
    ↓
method only if the phenomenon exposes a real lever
```

Not:

```text
interesting mechanism story
    ↓
search for evidence that makes it true
```

A topic should preferably begin from a curve, trace, behavior, state transition, collapse, recovery, reversal, or other phenomenon that is already visible before interpretation.

---

# 3. Search theme range

## 3.1 S-tier: Agentic RL / interactive post-training

Primary objects:

- tool-using agents;
- multi-turn environment interaction;
- search agents;
- web/navigation agents;
- symbolic environments with programmatic evaluation;
- RL-trained language-model policies.

High-value questions include phenomena around:

- training-time phase changes;
- peak-to-final degradation;
- recovery after errors;
- action-selection rigidity;
- repeated failed actions / loops;
- state-conditional behavior changes;
- tool-feedback use;
- within-episode adaptation;
- cross-episode retention of learned environment structure;
- exploration/exploitation shifts;
- capability redistribution across RL checkpoints;
- reward improvement accompanied by loss of some useful behavior;
- discrepancy between first-try success and recovery / robustness.

Do **not** automatically turn structural facts such as masked observation tokens or sparse action tokens into hypotheses. First look for a concrete behavioral consequence.

## 3.2 S-tier: reasoning RL / post-training dynamics

Search around:

- strategy transitions across checkpoints;
- emergence/disappearance of verification;
- backtracking behavior;
- branching and alternative-solution use;
- collapse into templates;
- reasoning-length transitions;
- changes in successful-solution diversity;
- reward vs capability-boundary shifts;
- SFT-to-RL rank reversals;
- capability loss and restoration across post-training stages;
- non-monotonic generalization.

Priority goes to **mechanically recognizable behavior**, not arbitrary trace taxonomies requiring a judge model.

## 3.3 A-tier: continual / self-improving learning

Relevant phenomena:

- learning from experience without gradient updates;
- learning from failures;
- behavioral adaptation across repeated interactions;
- retention vs interference;
- non-monotonic acquisition;
- temporary capability loss and spontaneous recovery;
- cross-context transfer of newly acquired behavior;
- selective forgetting;
- self-generated data changing later behavior.

Avoid generic catastrophic-forgetting papers unless there is a surprising new regime or direct modern-LLM phenomenon.

## 3.4 S/A-tier: embodied AI / VLA / robot policies

Primary objects:

- VLA policies;
- diffusion/action-chunk policies;
- world-action models;
- open robot foundation policies;
- simulation environments with reproducible rollouts;
- deployment-time closed-loop behavior.

High-value phenomena:

- action-chunk staleness;
- delayed reaction to new observations;
- abrupt failure after contact/task-state transitions;
- recovery vs terminal failure states;
- perturbation response;
- correction latency;
- mode switching;
- action uncertainty vs actual task-space uncertainty;
- policy behavior under embodiment / observation / timing changes;
- training-time or inference-time world-model contributions that are directly behaviorally measurable.

Do not prioritize generic benchmark gains or architecture swaps.

## 3.5 A-tier: mechanistic interpretability tied to behavior

Mechanism work is allowed only after the exact behavioral phenomenon is established.

Healthy order:

```text
visible failure / transition
-> localize when/where it happens
-> identify causal component
-> intervene
```

Unhealthy order:

```text
interesting hidden representation
-> search for a behavioral story around it
```

SAEs, probes, activation patching, steering, layer analyses, attribution, etc. are methods, not topic generators.

---

# 4. What is explicitly out of scope or low priority

Unless a uniquely strong phenomenon appears, do not spend serious search budget on:

- generic RAG;
- generic agent harness architecture;
- prompt engineering;
- another benchmark leaderboard;
- model-X-on-dataset-Y comparisons;
- generic hidden-state probing;
- arbitrary layer correlation;
- generic entropy collapse already covered heavily in current RL literature;
- generic "RL improves reasoning";
- generic "feedback helps agents";
- generic "longer action chunks reduce reactivity";
- topics whose main contribution is engineering scale without a scientific question.

This track also does not need to search traditional linguistic-resource topics merely for advisor compatibility; those belong in the advisor track.

---

# 5. Evidence sources and how to search

Search breadth should be high. Conference papers alone are insufficient.

## 5.1 Primary literature

Prioritize 2025–2026 for fast-moving areas:

- ICLR / ICML / NeurIPS / AAAI;
- CoRL / RSS / ICRA / IROS;
- COLM when relevant;
- arXiv for very recent agentic RL / embodied work.

Inspect:

- main figures;
- appendices;
- ablations;
- failure examples;
- checkpoint curves;
- negative results;
- trace visualizations;
- limitations / future-work sections.

## 5.2 Code and released artifacts

Strong evidence sources include:

- official GitHub repos;
- training scripts;
- released checkpoints;
- rollout logs;
- benchmark trajectories;
- experiment configs;
- issues/discussions documenting unstable behavior.

A phenomenon with accessible checkpoints/logs is much more valuable than a proprietary anecdote.

## 5.3 Practitioner / engineering evidence

Search:

- frontier-lab technical blogs;
- RL training diaries;
- practitioner writeups;
- public trace analyses;
- reproducibility reports;
- research-engineering posts.

Repeated practitioner observations can be strong anomaly sources, but they must eventually be grounded in reproducible public experiments.

## 5.4 Search pattern

For every promising area, search in this order:

```text
1. What surprising behavior has already been observed?
2. Is it visible in a figure / trace / released run?
3. Is it reproduced independently?
4. Has someone already named and solved it?
5. What residue remains unexplained after subtracting existing work?
6. Can that residue be measured directly on an open system?
```

Do not start with:

```text
What hypothesis can we test here?
```

---

# 6. Topic-mining workflow

## Stage U0 — anomaly inventory

Write only what is directly supported:

```text
Observed phenomenon:
Source:
Exact system / task:
Magnitude:
Is code/checkpoint/log available?
Why surprising?
Closest existing explanation:
```

No project title yet.

## Stage U1 — collision audit

Search aggressively for:

- same phenomenon;
- same experimental contrast;
- same method opening;
- more general paper that already subsumes the idea.

Use statuses:

- `KEEP_MINING`
- `COLLIDED`
- `WEAK_EVIDENCE`
- `REPLICATE_NOW`
- `KILL`

A broad headline being occupied does not necessarily kill a more precise residue, but the residue must itself be meaningful.

## Stage U2 — cheap direct replication

Before mechanism work:

- reproduce the phenomenon;
- use a meaningful model/task regime;
- save dense checkpoints when dynamics matter;
- collect raw traces;
- use programmatic metrics whenever possible;
- avoid broad hyperparameter search.

## Stage U3 — phenomenon characterization

Ask:

```text
What exactly changed?
Which behavior changed first?
Which behavior stayed intact?
Is the change abrupt, monotonic, reversible, state-dependent, or task-dependent?
Can representative examples make the phenomenon obvious?
```

## Stage U4 — paper-shape audit

Only after the phenomenon is alive:

```text
Why should the field care?
What common belief does this update?
What simple causal factor is now worth testing?
What lever does the phenomenon expose?
What method would naturally attack the failure?
```

Only then may the idea become a numbered `topicXX` project.

---

# 7. Hard promotion requirements

A lead should not become a numbered candidate unless all are true:

1. **Observed object exists.** Not merely plausible.
2. **Meaningful regime exists.** No toy-only rescue.
3. **Direct observable exists.** Prefer programmatic behavior over semantic judges.
4. **Effect is large enough to care about.** Tiny correlations are not enough.
5. **Collision audit survives.** Exact 2025–2026 work checked.
6. **Positive result is exciting.** A strong audience should update.
7. **A method opening exists.** There is something concrete to improve.
8. **Compute fits available infrastructure.** Avoid projects requiring frontier-scale API spend or multi-node bandwidth-heavy training.
9. **Null result is informative.** Failure should not reduce to “maybe another model/layer works.”
10. **No rescue fishing is planned.** One frozen pilot must be able to kill it.

---

# 8. Anti-failure rules inherited from the archive

- No cross-paper bridge hypotheses as primary projects.
- No mechanism without mechanism-level behavioral replication.
- No latent construct whose meaning depends on many exclusions.
- No LLM judge as core y-axis unless measurement itself is the contribution.
- No layer/threshold/model fishing after a miss.
- No whole-profile correlation interpreted as fine-grained mechanism.
- No toy phenomenon promoted without meaningful-regime qualification.
- No natural-crossover experiment before checking crossover density.
- No increasingly complex gate stack to rescue an unnatural construct.
- No project whose best positive result still sounds unsurprising.

---

# 9. Boundary with the advisor track

This track answers:

> **What research would the user personally want to do if advisor-style constraints were removed?**

The advisor track answers:

> **What problem would plausibly fit Sasano-lab's actual research style and be easy to defend in that environment?**

A topic can appear in both tracks only if it independently satisfies both standards. Do not copy an idea from one directory into the other merely by rewriting the title.

Examples:

- Agentic-RL recovery dynamics -> user-interest track by default.
- VLA control staleness -> user-interest track.
- quiz clue structure -> advisor track.
- semantic-frame coverage -> advisor track.
- scientific-document analysis -> advisor track unless the scientific question itself becomes a broader frontier-AI issue.

---

# 10. Current history

The existing [`../phenomenon_mining/`](../phenomenon_mining/) rounds are considered the first mining history for this track.

Current surviving objects there include:

- checkpoint-wise recovery dynamics in ordinary Agentic RL;
- environment-facing behavior lost between peak and final RL checkpoints;
- event-conditioned stale control inside VLA action chunks;
- mechanically identifiable reasoning-strategy transitions;
- direct capability redistribution during post-training.

These are **mining objects, not registered topics**. They must still survive direct replication and collision audit before promotion.
