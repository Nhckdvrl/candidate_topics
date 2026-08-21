# When Does Uncontrollability Generalize?

**Status:** active candidate / falsification-first pilot  
**Working subtitle:** Experience diversity and higher-order agency priors in LLM agents

## Natural question

> If a learner repeatedly discovers that its actions do not affect outcomes, does it learn only that *this situation* is uncontrollable, or does it acquire a broader expectation that acting is usually futile?

The sharp comparison is:

> With the **same amount of uncontrollable experience**, does experience concentrated in one recurring context or distributed across many unrelated contexts produce more passivity in a **new, objectively controllable** task?

This question exists independently of LLMs. LLM agents are useful because we can hold the causal dynamics exactly fixed while changing only the semantic identity of the situations.

## Why this is not a fabricated phenomenon

The candidate is anchored by a long literature on the generalization of controllability.

- **Lieder, Goodman & Huys (CogSci 2013)** formalize controllability hierarchically: action-specific transitions -> situation-level controllability -> how controllable the world is in general. Their paper explicitly predicts that, because generalization strengthens with the variability of examples, *multiple stressors should produce more general effects than an equivalent amount of stress in a single situation*.
- **Tiggemann & Winefield (1978)** found helplessness transferred strongly to a similar test task but not a dissimilar one, making cross-situation generalization itself an old empirical question rather than a new LLM metaphor.
- **Granwald et al. (Communications Psychology 2025)** fit 279 participants across two differently framed active-avoidance tasks and found that a model sharing one task-invariant prior fit better than separate task-specific priors. This supports the existence of a cross-task action-success prior, but does **not** establish how uncontrollable experience causally creates it.
- **Hew & Bramley (CogSci 2026)** manipulated control in dynamic causal environments and found that low practical control induced later passive behavior. Their public repository provides experiment code, participant data and analyses. We use this as a conceptual/experimental seed, not as copied implementation.
- The classic **yoked-control** logic remains crucial: controllable and uncontrollable learners should be exposed to the same outcomes, differing in whether their actions caused those outcomes. Maier & Seligman's 2016 review explains why this isolates contingency from mere exposure.

One conceptual caveat is important: modern neuroscience revised the original animal-level story that passivity itself is literally learned. This project therefore studies the **behavioral/computational generalization of action-outcome controllability in agents**; it does not claim that an LLM instantiates the animal neural mechanism of learned helplessness.

See [`LITERATURE.md`](./LITERATURE.md) for the collision audit and exact boundaries.

## Core hypothesis

Let `U` denote uncontrollable training histories, `C` controllable histories, `1` concentrated semantic context, and `10` distributed semantic contexts.

The primary prediction is not simply `U10 < U1`. We use a 2x2 design:

| training controllability | concentrated | distributed |
| --- | --- | --- |
| controllable | C1 | C10 |
| uncontrollable | U1 | U10 |

All four cells receive the same number of episodes, same action space, same latent episode plans, same episode boundaries, same intervention budget, and the same held-out test distribution.

The main estimand is the difference-in-differences on the **first test action**:

```text
interaction = (U10 - U1) - (C10 - C1)
```

where each term is the probability of an active intervention on test step 1.

A negative interaction means semantic diversity specifically strengthens the transfer of uncontrollability, over and above any generic effect of task switching or semantic variety.

## Environment: no fixed dataset is required

This is a causal-learning experiment, so a fixed benchmark dataset would be the wrong object. The repository contains a small controlled environment generator.

Each episode has:

- state/readout: an integer, initialized at `-4`, `-3`, `3`, or `4`;
- objective: keep the reading close to zero;
- actions: `A`, `B`, `C`, `WAIT`;
- intervention budget: 6 active interventions per 10-step training episode;
- hidden action effects in controllable episodes: a per-episode permutation of `{-1, 0, +1}`;
- uncontrollable episodes: actions are ignored and effects follow an exogenous schedule.

### Exact yoking

For every `(base_seed, diversity)` pair:

1. run the controllable learner first;
2. record the **realized effect on every training step**;
3. run the uncontrollable learner with the same latent episode plans, but force its effects to replay the controllable learner's realized effect sequence regardless of its actions.

Thus a paired C/U learner receives the same raw training outcome trajectory while only C has action-outcome contingency.

### Diversity manipulation

The causal mechanics do not change.

- `concentrated`: 10 episodes use one semantic family repeatedly;
- `distributed`: 10 episodes use 10 different semantic families;
- episode count and reset structure are identical;
- for the same `base_seed`, **concentrated and distributed use the same latent episode plans**;
- test families are disjoint from all training families.

Training wrappers currently include greenhouse, factory, bakery, traffic control, clinic, warehouse, theater, network operations, irrigation and cargo ship. Test wrappers are observatory, museum preservation lab, harbor control and data center.

## Why test step 1 is the primary outcome

All histories end in the same kind of novel, objectively controllable test. The primary endpoint is:

```text
P(active intervention on test step 1)
```

At that instant the learner has received **zero feedback from the new task**. Any group difference therefore comes from the prior interaction history rather than from different evidence acquired during testing.

Secondary outcomes are deliberately subordinate:

- first-three-step intervention rate;
- full-test intervention rate;
- time to first intervention;
- test control performance / absolute state;
- recovery after positive controllability evidence.

No self-report helplessness question, confidence proxy, hidden-state probe, hint, or altered test difficulty is part of G0.

## Validation order

The pilot is deliberately short and kill-oriented.

### G-1 — environment and plumbing audit

```bash
pip install -r requirements.txt
./run_preflight.sh
```

This checks:

- controllable and uncontrollable random-policy outcome marginals match;
- action-effect mutual information is high only in the controllable environment;
- exact-yoke replay reproduces the paired state trajectory;
- train/test semantic families are disjoint;
- concentrated/distributed episode construction is correct;
- parser, metrics, and end-to-end subject execution work.

The audited implementation currently passes all tests locally.

### G0 — cheap behavioral pilot

Serve one instruct model behind an OpenAI-compatible endpoint, then:

```bash
export MODEL=<served-model-name>
export OPENAI_BASE_URL=http://localhost:8000/v1
./run_pilot.sh
```

Default pilot:

```text
40 base seeds x 4 cells = 160 subject histories
10 training episodes x 10 steps = 100 training experiences / subject
8-step novel controllable test
```

The pilot asks three questions in order:

1. **Local sensitivity:** by late training, does uncontrollability reduce the probability of intervening at the start of a new episode?
2. **Cross-task transfer:** pooled across diversity, does uncontrollable history reduce intervention on the first action of the held-out controllable test?
3. **Diversity effect:** is the locked 2x2 interaction negative?

If the model does not even express local sensitivity to controllability, stop rather than escalating to memory systems or finetuning.

### G1 — locked 1,000-history confirmation

Only after the pilot is alive:

```bash
./run_full_confirmation.sh
```

This defaults to:

```text
250 base seeds x 4 cells = 1,000 subject histories
```

The environment, families, endpoints, contrast, thresholds, and analysis are frozen before confirmation. See [`VALIDATION.md`](./VALIDATION.md).

## Interpretation matrix

| result | interpretation | candidate decision |
| --- | --- | --- |
| `U < C` and `(U10-U1)-(C10-C1) < 0` | uncontrollability transfers, and diversity increases its abstraction/generalization | strongest result; replicate across model family and then study persistence mechanism |
| `U < C`, interaction ~ 0 | uncontrollability transfers but diversity is not the driver | diversity hypothesis falsified / major downgrade |
| transfer only when train/test semantics are similar | similarity-based transfer rather than a global agency prior | motivates a separate Experiment 2 only if G0 is robust |
| no `U < C` transfer | agent localizes the experience or ignores it across task boundary | kill as an AI candidate |
| reliable positive interaction | concentrated uncontrollability generalizes more | hypothesis wrong; inspect as a potentially interesting opposite phenomenon, without relabeling it post hoc |

## The context-priming boundary

The first experiment deliberately uses raw interaction history because it is the cheapest way to test whether the behavioral phenomenon exists. If positive, the defensible claim is:

> accumulated interaction history induces cross-task changes in the agent's intervention policy consistent with a higher-order controllability prior.

It is **not** yet evidence that model parameters acquired a durable worldview.

A stronger follow-up should preserve the identical causal experiment while changing the retention mechanism:

1. raw in-context interaction history;
2. persistent episodic / distilled memory;
3. actual online parameter adaptation (e.g. LoRA/SFT/RL).

Do not start at level 3. If the behavioral phenomenon is absent at level 1, expensive training is not a rescue strategy.

## Collision boundary as of 2026-08-21

Our search found adjacent work, but no controlled LLM-agent study matching the central contrast:

```text
same amount of uncontrollability
x one versus many semantic task families
-> first action in a never-seen controllable task
with matched latent dynamics and yoked outcomes
```

Relevant adjacent work includes:

- ReasoningBank (ICLR 2026): agents distill transferable memory from successful and failed trajectories; this establishes that cross-task experience reuse is an active agent problem, not the controllability-generalization question itself.
- *State-Dependent Refusal and Learned Incapacity in RLHF-Aligned Language Models* (2025 preprint): a qualitative single long-dialogue analysis using learned helplessness as an analogy; it does not manipulate action-outcome contingency or experience diversity.
- recent self-evolving-agent work broadly studies learning from past failures, generally with the objective of improving future performance rather than asking when negative experience produces over-generalized passivity.

Novelty must be re-checked before paper submission; this is a candidate registration, not a novelty guarantee.

## Repository layout

```text
06_uncontrollability_generalization/
├── README.md
├── VALIDATION.md
├── LITERATURE.md
├── requirements.txt
├── run_preflight.sh
├── run_pilot.sh
├── run_full_confirmation.sh
├── src/
│   ├── environment.py
│   ├── renderers.py
│   ├── agent.py
│   ├── experiment.py
│   ├── audit_environment.py
│   └── analyze.py
└── tests/
```

The code is an original minimal implementation. The Hew & Bramley repository is treated as an experimental reference rather than vendored/copied code.
