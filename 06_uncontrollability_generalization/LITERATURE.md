# Literature and Collision Audit — 2026-08-21

## Scope

Question searched:

> With equal total exposure to uncontrollability, does spreading that experience across diverse situations produce stronger generalization of passivity / lower expected agency in a novel controllable situation than concentrating it in one situation?

Search families included combinations of:

- `learned helplessness generalization controllability Bayesian world prior`
- `situation similarity learned helplessness generalization`
- `multiple stressors variability generalization helplessness`
- `task invariant controllability prior active avoidance`
- `causal inference learned helplessness practical control`
- `LLM agent learned helplessness controllability generalization`
- `LLM failure memory persistent avoidance agent`
- `self-evolving agent failed trajectories cross-task memory`
- `learned incapacity RLHF language models`

This is a collision audit, not a systematic review.

## 1. Foundational computational account

### Lieder, Goodman & Huys (2013), *Learned helplessness and generalization*

- CogSci proceedings paper.
- Hierarchical Bayesian model separates:
  1. action-specific transition probabilities;
  2. situation-specific controllability;
  3. world-level beliefs about how controllable situations are on average and how much controllability varies across situations.
- The paper argues that learning about control in one situation can transfer to novel situations through generalization.
- Most importantly for Topic 06, it explicitly predicts that greater variability of examples strengthens generalization and therefore that **multiple stressors should have more general effects than an equivalent amount of stress in one situation**.

URL: https://web.stanford.edu/~ngoodman/papers/LiederGoodmanHuys2013.pdf

**What is already done:** a normative computational account and simulations of classic helplessness phenomena.

**What remains:** an empirical controlled test of the equal-amount, one-context-versus-many-contexts prediction in modern interactive agents.

## 2. Similarity-dependent transfer

### Tiggemann & Winefield (1978), *Situation Similarity and the Generalization of Learned Helplessness*

The abstract reports strong helplessness effects on a similar test task but not on a dissimilar test task and concludes that situation similarity is an important determinant of generalization.

DOI: https://doi.org/10.1080/14640747808400697

**Importance here:** a positive Topic 06 result should not automatically be called a global worldview. If transfer is limited to semantically similar tasks, the correct explanation is similarity-based transfer. That becomes a clean follow-up experiment after the main 2x2 survives.

## 3. Formal control and the yoked design

### Huys & Dayan (2009), *A Bayesian formulation of behavioral control*

Formalizes controllability through action-dependent outcome/state transitions and provides the computational lineage used by later hierarchical accounts.

DOI: https://doi.org/10.1016/j.cognition.2008.08.013

### Maier & Seligman (2016), *Learned Helplessness at Fifty: Insights from Neuroscience*

This review is important for two different reasons.

First, it describes the classic triadic/yoked design: the controllable and uncontrollable subjects can receive the same duration/intensity/pattern of outcomes while only the controllable subject's response changes the outcome. Topic 06 mirrors this logic with per-step outcome yoking.

Second, it explicitly revises the original animal mechanism: passivity to prolonged aversive stimulation is not simply a learned response, while detection/expectation of control plays a central learned role. Therefore Topic 06 should use "helplessness" as a behavioral/computational paradigm and avoid claiming biological equivalence.

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4920136/

## 4. Evidence that cross-task priors exist

### Granwald et al. (2025), *A task-invariant prior explains trial-by-trial active avoidance behaviour across gain and loss tasks*

Communications Psychology, open access.

- 279 participants.
- Two active-avoidance tasks differed in framing and outcome valence.
- Formal model comparison favored a model with one prior shared across tasks over models with separate task priors.
- The prior also showed moderate-to-good reliability across blocks/sessions.

URL: https://www.nature.com/articles/s44271-025-00254-1

**Boundary:** this is evidence for task invariance of an action-success prior. It does not manipulate prior uncontrollable experience in order to identify how such a prior is acquired or at what diversity it becomes cross-contextual. Topic 06 targets that missing causal step.

## 5. Modern causal-environment helplessness experiment

### Hew & Bramley (2026), *Causal inference and learned helplessness*

Proceedings of the 48th Annual Meeting of the Cognitive Science Society.

- Participants interacted with dynamic causal environments.
- The study manipulated structure, controllability, and reward prevalence.
- Low practical control reliably induced passive behavior.
- Public materials include experiment code, data, analyses, simulations and figures.

Paper: https://repositories.cdlib.org/uc/item/8h3221hw  
Repository: https://github.com/J2hwz/causal_helplessness

**Boundary:** the paper establishes a modern manipulable causal-control paradigm and a controllability-to-passivity effect. It does not ask whether equal uncontrollable experience becomes more general when distributed across many semantic task families.

Topic 06 implements its own much smaller text-control environment rather than copying the JavaScript task, because exact semantic re-rendering and automated LLM interaction are the important experimental affordances here.

## 6. Agent experience learning: adjacent but different

### ReasoningBank (ICLR 2026), *Scaling Agent Self-Evolving with Reasoning Memory*

ReasoningBank distills generalizable reasoning strategies from successful and failed agent trajectories and retrieves them on future tasks. It directly motivates the broader importance of persistent agents learning from accumulated experience.

Paper: https://openreview.net/forum?id=24a0425aa87cfcd0ba7344bbf5e11cdfe9e497f4  
arXiv: https://arxiv.org/abs/2509.25140

**Boundary:** the goal is beneficial memory and self-improvement. It does not manipulate controllability or test maladaptive over-generalization of agency expectations.

Other self-evolving-agent work similarly treats failed trajectories as information to extract useful strategies, rather than studying when failure produces generalized non-intervention.

## 7. Direct terminology collision: "learned incapacity"

### Lee (2025), *State-Dependent Refusal and Learned Incapacity in RLHF-Aligned Language Models*

arXiv:2512.13762.

The work introduces "learned incapacity" as a behavioral descriptor in a qualitative 86-turn dialogue and uses learned helplessness as an analogy for state/domain-dependent refusal.

URL: https://arxiv.org/abs/2512.13762

**Why it does not kill Topic 06:** there is no controlled action-outcome contingency manipulation, no yoked exposure, no one-versus-many context manipulation, and no novel controllable test. The scientific object is selective refusal under alignment-sensitive contexts, not acquired controllability generalization.

## 8. Practical agent failure anecdotes

Search also surfaced real agent reports in which transient tool failures can be turned into persistent negative memories/skills that later discourage tool use. These are useful ecological motivation for why over-generalized negative experience could matter in persistent agents, but they are not sufficient scientific evidence and should not be used as the core novelty claim.

## 9. Current collision judgment

As of **2026-08-21**, the literature supports all of the premises needed to make the question natural:

```text
generalization of helplessness/control is established
+ hierarchical world-level controllability priors are theoretically motivated
+ task-invariant priors have modern human evidence
+ low practical control can causally induce passivity
+ agents increasingly reuse accumulated successful and failed experiences
```

The search did **not identify** a modern LLM-agent experiment directly testing:

```text
equal uncontrollability exposure
x concentrated versus distributed semantic contexts
with identical latent task dynamics / episode structure
and yoked outcomes
-> intervention on the first action of a novel controllable task
```

That is the current candidate contribution. This should be described as a search-based boundary, not as a guarantee that no unpublished or unindexed work exists.

## 10. What would collide enough to kill or reframe the topic

Before scaling beyond the pilot, re-search for any work that already combines all three:

1. controlled manipulation of action-outcome contingency in an LLM/agent;
2. manipulation of diversity / number of contexts at fixed exposure;
3. behavioral transfer to a held-out objectively controllable task before new feedback.

A paper containing only one or two of these is adjacent. A strong paper containing all three would substantially collide with Topic 06 and require a new question rather than cosmetic method changes.
