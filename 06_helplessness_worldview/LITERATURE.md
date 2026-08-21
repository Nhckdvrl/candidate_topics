# Literature audit — Topic 06

Audit date: **2026-08-21**.

## Direct theoretical basis

### Huys & Dayan (2009), *A Bayesian formulation of behavioral control*

Formalizes helplessness/behavioral control as beliefs about action-dependent outcome distributions. Establishes the normative Bayesian framing used by later generalization accounts.

### Lieder, Goodman & Huys (2013), *Learned helplessness and generalization*

The closest theoretical seed. The model learns action-outcome contingency at three abstraction levels and treats cross-situation generalization as central to helplessness. Importantly, the paper explicitly predicts that, because variability strengthens generalization, multiple stressors should make helplessness more general than an equivalent amount of stress in one situation. This is almost exactly the concentrated-vs-distributed manipulation proposed here, but it was not an LLM-agent experiment and did not provide the modern semantically controlled cross-task test.

### Raviv, Lupyan & Green (2022), *How variability shapes learning and generalization*

Broad review across domains: more variable training inputs often slow local acquisition but produce broader generalization. This supports the manipulation as a general learning question rather than a helplessness-specific trick.

## Modern empirical bridge

### Grahek et al. (2025), Communications Psychology

Across two active-avoidance tasks with different surface framing and outcome valence, a shared task-invariant prior fit behavior better than separate task priors. This supports the existence of cross-task expectations about action success, but the study did not experimentally manipulate how such a prior is acquired. Its conclusion explicitly points to experimental manipulation of controllability as future work.

### Hew & Bramley (2026), CogSci

Participants interacted with dynamic causal systems under manipulated practical control. Low control produced more passive behavior, and the later test used an objectively controllable environment. The authors released experiment code, data, analyses, simulations, and preregistration material at `J2hwz/causal_helplessness`.

This paper validates the core transfer phenotype but does not manipulate cross-context diversity.

## LLM / agent adjacency

### Song et al. (ICLR 2026), *Reward Is Enough: LLMs Are In-Context Reinforcement Learners*

Shows sequential response/reward histories can induce inference-time policy improvement in LLMs. This makes interaction-history-only acquisition a meaningful first experimental layer, though it must be described as in-context adaptation rather than permanent learning.

### Dong et al. (2025), PSYA

A psychologically structured generative-agent framework that replicates five classic psychology experiments. Its appendix includes a learned-helplessness simulation. The agent setup includes psychological modules and assigned beliefs/traits; it is evidence that the phenomenon has reached LLM-agent simulation, but it does not isolate spontaneous higher-order controllability generalization under a master–yoked, diversity-controlled history.

### State-Dependent Refusal and Learned Incapacity (2025)

Uses learned helplessness as a lens for long-horizon selective refusal in RLHF-aligned models. It is behaviorally adjacent but the causal variable is policy-sensitive context/refusal, not action-outcome contingency across task families.

### 2026 UCLA dissertation on in-context behavioral learning

Reports a "learned helplessness" behavioral analogue where an LLM can lock into an incorrect response despite corrective feedback, and suggests matched yoked control as a future experiment for reinforcement-sensitive variability. Important collision warning, but it does not test the current distributed-vs-concentrated uncontrollability transfer question.

## Novelty boundary after audit

Do **not** claim:

- first learned helplessness in an LLM;
- first LLM simulation of a helplessness experiment;
- first evidence that LLMs adapt to reward histories;
- first cross-task prior in cognition.

The candidate claim is narrower:

> Holding exposure and external outcomes constant with a master–yoked design, does **semantic/contextual diversity of uncontrollable experience** determine the breadth of transfer to a novel controllable task in an ordinary LLM agent?

No directly matching modern LLM-agent paper was found in the searches performed for this registration.
