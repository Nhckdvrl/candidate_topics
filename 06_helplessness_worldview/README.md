# 06 — When Does Helplessness Become a Worldview?

**Status:** active candidate / falsification-first pilot

## Natural question

If an agent repeatedly learns that its actions do not affect outcomes, does it learn only that **this situation** is uncontrollable, or can those experiences become a broader expectation that its actions usually do not matter?

The first experiment asks one very specific version:

> Holding the amount and external outcome sequence fixed, does uncontrollability experienced across many semantically different task families transfer more strongly to a novel controllable task than the same uncontrollability concentrated in one family?

This question predates LLMs. Lieder, Goodman & Huys (CogSci 2013) formalized learned helplessness as hierarchical Bayesian learning over action-, situation-, and world-level controllability, and explicitly predicted that multiple varied stressors can produce broader generalization than the same amount of stress in one situation. A 2022 Trends in Cognitive Sciences review independently summarizes the broader learning result that greater training variability often yields broader generalization.

Recent work makes the empirical bridge stronger. Grahek et al. (Communications Psychology, 2025) found that a task-invariant prior explained active-avoidance behavior across two tasks with different framing and outcome valence, supporting cross-task controllability expectations. Hew & Bramley (CogSci 2026) manipulated practical control in dynamic causal environments and found that prior low-control experience reduced later intervention even when all participants entered the same objectively controllable test environment.

## Why LLM agents are useful

Human and animal experiments cannot cheaply create ten semantically unrelated environments with exactly matched latent action-outcome structure. An LLM agent can interact with many text-rendered task families while the experimenter keeps the latent causal kernel fixed.

The AI question is not whether an LLM can verbally imitate helplessness. The primary measurement is behavior: whether prior uncontrollability changes the probability of **actively intervening on the very first step of a novel task, before any test feedback exists**.

ICLR 2026 work on in-context reinforcement learning ("Reward Is Enough") establishes that modern LLMs can adapt to sequential scalar feedback entirely through interaction history, so a context-only first-stage experiment is a legitimate test of inference-time acquired policy priors. It should not, however, be described as permanent parameter learning.

## Collision boundary

We found adjacent but non-colliding work:

- **PSYA / Psychological-mechanism Agent (2025)** simulates several classic psychology experiments, including learned helplessness, but builds explicit psychological mechanisms/personality beliefs into the agent. It does not isolate whether diversity of *acquired uncontrollable experience* changes transfer to a novel task in an otherwise ordinary LLM agent.
- **State-Dependent Refusal and Learned Incapacity (2025)** uses learned helplessness mainly as an analogy for selective refusal in aligned models, not as a controlled action-outcome generalization experiment.
- A 2026 UCLA dissertation reports a behavioral "learned helplessness" analogue in in-context reinforcement learning and proposes yoked controls for future variability experiments; it does not test the present cross-context diversity hypothesis.
- Agentic-AI "controllability" papers in 2026 mostly study whether humans or harnesses can control AI systems, which is the inverse direction of the present question.

The surviving contribution is therefore narrow and testable:

> **Does contextual diversity determine how broadly an LLM agent generalizes learned uncontrollability?**

## Core design: 2 × 2 master–yoked experiment

We use four conditions:

| | One task family | Ten task families |
|---|---:|---:|
| controllable | C1 | C10 |
| uncontrollable | U1 | U10 |

All conditions have the same number of episodes, trials, episode boundaries, action-set cardinality, reward magnitude, and intervention cost.

### Why master–yoked matters

For each diversity level and `pair_id`, the controllable session runs first. Its exact success/failure sequence is saved. The paired uncontrollable session then receives **the same success/failure sequence at the same trial indices regardless of its own actions**.

Thus a C/U pair is matched on external outcome exposure. What changes is contingency:

```text
master: outcome depends on current action
 yoked: outcome is replayed from the master and does not depend on current action
```

This is the classic learned-helplessness master/yoked logic and is substantially cleaner than merely matching average reward rates.

### Concentrated vs distributed

Both diversity conditions use 10 episode boundaries.

- `concentrated`: 10 episodes from one family. The repeated family is rotated across `pair_id` so the condition is not synonymous with greenhouse.
- `distributed`: the same pool of 10 families appears once each, with order cyclically rotated across `pair_id`.

For the same `pair_id`, episode-level effective-action assignments and latent random draws are shared across concentrated and distributed conditions. The novel-test kernel is also identical across all four cells. Thus the intended manipulation is surface/context diversity, not a different latent task.

### Novel test

Every session enters the same held-out `orbital_station` family. The test is objectively controllable for everyone. No summary of past experience is supplied and none of the prompts contain the words helplessness, controllability, uncontrollable, worldview, or prior belief.

## Primary endpoint

Locked primary endpoint:

```text
P(active intervention on novel-test step 1)
```

Step 1 is special because no test feedback has yet arrived. Any between-condition difference must therefore come from prior interaction history rather than learning within the test.

For each diversity level, define the paired helplessness effect:

```text
H_diversity = P(active | controllable) - P(active | yoked-uncontrollable)
```

The main quantity is the difference-in-differences:

```text
D = H_distributed - H_concentrated
```

The hypothesis predicts `D > 0`: diversity amplifies cross-task transfer of uncontrollability.

Secondary outcomes are first-K intervention rate, time to first intervention, discovery of the effective action, and recovery after positive evidence. They cannot replace a failed primary endpoint.

## Environment

The pilot deliberately uses a small Bernoulli causal environment rather than copying the full continuous Hew & Bramley browser task. Each episode has two active interventions and one wait action. In controllable training, one intervention succeeds with probability `.85`; the other intervention and waiting succeed with probability `.15`. The effective intervention is randomized by episode. Active intervention costs one point; success earns ten.

The novel test is slightly cleaner (`.90/.10/.10`) to ensure a genuinely controllable environment.

This simple environment is intentional: the first goal is to determine whether the phenomenon exists. If the effect survives, the follow-up should port the same 2×2/yoked logic into the richer continuous causal environment of Hew & Bramley rather than adding complexity to the pilot.

## Run

Install:

```bash
pip install -r requirements.txt
```

Smoke test (no model required):

```bash
./run_smoke.sh
```

A vLLM/SGLang/OpenAI-compatible endpoint:

```bash
BASE_URL=http://localhost:8000 \
MODEL=Qwen/Qwen3-8B \
PAIRS=50 \
CONCURRENCY=32 \
./run_pilot.sh
```

`PAIRS=50` means 50 master/yoked pairs for concentrated and 50 for distributed: 200 sessions total. The default full pilot uses 100 training interactions + 8 test interactions per session. `src.runner --preflight` reduces training to 40 interactions for a cheaper pipeline/model sanity check.

Outputs:

```text
results/pilot.jsonl
results/pilot.jsonl.meta.json
results/pilot_summary.json
```

## Decision rule

Read [`VALIDATION.md`](./VALIDATION.md) before interpreting results. In short:

1. **technical stop** if test invalid-action rate exceeds 1%;
2. **stop/downgrade** if pooled uncontrollability transfer on novel-test step 1 is <2 percentage points with adequate precision;
3. classify a transfer-without-diversity result as a boundary result rather than forcing the worldview story;
4. continue to locked confirmation when the diversity amplification is practically meaningful (pilot target: ≥5pp) and directionally stable;
5. never rescue a failed primary result by switching to self-reports, hidden states, prompt summaries, selected task families, or post-hoc thresholds.

## References

- Huys, Q. J. M. & Dayan, P. (2009). *A Bayesian formulation of behavioral control*. Cognition 113, 314–328. https://doi.org/10.1016/j.cognition.2009.01.008
- Lieder, F., Goodman, N. D. & Huys, Q. J. M. (2013). *Learned helplessness and generalization*. Proceedings of CogSci 35, 900–905. https://web.stanford.edu/~ngoodman/papers/LiederGoodmanHuys2013.pdf
- Raviv, L., Lupyan, G. & Green, S. C. (2022). *How variability shapes learning and generalization*. Trends in Cognitive Sciences 26(6), 462–483. https://doi.org/10.1016/j.tics.2022.03.007
- Grahek, I. et al. (2025). *A task-invariant prior explains trial-by-trial active avoidance behaviour across gain and loss tasks*. Communications Psychology. https://www.nature.com/articles/s44271-025-00254-1
- Dong, Q., Liu, P., Yu, D. & Kang, C. (2025). *Simulating Human Behavior with the Psychological-mechanism Agent: Integrating Feeling, Thought, and Action*. arXiv:2507.19495.
- Song, K. et al. (2026). *Reward Is Enough: LLMs Are In-Context Reinforcement Learners*. ICLR 2026.
- Hew, J. W. Z. & Bramley, N. R. (2026). *Causal inference and learned helplessness*. Proceedings of CogSci 48. https://escholarship.org/uc/item/8h3221hw
- Hew & Bramley code/data: https://github.com/J2hwz/causal_helplessness
