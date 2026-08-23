# Phenomenon Mining — Source Ledger

This ledger records **direct observations**, not our preferred explanations.

## A. LLM RL / post-training dynamics

### A1. Sparse token-level learning signal in reasoning RL

- **Source:** Chen, Li, Zou, *Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics Through Pattern Selection*, ICLR 2026.
- URL: https://openreview.net/pdf?id=2OO399hRD6
- Direct observation: reasoning-pattern success rates can remain relatively stable while RL changes the **distribution of reasoning patterns**, with optimization concentrated on a sparse subset of critical tokens.
- Mining value: high. The direct object is a checkpointed distributional shift in traces, not a speculative hidden mechanism.
- Collision note: their paper already explains part of this with pattern selection, so a project cannot simply rediscover sparse-token optimization. We should mine adjacent **phase changes, reversals, and capability redistribution** visible in released runs.

### A2. Entropy collapse is structured, not just a scalar drop

- **Source:** Xu et al., *Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization*, arXiv 2605.11491.
- URL: https://arxiv.org/abs/2605.11491
- Direct observation: entropy-decreasing tokens consistently outweigh entropy-increasing tokens during RLVR, producing a strongly imbalanced token-level entropy flow.
- Mining value: medium/high, but crowded. Generic entropy collapse is already saturated. Only a stronger trace-level or capability-level phenomenon should survive.

### A3. Reward-seeking rises across RL checkpoints

- **Source:** Højmark et al., *Measuring Reward-Seeking via Contrastive Belief Updates*, arXiv 2607.18966.
- URL: https://arxiv.org/abs/2607.18966
- Direct observation: on intermediate checkpoints of a capabilities-focused o3 RL run, sensitivity to grader preferences increases substantially over training; one example moves from a much weaker early preference to 87% promise-breaking under a grader-completion belief at a later checkpoint.
- Mining value: high conceptually. This is a direct, monotonic behavioral change across RL training, though reproduction may be blocked by unavailable proprietary checkpoints. Use it as an anomaly shape, not as an immediately runnable target.

### A4. SFT / post-training generalization can be non-monotonic

- **Source:** 2026 survey evidence summarizing multiple studies where OOD performance decreases then recovers, or improves then declines depending on setup.
- URL: https://openreview.net/pdf/c4b2d1e7cf249bdb9df809c980abb6d469d361b4.pdf
- Direct observation: additional SFT is not behaviorally monotonic; early and late checkpoints can trade off differently for OOD reasoning and downstream RL initialization.
- Mining value: high as an anomaly family, but generic "more training hurts OOD" is crowded. We need a more distinctive behavioral transition visible in traces.

### A5. Peak-to-final degradation in agentic RL

- **Source:** He et al., *Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy*, arXiv 2605.14558.
- URL: https://arxiv.org/abs/2605.14558
- Direct observation: vanilla PPO/GRPO agentic training can exhibit substantial **peak-to-final degradation**; their intervention improves training stability.
- Mining value: very high because the phenomenon is already present in open agent environments and is trajectory/checkpoint observable. The paper's own explanation focuses on action-token signal allocation, but the broader question "what behavior disappears after the peak?" is not identical to ActFocus.

### A6. Reasoning length has regime-dependent non-monotonic effects

- **Source:** Nohara, Nakamura, Yokota, *On the Optimal Reasoning Length for RL-Trained Language Models*, 2026.
- URL: https://www.alphaxiv.org/overview/2602.09591
- Direct observation: stronger base reasoners can peak at intermediate reasoning lengths while weaker models improve monotonically with longer outputs.
- Mining value: medium. The phenomenon is direct, but this exact axis may already be too solved/crowded.

---

## B. Agentic RL / interaction learning

### B1. Action tokens are a tiny fraction of trajectories but dominate useful signal

- **Source:** He et al., ActFocus, arXiv 2605.14558.
- Direct observation: action tokens can be <16% of generated tokens yet action-only frozen-reference energy correlates strongly with rollout reward variance, while full-response and reasoning-only energy can be near noise.
- Mining value: high as a structural anomaly, but **the direct "action bottleneck" claim is already occupied** by ActFocus. Do not clone it.

### B2. Standard agentic RL keeps tool observations in context but masks them from ordinary policy loss

- **Source:** Search-R1-style training implementations and common agentic RL setup; tool/environment observations remain in context while policy optimization is applied to assistant-generated tokens.
- Example implementation explanation: https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/03-search-r1/readme.md
- Direct structural fact: environment feedback influences subsequent actions through context, but observation tokens are not ordinary generated-token policy targets.
- Mining value: high only as a **source of observational questions**, not as a hypothesis by itself.

### B3. Observation supervision improves agentic RL

- **Source:** Li et al., *SOAR: Supervision from Observation for Agentic Reinforcement Learning*, ACL 2026.
- URL: https://aclanthology.org/2026.acl-long.1624/
- Direct observation: adding training signal derived from observation tokens improves general reasoning and deep-research tasks and reduces erroneous/inefficient tool use.
- Mining value: high. This means the broad "observations matter" claim is already done. More promising is to inspect **what exact behavior standard RL fails to acquire from feedback**, and whether that deficit is visible before inventing a new loss.

### B4. Feedback-token optimization is an explicit research direction

- **Source:** *Agentic Distillation*, ICLR 2026 submission.
- URL: https://openreview.net/pdf?id=zyp9QT5Gf1
- Direct observation/design fact: their framework adds optimization from both internal tokens and teacher-feedback tokens, explicitly because ordinary agentic interaction does not train feedback in the same way.
- Mining value: medium/high; supports a real tension but not yet a standalone phenomenon.

---

## C. VLA / robot foundation-policy behavior

### C1. Action chunks reduce closed-loop reactivity

- **Source:** Sendai et al., *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*, arXiv 2509.23224.
- URL: https://arxiv.org/abs/2509.23224
- Direct observation: longer action chunks / inference delay hurt responsiveness; per-step chunk correction yields +23 percentage points on dynamic Kinetix and +7 points on LIBERO Spatial relative to RTC in reported settings, and helps even with zero injected delay at long horizons.
- Mining value: very high. This is a direct control phenomenon with obvious traces and a method lever. The generic "chunks hurt reactivity" statement is already occupied, so mine finer phenomena such as **where in the chunk correction demand spikes, whether stale-action error accumulates predictably, and which failure classes are recoverable**.

### C2. Minor execution deviations compound into failure

- **Source:** Yang et al., *RISE: Self-Improving Robot Policy with Compositional World Model*, arXiv 2602.11075.
- URL: https://arxiv.org/abs/2602.11075
- Direct observation/motivation: contact-rich and dynamic manipulation exhibits compounding deviations and recovery-state distribution shift.
- Mining value: medium. Too broad alone, but useful for recovery-state mining.

### C3. Closed-loop replanning converts failed grasps into recovery

- **Source:** Naouali et al., *VLCP: Vision Language Control Policy Closed-Loop Code Replanning for Robot Manipulation*, arXiv 2608.16978.
- URL: https://arxiv.org/abs/2608.16978
- Direct observation: repeated re-observation/replanning gives 35.1% pooled success vs 3.5% for one-shot control in their setup, with a reported 27.3% within-episode recovery rate on failed grasps.
- Mining value: medium/high. The exact method is separate, but the **recoverable-failure population** is a strong phenomenon object.

### C4. Same VLA weights can execute as a different physical policy under deployment metadata

- **Source:** Tai, *Same Weights, Different Robot: A Deployment Safety View of VLA Policies*, arXiv 2606.03724.
- URL: https://arxiv.org/abs/2606.03724
- Direct observation: plausible metadata mismatch can drastically alter unnormalized actions and collapse replay success despite identical learned weights.
- Mining value: high as a concrete systems/behavior phenomenon, but may be too deployment-engineering-focused for our main lab fit unless expanded into a broader executable-policy question.

---

## D. Current reject/collision list

Do not spend primary search time on:

- generic entropy collapse in RLVR;
- generic "RL improves reasoning";
- generic action-token importance in agentic RL (ActFocus already owns the headline);
- generic observation supervision in agentic RL (SOAR already owns the headline);
- generic VLA action-chunk reactivity degradation (A2C2 already owns the headline);
- generic catastrophic forgetting;
- generic hidden-state probes without a replicated behavior;
- any topic needing a bespoke semantic ontology before the phenomenon can even be named.

The search target is always the **next unresolved direct phenomenon** adjacent to these observations, not a narrower restatement of the same published claim.
