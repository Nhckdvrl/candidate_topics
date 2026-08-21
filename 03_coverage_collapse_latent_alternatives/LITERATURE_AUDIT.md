# Topic 03 — Literature / Collision Audit

Audit date: **2026-08-21**.

This topic is only viable if the claim is kept narrower than several already-published or concurrent results.

## 1. Seed phenomenon: coverage shrinkage and fork commitment already exist

Nguyen et al., **Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road** (COLM 2026; arXiv:2605.17026) already establish the central behavioral phenomenon used here:

- SFT can improve `pass@1` while reducing larger-`k` coverage;
- the controlled Graph Branching task localizes an indecipherable first-step decision point;
- confidence at that fork polarizes during training, including confidently incorrect branch choices;
- semantic-preserving dependency-order perturbations can change branch selection, indicating reliance on spurious cues;
- prefix/diversity interventions can recover part of lost coverage.

Therefore this project cannot claim that coverage shrinks, that decision points become overconfident, or that alternative trajectories can be behaviorally reactivated.

A reproduction detail matters: Appendix A.2 reports 16 epochs at `lr=2e-5`; the pinned public shell launcher currently uses `1e-5` for Qwen2.5-0.5B. The primary reproduction in this repository uses the paper value.

## 2. Major direct collision: intermediate layers already retain exploration signal

Tan et al., **Restoring Exploration after Post-Training: Latent Exploration Decoding for Large Reasoning Models** (ICML 2026; arXiv:2602.01698) is the closest collision found in this audit.

They show that RL-post-trained reasoning models can have a sharply concentrated final-layer posterior while intermediate-layer posteriors retain substantially more entropy. Their Latent Exploration Decoding (LED) feeds intermediate hidden states through the model readout, aggregates latent posteriors, and improves `pass@1` / `pass@16` without additional training.

Consequences for Topic 03:

- **“intermediate layers retain exploration after post-training” is already taken;**
- **“the final layer collapses while latent layers remain diverse” is already taken;**
- simply plotting layer-wise entropy or applying an early-exit/logit-lens decoder is not a sufficient contribution;
- if Topic 03 survives G0, an LED/logit-lens-style intermediate-posterior baseline becomes mandatory in the full study.

The residual question here is more specific: does the latent signal identify the **globally viable concrete branch** at a known SFT coverage-shrinking fork, under matched graph counterfactuals, and does it beat the final native commitment on a label-free disagreement set?

## 3. Alternative-path information in hidden states is already known

Zur et al., **Are Language Models Aware of the Road Not Taken? Token-level Uncertainty and Hidden State Dynamics** (arXiv:2511.04527, 2025) show that hidden activations can predict future outcome distributions and contain information about alternative reasoning paths.

Therefore `hidden-state probe AUC > 0.5` is not sufficient novelty.

Topic 03 must tie the signal to:

1. exact graph-ground-truth branch viability;
2. a known coverage-shrinking SFT trajectory;
3. matched target interventions;
4. cross-checkpoint transfer / frozen readout;
5. a final-vs-latent disagreement test selected without correctness labels.

## 4. Forced branch viability is also not new

Liu et al., **When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning** (arXiv:2605.21606, 2026) use forced alternative tokens and continuation success to diagnose branch/token reliability.

Therefore forcing the suppressed first token and showing that the rest of the trajectory can succeed is a useful auxiliary measurement, not the main contribution.

## 5. Generic representation-collapse analyses are crowded

2026 work on sequential post-training representation collapse, collapse-aware regularization, semantic/mode collapse, and hidden-mechanism reactivation means that generic analyses such as:

```text
effective rank
CKA
anisotropy
mean hidden-state distance
layer entropy alone
```

are not enough for this topic.

The experiment should remain branch-ground-truth and decision-point-specific.

## 6. Probe methodology warning

Sahoo et al., **Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States** (TrustNLP 2026; arXiv:2606.02907) demonstrate that very high probe accuracy can be entirely explained by format confounds. Older probing-control work makes the related point that decodability alone does not establish functional use.

This motivates the current controls:

- same graph with only the query target switched to the opposite terminal;
- target-blind control;
- discovery/confirmation separation;
- behavior-gate IDs excluded from latent analysis;
- one reference-trained frozen probe rather than checkpoint-local probe refits;
- reference checkpoint candidate basis frozen across checkpoints;
- label-free final/probe disagreement selection.

If the target-blind control is positive or the score does not move under the target flip, the probe should be treated as shortcut-prone rather than interpreted mechanistically.

## 7. Remaining defensible claim

After this audit, the strongest remaining claim is approximately:

> During a controlled SFT trajectory that demonstrably shrinks sampled coverage at a binary reasoning fork, a branch-ground-truth viability signal learned before collapse may remain accessible after the final native readout becomes strongly committed. This signal must track which terminal is queried under a matched counterfactual, transfer under a frozen readout, survive shortcut controls, and outperform the final readout on high-confidence disagreements selected without labels.

Even a positive G0 does not yet establish a paper. Full confirmation should add an LED/logit-lens intermediate-posterior baseline, a second seed/backbone, and ideally a causal or behaviorally actionable intervention showing that the identified viability signal can improve branch selection rather than merely being decoded.

## 8. Fast decision implication

The current G0 remains useful precisely because it can terminate the topic before those expensive additions:

- no paper-exact behavior collapse -> stop;
- no counterfactually valid reference signal -> stop;
- no frozen late transfer -> stop/redesign;
- target-blind shortcut -> stop/redesign;
- too few label-free latent/final disagreements -> stop this mechanism;
- latent readout does not win those disagreements -> stop.

Only after all of those pass is it worth implementing the mandatory LED/logit-lens and causal follow-up controls.
