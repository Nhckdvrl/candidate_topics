# Candidate Research Topics

This repository collects research-topic candidates that are still in the **hypothesis / pilot** stage.

The default workflow for each topic is:

1. identify a strong seed paper or established phenomenon;
2. rotate only one adjacent variable;
3. define a cheap falsification experiment;
4. stop early if the core premise fails;
5. only then expand into a full paper.

## Current candidates

| Priority | Topic | Core question | First falsification test |
| --- | --- | --- | --- |
| 1 | [DLM Trajectory Fate](./02_dlm_trajectory_fate/) | Can hidden states distinguish recoverable/doomed and stable/overwritten states before the surface transition happens? | On 1,000 GSM8K examples, conditional hidden-state probes must beat entropy/confidence baselines with non-trivial lead time. |
| 2 | [Behavior vs. Representation Stabilization](./01_behavior_vs_representation_stabilization/) | After output behavior stabilizes, do internal representations keep reorganizing? | On Pythia-410M, behavioral KL must enter a stable regime while representation drift shows a clear, systematic temporal mismatch. |
| 3 | [Coverage Collapse vs. Latent Alternatives](./03_coverage_collapse_latent_alternatives/) | When a reasoning branch disappears from behavior, is branch-specific viability information erased or merely suppressed? | Before any training-dynamics study, the base/early model must reliably encode viability of unchosen branches at controlled graph forks. |

## Current working order

1. Run the **DLM trajectory-fate** pilot first: it has the cleanest one-step question and the most reusable existing code.
2. Run the **behavior/representation stabilization** pilot next: it has the strongest Training Dynamics ceiling if a real temporal decoupling appears.
3. Treat **latent alternatives during coverage collapse** as high-risk/high-reward: do not invest in checkpoint training unless the branch-specific G0 probe works first.
