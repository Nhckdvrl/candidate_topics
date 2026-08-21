# Literature collision notes

This folder tests a narrower claim than “diversity is not uncertainty.” That broad statement already has close prior art.

## Motor-control seed

The uncontrolled-manifold / goal-equivalent-manifold literature separates movement variability that changes a task variable from variability that lies along redundant directions. Skilled behavior need not minimize total motor variability; it can preferentially stabilize task-relevant variables while retaining variability in goal-equivalent directions.

## Generative robot-policy collision

**Diff-DAgger (ICRA 2025)** already argues that action disagreement in multimodal generative policies can be benign: multiple behavior modes can all be valid. Therefore the novelty cannot simply be “multiple sampled actions may all be correct.”

The distinction tested here is continuous/local **functional redundancy relative to a task manifold**, not only discrete behavior-mode ambiguity.

## Scalar action entropy collision

**FIPER (NeurIPS 2025)** samples batches of action chunks from a generative imitation policy and computes Action-Chunk Entropy (ACE) as a failure-prediction signal. Its appendix defines a fixed-range, joint-cell histogram estimator and sums entropy across chunk timesteps. The implementation here reproduces that estimator structure for the pilot.

FIPER also discusses converting actions to Cartesian end-effector coordinates for task relevance. Consequently, a result based only on joint kinematic redundancy is insufficient for the broad claim. The required follow-up uses a Cartesian action space whose task constrains only a lower-dimensional functional variable.

## Current novelty target

The question is therefore:

> Does the sampled action distribution of a generative robot policy have a task-relative geometry such that scalar entropy conflates variability normal to the task-equivalent manifold with variability tangent to it?

A convincing result must show more than high null variance. It should show a **matched scalar-entropy counterexample** in which differing task-sensitive variance predicts differing functional risk.

## References

- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, RSS 2023. https://diffusion-policy.cs.columbia.edu/
- Failure Prediction at Runtime for Generative Robot Policies (FIPER), NeurIPS 2025. https://arxiv.org/abs/2510.09459
- FIPER code. https://github.com/utiasDSL/fiper
- ManiSkill controller documentation (for the planned Cartesian replication). https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html
