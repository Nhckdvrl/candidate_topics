# Literature audit — 2026-08-22

## Verdict

I did not find a paper that directly performs the identification test proposed here:

> **hold the physical state fixed, vary only a closely related policy/checkpoint, and ask whether the model's internal success signal tracks the success of that specific policy rather than the shared difficulty of the state.**

Several 2026 papers make this question timely, but they stop one step earlier.

## 1. What Frozen VLAs Already Know About Success

**Zhang et al., 2026, arXiv:2605.28527**  
https://arxiv.org/abs/2605.28527

Key result: frozen OpenVLA and pi0.5 representations predict Monte-Carlo success/value-like targets; pi0.5 retains roughly 92% pairwise ordering under same-task / same-timestep matching. The same probe can improve candidate action selection on hard LIBERO tasks.

Most important observation for this topic: **DINOv2 and CLIP also carry comparably strong outcome information.** That is evidence that a large fraction of the signal may be visible in the scene itself rather than being a representation of the policy's own competence.

What it does not identify: for one fixed physical state where two policies have different outcomes, whether each VLA representation preferentially predicts **its own** outcome.

## 2. VLAConf

**Huang et al., 2026, arXiv:2605.29605**  
https://arxiv.org/abs/2605.29605

VLAConf estimates task-success confidence from frozen internal VLA representations and evaluates OpenVLA-OFT and pi0.5 on LIBERO, including shifted suites and real-robot trials.

This establishes that internal representations can support calibrated success prediction. Its evaluation is still fundamentally within-policy: a score is judged by how well it predicts that policy's rollouts. That does not separate generic state difficulty from policy-specific competence.

## 3. FabriMAE / LIBERO-Reflect

**Aniri et al., 2026, arXiv:2608.16697**  
https://arxiv.org/abs/2608.16697

FabriMAE is explicitly framed as VLA *self-evaluation*. It uses internal visual-modality attention entropy to estimate action-generation reliability across heterogeneous VLA architectures, and introduces LIBERO-Reflect with 4,000 episodes.

This is a very direct motivation and a novelty risk. However, the reported question remains “does this policy's score distinguish its successful and failed episodes?” It does not hold the physical state fixed while the successful policy identity flips. Therefore state difficulty remains a plausible shared component of the signal.

Because this paper is only days old, collision risk is high and should be rechecked before any large confirmation run.

## 4. Foresight

**Zhang et al., 2026, arXiv:2606.23085**  
https://arxiv.org/abs/2606.23085  
https://haoranzhangumich.github.io/Forsight_web/

Foresight trains failure detectors on action-conditioned world-model latents and explicitly evaluates cross-policy transfer. Some transfers are very strong, and the authors interpret transfer as evidence for execution-level failure cues rather than policy-specific artifacts.

This is not a collision: Foresight wants policy-agnostic failure detection. But it is a useful warning that success/failure signals can be dominated by shared execution-state cues.

## 5. COAST: the experimental enabler

**Miao et al., 2026, arXiv:2605.17144**  
https://arxiv.org/abs/2605.17144

COAST studies success/failure structure in VLA hidden states and activation steering. For pi0.5 on LIBERO it reports a same-family fine-tuning path and releases checkpoints at **2,000, 3,000, and 9,000** steps; the 2k checkpoint is chosen because it is the earliest checkpoint with non-zero success across most LIBERO-10 tasks. The appendix also reports success/failure overlap around layer 11.

Public checkpoint mirrors found during the audit include:

- `brandonyang/openpi-libero-2000`
- `brandonyang/openpi-libero-3000`
- `brandonyang/openpi-libero-9000`

These are unusually useful here because they provide changing competence while holding architecture, data family, and training recipe much more nearly fixed than a cross-architecture comparison.

COAST itself asks whether success/failure subspaces can be identified and steered. It does not ask whether the success signal is checkpoint-specific under identical physical states.

## 6. LIBERO-Plus

**Fei et al., CVPR 2026, LIBERO-Plus**  
https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html

LIBERO-Plus shows very large differences in success and robustness across VLA families and perturbation types. This confirms that “policy competence” is not a single monotone property of the state distribution.

But using different architectures as the primary test would be a bad design for this question: architecture identity, inference algorithm, representation space, and competence all change together. It is motivation only, not the first experiment.

## Novelty boundary

Do **not** claim:

- VLAs encode success information — already shown;
- internal entropy can predict failures — already shown;
- failure detectors transfer across policies — already shown;
- different VLAs have different robustness profiles — already shown.

The narrow claim under test is:

> **VLA hidden states contain policy-specific success information that changes with which same-family checkpoint will succeed from the identical state, beyond generic state difficulty.**

If this cannot be demonstrated with one paired contrast, the topic should be stopped rather than expanded with controls.
