# Coverage Collapse vs. Latent Viability of Suppressed Branches

## Background

**Why Do Reasoning Models Lose Coverage?** (Nguyen et al., 2026) shows that supervised reasoning post-training can improve `pass@1` while shrinking `pass@k`. Its controlled `arithchain_2_10` Graph Branching task is especially useful because the first reasoning step is an exact fork: two branches are locally valid from the premise, but only one reaches the queried target. The official generator randomizes node letters per problem, so the globally viable branch cannot be identified from a fixed letter identity.

The seed paper's important mechanistic observation is not simply that the mean probability of the correct branch falls. Training makes decision points increasingly **committed/polarized**, including on examples where the model commits to the wrong branch. Prefix perturbations can recover some lost coverage, which motivates a suppression-style interpretation.

Separately, **Are Language Models Aware of the Road Not Taken?** (Zur et al., 2025) shows that hidden activations can predict future outcome distributions and carry information about alternative reasoning paths.

The one-step question is:

> When SFT makes a model strongly commit to the wrong branch at a coverage-shrinking fork, is the globally viable branch still linearly recoverable from the hidden state?

This is deliberately narrower than generic representation collapse. The claim must be **branch-specific**, tied to exact graph-ground-truth viability, and evaluated along the same SFT trajectory that exhibits coverage shrinkage.

## What we want to do

At the exact first fork, separate three quantities:

1. **Coverage / commitment** — repeated sampling (`pass@k`, first-branch entropy) establishes that an early checkpoint has broader coverage than the late checkpoint.
2. **Normal output access** — teacher-forced candidate log-probability margin shows which branch the ordinary next-token readout prefers.
3. **Latent branch viability** — a low-capacity candidate-conditioned hidden-state probe asks which concrete candidate reaches the target.

The strongest result is not merely `latent AUC > 0.5`. It is the conditional result:

```text
late checkpoint:
normal output readout strongly chooses the wrong branch
but a held-out hidden-state probe still identifies the globally viable branch
```

That is direct evidence that useful branch-specific information remains accessible internally after normal generation has become committed elsewhere.

## Validation: falsification-first G0

The old validation order was too expensive and could falsely kill the topic. The audited pipeline is now staged.

### 0. Prepare the exact upstream snapshot

```bash
./prepare_upstream.sh
```

This pins `NNHieu/reasoning_forks` to the audited commit `64bf9e3e86231bc6b52f2974ca285ad8aa8fc181` and reconstructs exact first-fork labels for the official 1,000-test split.

Train the small official forward SFT once if checkpoints are absent:

```bash
(cd external/reasoning_forks && ./run_sft.sh arithchain_2_10_forward qwen2.5_0.5b 16)
```

### G0-A: cheap behavior premise gate

Run:

```bash
GPUS=0,1,2,3 ./run_g0.sh
```

The first scientific gate samples only 200 test problems, 16 samples/problem, at epochs 1, 2, 4 and 16. It automatically selects the best early checkpoint by sampled coverage and compares it with epoch 16. We continue only if both are reproduced on the same problems:

- sampled coverage (`pass@k`) is lower late;
- first-fork branch entropy is lower late (stronger commitment/polarization).

A paired problem-level bootstrap is used for both differences. If either premise fails, **stop before hidden-state extraction**. A 200-example teacher-forced state preflight is also available, but it is explicitly not allowed to establish coverage shrinkage by itself.

### G0-B: early-vs-late latent gate

Only if G0-A passes, the pipeline extracts hidden states for the behavior-selected early reference checkpoint, epoch 16, and a target-blind negative control at the reference checkpoint.

The probe uses:

```text
de = embedding(candidate_A) - embedding(candidate_B)
z_l = h_l * de
```

but the audit no longer interprets this feature as independent of the output head: Qwen2.5 ties input/output embeddings, so the compatibility geometry is related to ordinary readout geometry.

Instead, the decisive measurement is **hidden rescue on output-wrong examples**. On an independent confirmation split, among examples where the ordinary candidate margin points to the wrong branch, we measure whether the hidden probe still selects the true viable branch.

To avoid layer fishing: split problems 60% discovery / 40% confirmation; select one layer using only the early reference checkpoint's discovery split; lock it; train/evaluate checkpoint-local low-capacity probes at the locked layer; report held-out AUC/accuracy and problem-level bootstrap CIs.

### G0-C: target-blind shortcut control

The query's target letter is removed while the graph rules remain unchanged. With no queried target identity, the two chains are symmetric with respect to which one is globally relevant.

If branch viability remains strongly decodable (`AUC >= 0.60` in the G0 gate), the experiment is likely exploiting a shortcut or leakage and should stop/redesign.

## Decision rule

Continue only when the evidence chain is complete:

```text
coverage shrinks
AND first-fork behavior becomes more committed
AND reference checkpoint contains a robust viability signal
AND late output-wrong cases still contain recoverable latent viability
AND target-blind control is near chance
```

Possible outcomes:

- **Suppression / access failure:** late output is wrong/committed while hidden viability remains recoverable — strongest result.
- **Branch-specific erasure:** hidden viability collapses together with coverage — potentially interesting, but lower priority and must be distinguished from generic representation collapse.
- **No first-fork linkage:** coverage shrinks without first-fork polarization — stop this mechanism; do not force the story.
- **No learned viability signal:** early competent checkpoint has no robust hidden viability signal — stop.
- **Shortcut control positive:** stop/redesign.

## Full confirmation only after G0

```bash
# 1,000 problems x 64 samples x 5 checkpoints
./run_behavior_passk_forward.sh

# save all five checkpoint hidden states only after G0 is alive
./run_sft_dynamics_example.sh
```

Do not use full sweeps to rescue a failed G0.

## Collision boundary — 2026-08-21

- **Why Do Reasoning Models Lose Coverage?** already shows decision-point-driven coverage shrinkage and inference-time recovery through prefix diversification.
- **Are Language Models Aware of the Road Not Taken?** already shows that hidden activations can encode alternative future outcomes in ordinary reasoning.
- **When Are Teacher Tokens Reliable?** already introduces a behavior-level branch-viability diagnostic by forcing alternative tokens and checking successful continuation.
- 2026 work on generic post-training representation collapse means rank/anisotropy/CKA collapse alone is not a contribution here.

The remaining candidate contribution is specifically:

> **branch-ground-truth hidden viability through a known coverage-shrinking SFT trajectory, with direct analysis of late wrong-commitment examples.**

See [`VALIDATION.md`](./VALIDATION.md) for the exact audit, metrics, failure modes and experiment contract.
