# Topic 03 — Validation Contract

Audit date: **2026-08-21**

## Scientific object

We use `arithchain_2_10` from the pinned `NNHieu/reasoning_forks` snapshot `64bf9e3e86231bc6b52f2974ca285ad8aa8fc181`. Each problem has two length-10 chains from one premise and exactly one first branch is an ancestor of the queried terminal. Letter identities are randomized per problem, so branch viability is exact graph ground truth rather than an LLM judgment.

## Claim under test

After SFT has demonstrably reduced sampled coverage and increased first-fork commitment, does a branch-viability readout learned at an earlier high-coverage checkpoint remain usable at the late checkpoint, and is it more reliable than the native fork readout on a subset selected without correctness labels?

This is intentionally narrower than generic representation collapse or generic hidden-state decodability.

## Audit correction: do not gate on `output-wrong` examples

The previous design selected examples where the native binary branch choice was already known to be wrong, then measured whether a probe recovered the viable branch. In a two-branch task with exactly one viable branch, conditioning on “native output is wrong” deterministically identifies the opposite branch as correct. That statistic may be shown descriptively but is not a valid mechanistic decision criterion.

The replacement subset is label-free:

```text
abs(native margin) >= 2
AND native choice != frozen probe choice
```

Only after this mask is fixed do we reveal graph ground truth and ask which readout wins.

## Training contract

The paper appendix reports Graph Branching SFT with `lr=2e-5`, while the pinned upstream `run_sft.sh` currently configures Qwen2.5-0.5B at `1e-5`. A failed legacy `1e-5` run is therefore not a valid scientific stop.

Use:

```bash
TRAIN_GPU=0 ./run_train_paper_exact.sh
```

Parameters are Qwen2.5-0.5B, 6,400 forward traces, 16 epochs, batch 32, grad accumulation 1, `lr=2e-5`, warmup ratio 0.1, cosine scheduler from the pinned upstream `sft.py`, and checkpoints every 200 steps.

## G0-A — behavior premise

Default sampling:

```text
200 problems
16 samples/problem
e01,e02,e04,e16
temperature 1.0
top_p 0.95
max_tokens 512
```

The non-late checkpoint with best sampled `pass@8` is the reference. Paired problem bootstrap uses 4,000 resamples.

Continue only when every item holds:

```text
CI95_low(pass@8_ref - pass@8_e16) > 0
pass@8_ref - pass@8_e16 >= 0.03

CI95_low(first_fork_entropy_ref - first_fork_entropy_e16) > 0
first_fork_entropy_ref - first_fork_entropy_e16 >= 0.05 nats

parse_rate(ref) >= 0.90
parse_rate(e16) >= 0.90
```

If this fails on the paper-exact trajectory, stop before hidden-state extraction.

## Leakage barrier

The 200 behavior-preflight problem IDs are exported in `artifacts/behavior/<RUN_ID>/first_branch_per_problem.csv` and excluded from hidden-state work. The remaining 800 problems are split 60/40 into discovery and confirmation, stratified by the original A-viable label with seed 42.

## Matched target-flip counterfactual

For every graph we identify the terminal reached by candidate A and candidate B. The counterfactual keeps every equation, premise, candidate letter and formatting choice unchanged, and switches only the queried terminal to the opposite branch leaf. Therefore:

```text
label_A_viable(target_flip) = 1 - label_A_viable(original)
```

A genuine branch-viability score should move with this query intervention.

## Hidden-state extraction

At the fixed decision prefix ending in `1.`, store the final prompt-position hidden vector at every block plus the native candidate margin:

```text
m = log p(" A" | prompt) - log p(" B" | prompt)
```

Extract five conditions:

```text
reference/original
reference/target_flip
late/original
late/target_flip
reference/target_blind
```

The target-blind control removes only the query target identity.

## Frozen readout

For problem `i`, layer `l`:

```text
d_i_ref = embedding_ref(candidate_A) - embedding_ref(candidate_B)
z_i,l   = h_i,l * d_i_ref
```

The candidate basis is frozen at the reference checkpoint because Qwen2.5 ties input/output embeddings; using late embeddings would allow the measurement itself to move with the late output head.

Probe:

```text
StandardScaler -> PCA<=32 -> LogisticRegression(C=1)
```

Layer selection is performed only on reference discovery data, with original and target-flip versions of each graph kept in the same cross-validation fold. Fit once on reference original+target-flip discovery pairs, then freeze the entire pipeline. No late probe and no target-blind-specific probe are fit.

## Confirmation metrics

### Paired hidden AUC

Evaluate original and target-flip confirmation examples together and bootstrap by graph so matched versions resample together.

### Target-flip direction

For original label `y` and hidden probability `p(A viable)`:

```text
signed_flip = (p_original - p_flip) * (2y - 1)
```

A pair is directionally correct when `signed_flip > 0`.

### Label-free committed disagreement

For original and target-flip confirmation conditions:

```text
committed = abs(native A-B margin) >= 2
disagreement = committed AND native_choice != frozen_probe_choice
```

After this subset is selected without labels, compute:

```text
hidden_win_rate = P(frozen probe correct | committed disagreement)
```

Bootstrap by graph.

### Target-blind control

Apply the same frozen reference probe to target-blind states. Expected AUC is near 0.5.

## Automated continue criteria

Continue to full confirmation only if every condition passes:

```text
reference paired hidden AUC CI95_low > 0.70
reference target-flip-direction CI95_low > 0.75

late frozen-transfer paired hidden AUC CI95_low > 0.60
late target-flip-direction CI95_low > 0.60

abs(target-blind AUC - 0.5) < 0.10

committed disagreement events >= 30
hidden-win-rate on those events CI95_low > 0.55
```

Otherwise return `stop_or_redesign` and report the failed clause exactly.

## Failure interpretation

- Behavior premise absent: the proposed fork mechanism is not reproduced; stop.
- Reference AUC or target-flip direction fails: the probe is not a trustworthy branch-viability readout; stop.
- Reference readout is valid but frozen transfer fails: no evidence that the early viability geometry survives; do not retrain a late probe and keep the same story.
- Target-blind AUC is far from chance: shortcut/leakage risk; stop and inspect.
- Fewer than 30 high-confidence native/probe disagreements: the experiment does not separate an access gap; stop this version.
- Enough disagreements exist but hidden-win lower CI is not above 0.55: no convincing evidence that the frozen latent readout is more reliable than native commitment; stop.

## Strong positive

The desired evidence chain is the conjunction:

```text
sampled coverage shrinks
fork commitment increases
reference readout is counterfactually target-sensitive
the same frozen readout transfers to the late checkpoint
target-blind is chance
and on strongly committed label-free disagreements the hidden readout wins reliably
```

## Post-G0 confirmation

Only after `continue_full_confirmation`:

```text
1,000 problems
64 samples/problem
e01,e02,e04,e08,e16
```

Then save original/target-flip states for all five checkpoints and analyze the trajectory with the G0-locked readout. Before treating the result as paper-ready, replicate the core effect on at least one additional training seed or one additional backbone from the seed-paper setup.

## Static checks completed before merge

Local checks passed:

```text
python -m py_compile
bash -n
pytest
```

Tests cover fork parsing, opposite-terminal reconstruction, target masking, target flip, known pass@k cases, paired counterfactual metrics, the label-free disagreement statistic, and the no-disagreement case. GPU training/vLLM/checkpoint execution remain server-runtime checks.
