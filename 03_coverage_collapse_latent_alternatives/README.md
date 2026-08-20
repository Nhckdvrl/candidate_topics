# Coverage Collapse vs. Latent Viability of Suppressed Branches

## Background

**Why Do Reasoning Models Lose Coverage?** (2026) shows that SFT can improve `pass@1` while reducing `pass@k`. Its controlled `arithchain_2_10` Graph Branching task exposes an explicit first-step decision point. The official generator creates two locally valid 10-hop chains from one premise, but only one reaches the queried target; the other is a distractor. Node letters are randomized per problem. The paper also shows that prefix diversification can recover some lost coverage, motivating a **suppression rather than forgetting** interpretation.

**Are Language Models Aware of the Road Not Taken?** (2025) separately shows that unchosen future outcomes can be represented in hidden activations.

The one-step question is therefore:

> When SFT makes the globally target-reaching branch difficult to access at an ambiguous fork, does the hidden state still encode which branch is globally viable?

This is **not** a “two correct solution paths” experiment. The variable is global branch viability under local ambiguity.

## What we want to do

At the exact first fork, compare two quantities across the same SFT checkpoints:

1. **latent viability** — can a low-capacity candidate-conditioned probe identify which candidate reaches the target?
2. **output accessibility** — does normal next-token probability / sampled behavior actually select that viable candidate?

The strongest result would be:

```text
pass@k / viable-branch access decreases
but latent global-viability AUROC remains high
```

That would provide representational evidence for suppression rather than erasure.

## Validation experiment

### G0: establish the latent variable first

Use the official 1,000-test Graph Branching split and `unsloth/Qwen2.5-0.5B`.

`src/graph_parser.py` reconstructs each graph deterministically and labels the two first-fork candidates. No LLM judge is used.

At the canonical upstream decision prefix:

```text
To find the target value, we compute the following variables step by step:
1.
```

save hidden state `h_l` and candidate embedding difference

```text
de = e_A - e_B
z_l = h_l * de
```

A linear probe on `z_l` predicts whether alphabetically first candidate `A` is globally viable. This is a low-capacity diagonal bilinear compatibility probe. Five-fold stratified CV produces both aggregate AUROC and per-problem out-of-fold scores.

At the same state, compute the output baseline

```text
log p(" A") - log p(" B")
```

so a hidden-state result cannot be confused with ordinary next-token preference.

Run:

```bash
./prepare_upstream.sh
./run_g0.sh
```

Kill the topic if base-model viability is not stably decodable above chance.

### SFT dynamics

Reproduce the official forward SFT and evaluate the same checkpoints used by the seed paper's pass@k workflow:

```text
epoch 1  -> checkpoint-200
epoch 2  -> checkpoint-400
epoch 4  -> checkpoint-800
epoch 8  -> checkpoint-1600
epoch 16 -> checkpoint-3200
```

Then run:

```bash
./run_sft_dynamics_example.sh
```

### Behavior-side reproduction

The upstream helper mixes forward/reverse jobs and hard-codes GPUs `4 5 6 7`, so this folder provides a forward-only wrapper that still reuses the upstream prompt builder, `VLLMSampler`, and pass@k evaluator:

```bash
GPUS=0,1,2,3 NUM_SAMPLES=64 ./run_behavior_passk_forward.sh
```

It preserves the seed settings: temperature `1.0`, top-p `0.95`, max 512 tokens, 64 samples/problem, epochs `1/2/4/8/16`.

`src/analyze_sampled_branches.py` additionally measures per-problem probability of selecting the globally viable first branch and first-branch entropy.

## Decision rule

- **Suppression:** behavior/accessibility falls while latent viability remains readable → strongest result.
- **Erasure:** latent viability falls together with coverage → potentially interesting if branch-specific and robust.
- **Latent loss first:** representation degrades before behavior → surprising, worth follow-up.
- **No base signal / unstable probe:** stop; do not rescue with generic CKA/effective-rank collapse.

## Collision status — 2026-08-20

The novelty boundary is narrow. **When Are Teacher Tokens Reliable?** already introduces a behavior-level “branch-viability diagnostic” by forcing alternative next tokens and testing whether their continuations still succeed. Therefore branch viability itself is not new.

The remaining gap requires all three:

1. exact graph-ground-truth global viability;
2. **hidden-state** encoding of which concrete candidate is viable before selection;
3. tracking that representation through a coverage-shrinking SFT trajectory.

See [`VALIDATION.md`](./VALIDATION.md) for the exact source/code checks, implementation details, outputs, and current collision notes.
