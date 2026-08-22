# Server handoff

Work from the repository's current topic branch / merged `main` after this topic lands.

## What the project is asking

Recent VLA work shows that frozen internal signals can predict whether a rollout will succeed. We want to know whether those signals represent **the current policy's own competence** or mostly **generic state difficulty**.

The clean experiment uses the same LIBERO state with several checkpoints from one pi0.5 fine-tuning trajectory (2k / 3k / 9k).

## Step 1 — reproduce the released policy path

Make sure the public pi0.5 LIBERO checkpoints load under the intended openpi / LIBERO evaluation stack and obtain sensible published-scale success behavior.

Purpose: rule out checkpoint, normalization, action-chunk, or environment mismatches before interpreting any failure.

## Step 2 — behavioral panel only

Run all three checkpoints on exactly the same task/seed panel and save:

```text
task, seed, checkpoint, success
```

Run `src/analyze_disagreement.py`.

Purpose: determine whether natural **bidirectional** checkpoint disagreement exists. If not, stop. Do not manufacture hard states.

## Step 3 — freeze one identifiable pair

Choose the checkpoint pair using discovery states only, based on two-way crossover support. Freeze it and use disjoint states for the representation test.

Purpose: prevent a global “later checkpoint is better” shortcut.

## Step 4 — extract one predeclared hidden state

Audit COAST / openpi code and hook the corresponding pi0.5 action-expert representation around layer 11. Do not layer-sweep.

Purpose: test the already motivated success representation, not search for any layer that works.

## Step 5 — shared-probe paired test

Fit one shared linear success probe on discovery states from both checkpoints. On independent crossover states, compare the two probe scores from the identical physical state.

Purpose: ask whether the score follows **which checkpoint actually succeeds**, while generic state difficulty is held fixed by construction.

## Resources

Use idle GPUs freely on:

```text
fvcrc10 fvcrc11 fvcrc12 fvcrc13 fvcrc15 fvcrc20 fvcrc21
```

Independent checkpoint/seed shards are preferable to cross-node distributed training. There should be essentially no large-model training here; the expensive part is simulator rollout and frozen VLA inference.

Prefer existing local environments. If dependencies conflict, make an isolated environment and fix them without modifying shared system packages.

## Stop behavior

The goal is to kill the topic quickly if the clean contrast is absent.

Do not add perturbations, new confidence methods, hand-selected states, layer sweeps, or complicated controls after a negative result.
