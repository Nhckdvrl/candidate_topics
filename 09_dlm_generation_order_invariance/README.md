# 09 — Is DLM Generation Order Invariant to Problem Isomorphisms?

**Status:** REGISTERED CANDIDATE — HIGH PRIORITY

## Natural question

> When a problem is changed only by an exact symmetry that preserves its underlying structure, does a diffusion language model preserve the order in which it solves the problem?

This targets a core scientific interpretation of arbitrary-order generation: whether observed generation order reflects **problem structure** or is substantially induced by **serialization / sampler bias**.

## Seed tension

### Seed A — generation order appears adaptive

*Parallelism and Generation Order in Masked Diffusion Language Models* (ACL Findings 2026) studies multiple masked diffusion LMs across many tasks and reports that generation order varies with task, reasoning stage, and correctness. Sudoku is especially suggestive: models tend to fill easier cells earlier, motivating the interpretation that arbitrary-order decoding can follow computational/problem structure.

### Seed B — confidence-based decoding has strong non-semantic ordering bias

ACL 2026 work on uncertainty decoding bias shows that confidence/uncertainty-based token selection can exhibit rigid boundary and trivial-token biases. Therefore, an early token is not automatically evidence that the model judged it logically prior or easier.

These results leave a direct identification gap:

> Is the observed Sudoku generation order an invariant of the underlying constraint problem, or partly an artifact of how the same problem is serialized into token positions?

## Why Sudoku is unusually clean

Sudoku admits exact structure-preserving transformations, including:

- row permutations within a band;
- band permutations;
- column permutations within a stack;
- stack permutations;
- transpose;
- digit relabeling.

For a puzzle `x` and an isomorphic puzzle `T(x)`, the solution spaces are in one-to-one correspondence. Every blank cell `c` maps to a unique blank `T(c)`.

The experiment therefore does **not** claim a perfect `do(position)` intervention with every token-level trajectory state fixed. The cleaner scientific object is:

> **Is the generation policy approximately equivariant under exact problem isomorphisms?**

## G-0: paired isomorphism test

Take a fixed set of Sudoku puzzles. For each puzzle, generate several randomly sampled valid isomorphs.

For every blank cell, record its **finalization rank** rather than only raw diffusion step:

`r_x(c)` = rank at which cell `c` becomes finalized in puzzle `x`.

Map ranks through the known isomorphism and compare:

`r_x(c)` vs `r_T(x)(T(c))`.

### Primary first figure

For each original/isomorph pair, measure rank-order preservation over mapped blanks, e.g. Kendall rank correlation:

`tau_iso = Kendall(r_x(c), r_T(x)(T(c)))`.

Then show the distribution of `tau_iso` across puzzles / transforms.

No hidden states, no learned probe, no threshold search, no training.

## Interpretation

### High isomorphism invariance

Mapped cells retain nearly the same relative finalization order across aggressive spatial re-serializations. This is strong evidence that DLM generation order tracks a structural property of the constraint problem rather than merely absolute token position.

### Low invariance with systematic positional movement

The ordering changes substantially when the same logical cells move through serialization. Then the common "easy-first / structure-following" interpretation must be weakened: observed order is substantially sampler/serialization dependent.

### Mixed result

A stable competition between structural invariance and positional bias would motivate a clean decomposition into semantic/problem scheduling versus decoder/sampler scheduling.

## Minimal secondary diagnostic

Only after the primary isomorphism result is visible, inspect whether rank changes covary with simple positional quantities such as distance to sequence boundaries. This is explanatory follow-up, not a prerequisite for the phenomenon.

## Kill line

The first paired symmetry experiment must carry the topic.

Kill or sharply narrow the topic if generation order is too unstable even across repeated decoding of the **same** serialization to define a meaningful policy-level observable, or if isomorphic and same-instance variation cannot be separated with the frozen protocol.

Do not rescue the project with hidden-state analysis, hand-designed Sudoku difficulty scores, or broad sampler/model fishing before the basic invariance object is established.

## Why either result matters

The result changes the interpretation of a claimed distinctive advantage of DLMs:

- invariance would provide unusually direct evidence that arbitrary-order generation follows problem structure;
- non-invariance would show that a visually compelling "logical solution order" can be induced substantially by serialization and confidence-decoding artifacts.

Either direction answers the same scientific question rather than erasing it.
