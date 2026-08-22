# Topic 10 archive summary

## Final status

**ARCHIVED — positive 4×4 G0, failed non-toy qualification.**

The project found a genuine representation-sensitivity effect in a published competent 4×4 Sudoku setting, but attempts to establish a competent 9×9 object failed. Continuing would require model/data/configuration fishing or a substantially larger model, so the project is stopped on significance/scalability grounds rather than because the core hypothesis was disproved.

## Scientific question

Does a diffusion language model preserve solving outcome and mapped generation order when the underlying problem is changed only by an exact mathematical isomorphism?

Sudoku is attractive because row/band, column/stack, and transpose transformations preserve the constraint-satisfaction problem exactly while changing serialization position.

## Sequence of evidence

### 1. G0-v2 — 9×9 LLaDA-8B prerequisite failure

The first 9×9 fixed-grid formulation had healthy measurement but insufficient competence:

- identity exact `0/8`
- blank-cell accuracy `38.33%`
- same-serialization tau `1.0`
- native scheduler agreement `0.958`

This was correctly treated as an experimental-object failure, not a hypothesis result.

### 2. G0-v3 — published UPO 4×4 object succeeds

The public UPO setting was reproduced first:

- blank-cell accuracy `72.675%`
- exact-puzzle accuracy `59.0%`

Frozen discovery and untouched confirmation then gave substantial solve/fail instability under exact spatial isomorphisms:

- discovery flip rate `39.45%`, 95% CI `[31.64%, 47.27%]`
- confirmation flip rate `45.31%`, 95% CI `[37.89%, 52.73%]`

Mapped order remained weakly positive (`tau ≈ 0.11–0.12`) but did not cleanly beat both row-major and boundary-first nulls in both splits.

Interpretation: the strongest reproducible phenomenon is outcome non-equivariance. There may be a weak structural component in generation order, but it is not the clean headline.

### 3. G1-v4 — Dream-7B 9×9 seed-aligned reconstruction fails

The seed paper's reported 9×9 Dream curve could not be exactly reproduced because the paper does not release its 9×9 dataset/generator or enough model/training provenance to identify the exact object.

A locked reconstruction using `Dream-org/Dream-v0-Instruct-7B`, the public Dream trainer, 50 train / 100 test puzzles, and the paper's described prompt produced:

- epoch 2: `6/100` exact
- epoch 5: `3/100` exact

Training loss became very small while held-out exact accuracy remained near zero and declined. Output failures were dominated by instruction repetition and malformed/flat/truncated matrices. The run was stopped before epoch 10 and before any 9×9 symmetry generation.

## Why this is an archive, not a hypothesis kill

The 4×4 result demonstrates that exact problem isomorphisms can substantially change DLM behavior. Therefore the phenomenon exists.

However, the intended scientific claim needs a non-toy competent regime. We failed to instantiate one with the recoverable 7B/8B routes. The remaining options require unresolved provenance changes or a much larger model such as LLaDA2.0-flash-100B.

That would turn the project from a clean question into a search over experimental objects. We stop before that happens.

## Transferable lessons

1. **Separate phenomenon existence from meaningful-regime existence.** A clean effect on a toy object is not enough if the paper claim implicitly depends on harder reasoning.
2. **A failed prerequisite is not a negative scientific result.** The 9×9 failures say the intended object was not instantiated; they do not say representation sensitivity disappears at 9×9.
3. **Do not rescue scale with configuration fishing.** Once competence depends on unresolved model variant, unreleased data distribution, prompt, optimizer, and decoder choices, the confirmation is no longer clean.
4. **Exact isomorphism is still a useful intervention.** The 4×4 result shows that problem-preserving transformations can expose large behavioral instability with very little interpretive machinery.
5. **The strongest observable may differ from the original title.** The original project focused on generation order, but solve/fail outcome non-equivariance became the clearest effect.

## Reopen condition

Only reopen if a genuinely competent 9×9 object becomes directly available, e.g. released seed-paper data/checkpoint/configuration or accessible large-model infrastructure. Do not reopen merely to try another prompt/model/config combination.
