# G1/v4 — Dream-7B 9×9 seed-aligned replication

## Scope

G0-v3 established the phenomenon in the published UPO 4×4 setting. G1/v4 is the next prerequisite: test whether a 7B Dream model can be made competent on the 9×9 Sudoku setting used in the seed paper before any new symmetry manifest is frozen.

The required order is:

1. generate and checksum a 150-puzzle corpus;
2. train on 50 and hold out 100;
3. evaluate ordinary exact-grid accuracy at epochs 0, 2, 5, and 10;
4. only after the competence result is recorded, freeze the spatial-isomorphism confirmation.

No G1 symmetry traces are valid yet.

## Provenance audit

The seed paper says it randomly generated 150 unique 9×9 puzzles, split them 50/100, used the prompt shown in Appendix A.6.1, and fine-tuned Dream-7B with the official training framework. It reports Dream scores of 9, 31, 65, and 80 at epochs 0, 2, 5, and 10 respectively.

The official [Dream repository](https://github.com/DreamLM/Dream) releases the model, inference code, and SFT/FSDP trainer. Its checked-in Sudoku evaluation data are 4×4 files, not the seed paper's 9×9 corpus. The seed paper does not release the 9×9 generator or the 150 puzzle file in its arXiv source. Therefore this repository records the current corpus as **seed-aligned reconstruction**, not an exact recovery of the seed data.

This distinction is locked so a successful curve cannot be over-claimed as a literal reproduction.

## Dataset reconstruction

`src/make_g1_v4_dataset.py` uses the already audited randomized 9×9 solver/generator, a fixed seed, unique-solution clue removal, and a documented blank-count cycle. It emits JSONL records with the initial grid, unique solution, exact prompt, and exact response. The generated files are checked by the existing Sudoku tests and receive SHA-256 hashes in the manifest metadata.

The prompt follows the paper's Appendix A.6.1 wording: provide a zero-filled 9×9 matrix, require rows/columns/3×3 subgrids to contain 1–9, and require only a completed Python-style 2D array with no explanation.

`src/eval_g1_v4.py` scores only exact 9×9 array equality against the locked solution. It uses `ast.literal_eval`, never executes model text, rejects malformed arrays, and requires prediction IDs to match the frozen test set exactly.

## Model and cache

The target is `Dream-org/Dream-v0-Instruct-7B`. It is downloaded to the shared public Hugging Face cache at `/home/xiang/.cache/huggingface/hub`, never into this repository. The exact snapshot path and file hashes are recorded in the run log when download completes.

## Scientific gate

The ordinary exact-solve curve is a prerequisite diagnostic, not a hypothesis result. We will not tune prompt wording, change the test subset, alter the exact-match rule, or freeze symmetry after seeing the curve. If the reconstructed corpus cannot support a competent object, the outcome is a provenance/replication limitation and not evidence against Topic 10.
