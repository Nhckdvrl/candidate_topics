# Literature audit — Topic 10

## Seed phenomenon

**Zhong et al., Findings of ACL 2026, _Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow_.** The paper evaluates eight MDLMs across 58 benchmarks and reports task-dependent generation order. Sudoku is the clearest motivating case: easier blanks tend to be filled earlier, supporting the interpretation that arbitrary-order decoding can follow problem structure.

The present project does not re-measure only that correlation. It asks whether the ordering is preserved when the same constraint problem is moved through an exact Sudoku symmetry.

## Decoder implementation anchor

The public LLaDA generator reveals masked positions by computing model logits, obtaining the confidence of each proposed token, selecting the highest-confidence masked positions, and making those positions irreversible. The G0 decoder keeps this selection rule but instruments each reveal step. It uses one reveal per step so the finalization order is exact rather than tied.

Reference implementation: `ML-GSAI/LLaDA`, `generate.py`, model `GSAI-ML/LLaDA-8B-Instruct`.

## Why the spatial-only primary test matters

Sudoku has both spatial and digit-relabel symmetries. Digit relabeling is a valid logical isomorphism but changes token identity; that could mix structural invariance with token-frequency/token-embedding effects. Therefore the primary G0 uses only row/band, column/stack and transpose automorphisms and keeps all digit labels fixed. Digit relabeling can be a later robustness experiment only after the primary claim is established.

## Nearby decoding-bias result

ACL 2026 work on uncertainty/confidence decoding reports rigid boundary and trivial-token biases in DLM decoding. That result supplies an independently established non-semantic force that can affect generation order. It is precisely why observational "easy first" behavior is insufficient to identify structural scheduling.

## Exact gap retained after audit

The seed paper establishes adaptive/easy-first order. Bias work establishes non-semantic confidence-decoding preferences. The missing clean contrast is:

> under a known exact problem isomorphism that preserves Sudoku constraints, do mapped cells preserve their relative finalization order?

The G0 is intentionally designed so this question can be answered without hidden-state analysis, a learned difficulty model, a judge, or a new decoding method.
