# Literature audit — Topic 10

## Seed phenomenon

**Zhong et al., Findings of ACL 2026, _Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow_.** The paper evaluates eight MDLMs across 58 benchmarks and describes generation order as adaptive to domain, reasoning stage, and correctness. Sudoku is the headline non-sequential case: the paper states that easier blanks tend to be filled first and presents a 150-puzzle Sudoku study with varying difficulty.

The present project does not merely re-measure that correlation. It asks whether the policy is **equivariant to an exact representation change of the same Sudoku CSP**.

## Independent pressure from decoding bias

**Huang et al., ACL 2026, _Empirical Analysis of Decoding Biases in Masked Diffusion Models_.** This work shows that uncertainty/confidence samplers can induce rigid boundary-first trajectories and over-select trivial high-frequency tokens. Sudoku is explicitly included among their planning tasks. This gives an independently established non-semantic force capable of shaping apparent generation order.

That tension motivates the exact question here: if logical structure is unchanged but serialization moves, which force wins?

## Competence audit correction

The original G0-v2 audit overclaimed the evidence for its experimental object. Zhong et al.'s 9x9 zero-shot Sudoku result uses LLaDA2.0-flash-100B, not LLaDA-8B-Instruct. The independent ICLR 2026 UPO result that uses LLaDA-8B is a **4x4** masked-Sudoku setting, and its reported 70.5% max-confidence number is blank-cell accuracy, not exact-puzzle accuracy; the released validator scores only originally blank cells.

Our 9x9 LLaDA-8B fixed-grid smoke therefore records a prerequisite failure, not a hypothesis rejection: identity exact accuracy was 0/8 and blank-cell accuracy was 38.33% over the eight identity traces. The pipeline itself was healthy (same-serialization tau 1.0; native scheduler agreement 0.958). This distinction is now explicit.

G0-v3 returns to the published competent object: the UPO 4x4 test CSV, published system prompt and tuple-rendered answer template, LLaDA-8B-Instruct, and full-vocabulary max-confidence one-position decoding. The 500-row reproduction obtained 72.675% blank-cell accuracy and 59.0% exact-puzzle accuracy before any symmetry test.

## Decoder implementation anchor

The public `ML-GSAI/LLaDA` `generate.py` implementation predicts masked tokens, computes each proposed token's probability under the full vocabulary, and reveals the highest-confidence masked positions for `remasking='low_confidence'`.

Our controlled Sudoku decoder necessarily constrains cell **content** to digits `1..9`, because each mutable position is definitionally a Sudoku digit. The important audit correction in G0 v2 is that confidence is **not renormalized over those nine digits**. We select the best valid digit but rank positions by that digit's probability under the original full vocabulary.

G0 v2 goes one step further: at every reveal step it also computes which position the completely native full-vocabulary confidence scheduler would have selected. Traces therefore report both `native_digit_argmax_fraction` and `native_scheduler_pick_same_fraction`. The latter directly quantifies whether the task grammar changed the scheduling policy rather than assuming it did not.

## Why spatial-only is primary

Sudoku also admits digit relabeling. That is a valid logical isomorphism but changes token identity, mixing structural equivariance with frequency/embedding effects. Primary G0 therefore uses only row/band, column/stack, and transpose automorphisms and keeps digit labels fixed. Digit relabeling is a later robustness test, not a discovery degree of freedom.

## Exact gap retained after audit

The existing literature gives:

1. an observational claim that Sudoku generation order can look problem-adaptive;
2. a separate demonstration that confidence decoding contains strong position/surface biases.

What it does not identify is:

> under a known exact Sudoku isomorphism, are solve outcome and mapped-cell finalization order preserved beyond what simple positional schedulers would predict?

G0-v3 answers that with paired outcome flips, mapped-cell Kendall tau, same-serialization stability, transform-matched positional nulls, and source-puzzle-cluster uncertainty, first on the published 4x4 object. No hidden-state mechanism is needed for the question to stand. The 9x9 LLaDA2.0-flash/Dream confirmation remains a later stage, not a rescue of the failed v2 object.
