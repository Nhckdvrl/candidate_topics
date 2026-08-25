# Topic 28 — G1 Adjacent-Order Results

## Frozen verdict

`STOP_ORDER_DEPENDENCE`

The frozen outcome-blind adjacent-swap G1 does not support the proposed
path-dependent reversal explanation on Qwen2.5-7B-Instruct. All
artifact/measurement gates passed. The scientific gates did not.

Per the preregistration, this result does not authorize another model, prompt,
panel, scorer, threshold, random shuffle, full reversal, or mechanism run.

## Receipt

| Field | Frozen value |
|---|---|
| question artifact | `mgor/protobowl-11-13`, `progressive-clues/eval` |
| question revision | `3dae05a66d3e0fd8c6b23ef8656ff6f4437bb1d4` |
| model | `Qwen/Qwen2.5-7B-Instruct` |
| model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| inference | bfloat16, greedy, `max_new_tokens=24` |
| scorer | frozen exact normalized `clean_answers` alias match |
| bootstrap | 2,000 whole-question resamples, seed `20260825` |
| host/GPU | `fvcrc20`, NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 3 |
| environment | `/home/xiang/venvs/ragen` |
| Python / PyTorch / Transformers | 3.12.0 / 2.8.0+cu128 / 4.57.6 |

Commands:

```bash
cd 28_progressive_truthful_clue_reversal
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /home/xiang/venvs/ragen/bin/python -m unittest discover -s tests -v

# DEBUG schema smoke only; code forces DEBUG_NO_VERDICT
CUDA_VISIBLE_DEVICES=3 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /home/xiang/venvs/ragen/bin/python g1_order_swap.py \
  --debug-limit 8 --batch-size 16 --device cuda:0 \
  --out-dir artifacts/g1_debug

# Complete frozen scientific run
CUDA_VISIBLE_DEVICES=3 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /home/xiang/venvs/ragen/bin/python g1_order_swap.py \
  --batch-size 32 --device cuda:0 --out-dir artifacts/g1
```

The final implementation passed 19/19 unit tests. The full run generated all
1,992 planned states exactly once.

## Outcome-blind panel audit

The panel builder read only the pinned question artifact. It did not read
released responses, released scores, G0 trajectory outcomes, or G0 reversal
events.

| Quantity | Result |
|---|---:|
| official cumulative question rows | 3,042 |
| source questions | 782 |
| frozen specificity median | 6.8378626 |
| eligible panel boundaries | 498 |
| unique panel questions | 415 |
| expected state outputs | 1,992 |
| observed state outputs | 1,992 |
| duplicate `(boundary,state)` outputs | 0 |
| valid four-output boundaries | 498/498 |
| non-empty outputs | 1,992/1,992 |
| non-truncated outputs | 1,992/1,992 |
| single-line outputs | 1,991/1,992 |
| final original/swap clue-count mismatches | 0 |

The one multiline continuation was non-empty and non-truncated and was scored
exactly as frozen; it did not create a reversal event.

## Primary paired result

| Quantity | Original | Adjacent swap |
|---|---:|---:|
| first-state correct | 316/498 (63.45%) | 337/498 (67.67%) |
| final-state correct | 357/498 (71.69%) | 349/498 (70.08%) |
| reversal events | 15/498 (3.01%) | 15/498 (3.01%) |
| reversal rate conditional on first-state correct | 15/316 (4.75%) | 15/337 (4.45%) |

The preregistered primary paired estimate was:

```text
delta_order = P(original reversal) - P(swap reversal)
            = 0.0000
95% qid-cluster bootstrap CI = [-0.02001, +0.02016]
```

The clean same-clue-multiset final-state estimate also went in the opposite
direction from the path-dependence prediction:

```text
delta_final_error = P(original final wrong) - P(swap final wrong)
                  = -0.01606
95% qid-cluster bootstrap CI = [-0.03726, +0.00581]
```

There is no positive original-order effect under either frozen estimand.

## Frozen gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| panel boundaries | `==498` | 498 | PASS |
| unique questions | `==415` | 415 | PASS |
| valid four-output boundaries | `>=0.98` | 1.000 | PASS |
| O1-correct support | `>=100` | 316 | PASS |
| S1-correct support | `>=100` | 337 | PASS |
| common-belief support | `>=75` | 285 | PASS |
| original reversal support | `>=20` | 15 | FAIL |
| `delta_order` | `>=0.02` | 0.0000 | FAIL |
| lower CI for `delta_order` | `>0` | -0.02001 | FAIL |
| `delta_final_error` | `>=0.01` | -0.01606 | FAIL |
| lower CI for final error | `>0` | -0.03726 | FAIL |

## Competing-explanation audit

The paired structure is inconsistent with the preregistered history-dependent
destabilization account:

- `280/285` common-belief boundaries remained correct in both final orders;
- among those 285 boundaries, original-only final harm occurred once and
  swap-only final harm occurred once;
- three common-belief boundaries reversed under both orders;
- original-only and swap-only reversal counts were both 12; three boundaries
  reversed under both orders;
- 130 boundaries were final-wrong under both orders, versus 11 original-only
  final harms and 19 swap-only final harms;
- the final normalized prediction was identical across orders in 437/498
  boundaries (87.75%); among the 130 both-wrong boundaries, 100 produced the
  same normalized prediction;
- Q3 had 5 original versus 7 swap reversals; Q4 had 10 versus 8, cancelling in
  aggregate rather than yielding a stable specificity-by-order effect.

These diagnostics favor order-independent clue-set difficulty/conflict over
the proposed arrival-history account on this frozen model and panel. They do
not establish a semantic conflict mechanism.

## Scorer audit and limitation

The strict primary scorer was frozen before inference and was not modified.
Manual inspection shows that several of the 15 nominal original reversals are
strict-alias false negatives rather than clear semantic errors, including
`Wolfgang Amadeus Mozart -> Mozart`, `Crusades -> Crusade`, `muscles ->
Muscle`, `tornadoes -> Tornado`, and `iPhone -> iPhone 5`. No aliases were
added and no result was rescored.

This undercoverage means the absolute 15-event reversal count should not be
treated as a precise semantic-reversal estimate. It does not rescue the order
hypothesis: the frozen paired reversal counts are exactly symmetric, the
same-multiset final contrast is non-positive, and the common-belief discordance
is one event in each direction. The correct classification is therefore a
scientific stop for this preregistered order explanation, with a measurement
caveat on absolute event labels—not a measurement-gate failure and not a
license for outcome-driven scorer repair.

## Interpretation

G0 still establishes large-scale truthful-evidence reversals in released AI
trajectories, and the descriptive specificity relation remains real for that
artifact. G1 does **not** support the stronger claim that adjacent arrival
history causally drives those reversals in the frozen local-model regime.

The proposed paper upgrade to "belief updating is path-dependent" stops here.
Do not continue to hidden states or search other models/configurations for a
positive order result.
