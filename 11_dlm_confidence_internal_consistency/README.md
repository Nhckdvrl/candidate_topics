# 11 — What Does Diffusion Confidence Actually Know?

**Status:** **ARCHIVED — `KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL` at frozen G-0**

- [Full G-0 result](./G0_RESULT.md)
- [Archive summary](./ARCHIVE_SUMMARY.md)
- [Pre-run audit / locked design](./AUDIT.md)

## Scientific question

> Does native diffusion-LM confidence encode **global internal consistency of a reasoning trajectory**, independently of whether the trajectory is externally correct?

The project was motivated by the observation that diffusion-LM confidence can respond strongly to reasoning correctness and arithmetic contradictions. The stricter question here was whether that signal behaves like a meaningful **global reasoning-state** signal rather than merely an aggregate/local sequence-compatibility score.

## Frozen v3 identification

The final G-0 used a retroactive `internal consistency × external correctness` factorial.

A fixed arithmetic trajectory appears first. External correctness is manipulated only in the prompt. Internal consistency is manipulated only in a semantic consistency-check suffix that comes **after** the trajectory.

Prompt and suffix encode the relevant anchor through different arithmetic aliases rather than literal digit copies.

The locked primary metric is:

```text
confidence_result_middle
```

which scores the unchanged Step-2/Step-3 result tokens **before** the future consistency suffix.

Therefore a positive primary effect would require a future contradiction to retroactively change confidence on earlier, unchanged reasoning tokens.

## G-0 protocol validity

Both preregistered prerequisites passed strongly:

```text
Arithmetic result substitution
mean gap = 0.426361
95% CI   = [0.390624, 0.462064]
locked floor = 0.100

Semantic-alias comprehension
mean gap = 0.214693
95% CI   = [0.186151, 0.244417]
locked floor = 0.020
```

Eligible mirrored anchor pairs: **140**.

A narrow digit-level-tokenizer compatibility repair was made to preserve the frozen one-token eligibility semantics. It did not change the scientific factors, primary metric, thresholds, or verdict logic.

## Decisive result

```text
Primary: confidence_result_middle
Delta_consistency = -0.000003
95% CI            = [-0.000055, 0.000025]
locked meaningful floor = 0.010
```

The upper confidence bound is around **400× below** the predeclared `0.010` minimum-worthy effect.

This is not an underpowered gray zone. The frozen G-0 excludes a retroactive consistency signal of the magnitude required for the project-level claim.

The strongest cross-cell contrast also went in the opposite direction:

```text
coherent-wrong - incoherent-correct = -0.000587
95% CI                            = [-0.001737, -0.000009]
```

## Why the large full-sequence effect does not rescue the topic

`confidence_full` showed:

```text
Delta_consistency = 0.013765
95% CI            = [0.013069, 0.014462]
```

But `confidence_full` includes the manipulated future consistency-check suffix itself.

The entire purpose of v3 was to distinguish:

```text
aggregate/full-sequence consistency sensitivity
```

from:

```text
meaningful retroactive consistency signal on earlier unchanged reasoning tokens
```

The former is strong. The latter is essentially absent.

Switching to `confidence_full` after seeing the result would therefore change the scientific question and violate the frozen measurement contract.

## Final conclusion

The supported bounded statement is:

> Under a controlled retroactive intervention, LLaDA's native final-forward confidence does not exhibit a meaningful global consistency signal on unchanged middle reasoning-result tokens. Strong sequence-level consistency sensitivity therefore should not by itself be interpreted as evidence that confidence is globally distributed over the reasoning trajectory.

This does **not** claim that DLM confidence contains no consistency information anywhere. It falsifies the stronger global/retroactive interpretation that made Topic 11 worth pursuing.

## Final action

Topic 11 is archived.

Do **not** proceed to G-1, increase `n`, switch the primary metric, search token regions, change pooling, or sweep models/prompts/datasets to rescue the same claim.

Reopen only if a genuinely new external observation motivates a separately registered scientific question with a distinct identification strategy.

## Preserved validation code

The directory keeps the frozen validation harness for auditability:

```text
build_design.py
score_llada.py
analyze.py
run_g0.sh
configs/g0.json
tests/
```

The code remains useful as a record of the exact experiment that produced the negative result; it should not be treated as an invitation to tune the archived hypothesis.
