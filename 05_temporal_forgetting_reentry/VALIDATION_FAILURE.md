# Validation outcome: conceptual identification failure

The validation run for this topic was stopped before scoring because the
experimental design cannot identify the claimed mechanism.

## Why the controls are insufficient

Adding a prefix changes the task itself:

```text
P(solve | x) != P(solve | x + correct prefix)
```

The prefix can shrink the search space, provide intermediate variables,
change local token compatibility, and rule out incorrect strategies. Thus,
even an old-self advantage over other-correct may reflect stylistic or local
continuation compatibility rather than re-entry into a stored old skill.

The notion of an "old route" is also not a stable object. A trajectory can
transition through old, new, and different reasoning states before reaching a
correct answer, so success after supplying an old prefix does not establish
that the model re-entered the former route.

Teacher-forced suffix NLL has the same limitation: a high value only shows
that the old continuation is compatible after the old prefix is given. It does
not show that the skill remains available without that cue.

Therefore the controls (old-self, other-correct, final-wrong, never-correct,
and teacher-forced NLL) do not resolve the central identification problem.

## Run disposition

- The run was stopped before scoring and before any claim-level gate.
- No scientific result is reported from the partial samples.
- All eight downloaded `UWNSL/Qwen2.5-7B-deepscaler_4k_step_{32,64,96,128,160,192,224,256}` checkpoints were removed from the Hugging Face cache after stopping generation.
- The local raw outputs and logs remain untracked for forensic reference and are intentionally not part of this commit.
