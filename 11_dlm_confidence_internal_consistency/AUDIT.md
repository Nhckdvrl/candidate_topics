# Topic 11 — second pre-run audit (v3)

Date: 2026-08-22

This audit was completed **before any Topic-11 G-0 model scores were inspected**.

## Bottom line

The v2 implementation was mechanically sound and substantially better than the original registration, but its construct was still vulnerable to a shallow explanation:

> internally consistent examples literally repeated the same anchor number between the announcement and Step 1, while inconsistent examples did not.

For a bidirectional DLM, excluding Step-1 from the metric does not eliminate that relation: the whole observed sequence can influence every final-forward position. A positive v2 result could therefore be "numeric match sensitivity" rather than global reasoning consistency.

The v3 design removes that ambiguity instead of adding more post-hoc controls.

## Changes made

### 1. Move the consistency intervention into the future

v2:

```text
[announcement intervention]
[trajectory]
```

v3:

```text
[trajectory]
[future consistency-check intervention]
```

The primary scored tokens are inside the trajectory, *before* the consistency intervention.

This converts the key effect into a retroactive test. A future semantic contradiction must alter confidence on earlier unchanged reasoning tokens.

### 2. Remove literal anchor copying

Prompt and future check encode the anchor through different arithmetic aliases.

For anchor 23:

```text
prompt: 7 + 16
trajectory literal: 23
check: 11 + 12
```

The builder rejects accidental residuals that collide with anchors, operation values, or states in either mirrored trajectory.

Thus the factor is semantic equivalence, not "same digit appears twice."

### 3. Bracket the trajectory

External correctness is stated at the end of the user prompt. Internal consistency is stated after the trajectory.

The primary Step-2/3 result tokens lie between the two manipulated regions, reducing the old positional asymmetry.

### 4. Add semantic-alias prerequisite

Because v3 deliberately removes literal copies, a null is uninterpretable if the frozen model/scorer does not understand the simple arithmetic aliases.

A second protocol probe therefore tests confidence on an unchanged target token under correct vs incorrect alias expressions.

Failure => `INVALID_PROTOCOL_DO_NOT_INTERPRET`, not a topic kill.

### 5. Strengthen the original protocol gate

The arithmetic positive control no longer passes merely because its confidence interval is infinitesimally above zero.

Locked arithmetic gate:

- lower 95% CI > 0;
- mean gap >= 0.10.

This guards against a changed/broken environment that technically produces the right sign but no longer resembles the seed-paper phenomenon.

### 6. Add a scientific effect-size floor

Statistical significance alone cannot make the topic interesting.

Locked primary floor:

```text
Delta_consistency >= 0.01
```

If the 95% CI upper bound is below 0.01, G-0 has excluded a scientifically meaningful retroactive signal and the topic can be archived cleanly.

### 7. Stop requiring "coherence beats correctness" for survival

The scientific question first asks whether internal consistency is an independently identifiable signal.

Therefore:

- a meaningful stable retroactive consistency effect => topic stands;
- `CW > IC` => stronger headline;
- failure of `CW > IC` alone no longer kills a real consistency phenomenon.

This prevents an unnecessarily strong secondary contrast from erasing the actual question.

### 8. Freeze the runner in code, not only prose

The previous shell wrapper described the design as frozen while still allowing several scientific values to be overridden through environment variables.

v3 reads model identity, design size, seeds, intervention limit, probe gates, effect floor, and statistical settings directly from the locked config. Only infrastructure settings such as GPU count/IDs and batch size remain overrideable. The run directory snapshots the config and repository commit.

## Why this can now be one-shot decisive

A strong positive result cannot be explained by:

- confidence on the edited token itself — primary tokens occur earlier and are unchanged;
- immediate adjacency — primary tokens are Step 2/3, away from both manipulated boundaries;
- literal anchor copying — prompt/check use distinct semantic aliases;
- a favorite anchor token — X/Y roles are mirrored before inference;
- final-answer correctness — it is an orthogonal factorial factor;
- padding behavior — scoring batches contain only exact-length sequences;
- LLM-judge subjectivity — all labels are programmatic;
- a broken confidence scorer — two frozen protocol prerequisites must pass first.

A meaningful negative is also interpretable: if both prerequisites work yet the primary CI excludes a 1pp effect, the particular "global retroactive structural-consistency confidence" hypothesis has failed.

## Accepted limitations

1. G-0 is synthetic arithmetic. This is deliberate because identification is the first goal.
2. The final-forward confidence is the seed-paper observable; G-0 does not claim to identify the entire denoising dynamics.
3. The 1pp effect-size floor is a preregistered scientific judgment, not a universal constant.
4. Cross-DLM replication is G-1, not required to decide whether the phenomenon exists in LLaDA.

## Frozen execution discipline

Engineering fixes are allowed only if they restore the intended frozen measurement, e.g.:

- compatible transformers/remote-code API changes;
- GPU OOM solved by lowering batch size;
- cache/model-path repair;
- tokenizer API repair that preserves the one-token eligibility criterion;
- shard/runtime bugs.

Do not change after observing scores:

- factorial construction;
- primary metric;
- alias templates/bases;
- protocol thresholds;
- effect-size threshold;
- verdict rules;
- seed/subset selection to improve the result.

If a true scientific redesign is needed, preserve the v3 result and register the redesign explicitly rather than silently tuning G-0.
