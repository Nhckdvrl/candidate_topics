# Validation contract — Topic 06

This file freezes the first falsification experiment before inspecting model results.

## 1. Claim ladder

The pilot can support only the following sequence.

### Claim A — acquisition premise

The model's behavior is sensitive to action–outcome contingency during training.

### Claim B — cross-task transfer

Past uncontrollability reduces active intervention on step 1 of a novel, objectively controllable task.

### Claim C — diversity amplification

The transfer in Claim B is larger when the same amount of uncontrollability was distributed across semantically different task families rather than concentrated in one family.

The pilot **does not** establish a permanent parameter-level worldview. It tests an interaction-history-induced policy prior. Parameter adaptation / persistent memory is a follow-up only after Claims A–C are alive.

## 2. Identification logic

The experiment must preserve all of the following.

### 2.1 Master–yoked outcomes

For each `(diversity, pair_id)`, the uncontrollable session receives exactly the training success/failure sequence observed by the paired controllable master.

Hard invariant:

```text
success_master[t] == success_yoked[t]  for every training t
```

If this fails, discard the run as a software error.

### 2.2 Same exposure count

C1, U1, C10, U10 have exactly the same:

- number of episodes;
- number of trials per episode;
- episode boundary count;
- active-action cardinality;
- intervention cost;
- success reward;
- test task and test potential-outcome realization for the same `pair_id`;
- episode-level latent random numbers / effective-action assignments for the same `pair_id`.

Distributed experience must not silently add more resets or more observations.

### 2.3 No construct naming

Prompts must not say `helplessness`, `uncontrollable`, `controllability`, `worldview`, or `prior belief`. The model should not be asked to summarize whether its actions matter.

### 2.4 Novel test is identical

For a given `pair_id`, all four conditions use the same held-out family, effective action, and test potential-outcome random draws. The primary endpoint is taken before any test outcome is shown.

## 3. Primary endpoint

For valid responses only:

```text
Y = 1 if test-step-1 action is either active intervention
Y = 0 if it is wait
```

Within each diversity level, master and yoked observations share `pair_id`.

```text
H1  = mean(Y_C1  - Y_U1)
H10 = mean(Y_C10 - Y_U10)
D   = H10 - H1
```

Hypothesis: `D > 0`.

Use pair-level bootstrap resampling within each diversity level. No model/task/temperature selection may be based on the sign of `D` and then reported as confirmatory.

## 4. Secondary outcomes

Report but do not use to rescue a failed primary endpoint:

- active-intervention rate over test steps 1–K;
- first active-intervention step;
- first selection of the effective intervention;
- test success / net score;
- recovery trajectory after the first successful active intervention;
- late-training active-intervention rate.

No self-report questionnaire or hidden-state probe belongs in G0.

## 5. Technical gates

Before science:

1. `python -m pytest -q tests` passes;
2. master/yoked outcome sequences are exact matches;
3. all four conditions have equal exposure counts;
4. invalid-action rate at test is ≤1%;
5. logs preserve raw model output, normalized action, latent action, success, cost, and condition metadata;
6. `orbital_station` never appears in training renderers.

An invalid response uses a deterministic active-A fallback only so the environment can continue. It is marked `valid_action=false` and excluded from the primary endpoint. If invalidity exceeds 1%, stop rather than interpreting passivity.

## 6. Staged execution

### Stage S0 — software smoke

```bash
./run_smoke.sh
```

The mock client is random and has no scientific meaning. This stage tests mechanics, JSONL writing, yoking, and analysis only.

### Stage S1 — model preflight

Use 16–25 master/yoked pairs per diversity with `--preflight` (40 training experiences/session). This is for:

- instruction-following / parse reliability;
- whether the model reacts at all to sequential feedback;
- rough magnitude estimation.

Do not tune semantic families or endpoint definitions after seeing S1.

### Stage S2 — locked pilot

Default:

```text
50 pairs × 2 diversity levels × 2 control levels = 200 sessions
100 training experiences/session
8 test trials/session
```

The 200-session pilot is the first substantive decision point.

### Stage S3 — confirmation

Only if S2 is alive. Run a fresh seed block with at least 250 pairs per diversity (1,000 sessions total) and the same frozen endpoint. Prefer a second model family as a separate replication, not pooled model selection.

## 7. Decision categories

### TECHNICAL STOP

Invalid test-action rate >1%, broken yoking, unequal episode structure, or prompt leakage. Fix software and rerun; this is not a scientific negative.

### SCIENTIFIC STOP / DOWNGRADE

With adequate precision, pooled novel-test transfer is negligible:

```text
mean(H1, H10) < ~0.02
```

Interpretation: the model localizes the interaction history or ignores it; do not add probes to manufacture a prior.

### BOUNDARY RESULT

Pooled transfer exists but `D` is near zero:

```text
H1 > 0 and/or H10 > 0
D ≈ 0
```

Interpretation: uncontrollability transfers, but experience diversity does not appear to control abstraction breadth. This directly answers the natural question but weakens the proposed "worldview" paper. Do not silently pivot the primary claim.

### CONTINUE

Pilot target:

```text
D >= 0.05
```

with directionally stable bootstrap mass and no technical failures. Confirmation, not pilot significance, establishes the result.

### STRONGEST PATTERN

```text
P(active | C1) ≈ P(active | C10)
P(active | U10) < P(active | U1) < controllable
```

and U10 recovers more slowly after test evidence becomes favorable.

This pattern separates a generic diversity-induced conservatism explanation from diversity-specific generalization of uncontrollability.

## 8. Alternative explanations already controlled

The 2×2/yoked design directly addresses:

- total number of failures/successes (matched within master–yoked pair);
- reward valence and timing (matched within pair);
- number of episode resets (equal C1/U1/C10/U10);
- generic semantic diversity effects (`C10 - C1` is the control term in the interaction);
- test difficulty (identical test task);
- explicit demand characteristics (construct words absent).

It does **not** identify parameter-level learning versus in-context adaptation. The initial claim must stay at the interaction-history level.

## 9. Follow-ups only after a positive result

Do these in order, one at a time.

1. **Semantic-distance curve:** similar vs intermediate vs distant held-out test families. If concentrated failure transfers only to similar tasks but distributed failure transfers to distant tasks, the abstraction story strengthens.
2. **Richer causal dynamics:** port the 2×2/yoked manipulation to Hew & Bramley's continuous causal environment.
3. **Memory mechanism:** raw transcript vs persistent episodic memory / distilled memory.
4. **Parameter acquisition:** online LoRA/RL or small-model finetuning using the same experience structure, asking whether the effect survives after history is removed from context.

Do not start at steps 2–4 before the behavioral contrast exists.
