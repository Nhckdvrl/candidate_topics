# Validation contract — Topic 05

This file is the **locked scientific contract for fast topic validation**. The aim is not to save GPU-hours; the aim is to make the first experiment maximally decisive while preventing post-hoc rescue.

## 0. Scientific question

> When a reasoning model reliably solved a problem earlier in training but reliably fails it later, is the former route still accessible/compatible and merely no longer self-selected, or has that route/competence substantially eroded?

Primary evidence is behavioral + likelihood based. **No hidden-state probe is part of G-1/G0.**

---

# G-1 — Does robust temporal forgetting survive repeated sampling?

## Dataset / lineage

Primary:

```text
model lineage  UWNSL Qwen2.5-7B DeepScaleR GRPO
checkpoints    step_32,64,96,128,160,192,224,256
dataset        HuggingFaceH4/MATH-500 (500 problems)
samples        16 / problem / checkpoint
temperature    0.6
top_p          0.95
max tokens     8192 initially; rerun truncation-heavy cases at 16000
```

The seed paper publicly releases these checkpoints and uses temperature 0.6 / top-p 0.95 in its sampling pipeline.

## Answer scoring

Primary scorer must be aligned with the seed stack:

1. official `Temperal_sampling/prime_math.compute_score` rule/sympy scorer;
2. for rule-negative cases with an extractable candidate, Qwen2.5-32B-Instruct fallback (`--method hybrid`).

Before freezing F/N/S, manually audit a stratified random sample of at least:

```text
25 rule-positive
25 judge-rescued
25 final-negative
```

If disagreement > 5%, fix the scorer and rescore **all** checkpoints before classification.

## Integrity checks

Primary classification requires:

- all 8 expected checkpoint orders present for a problem;
- `>=16` scored samples at every checkpoint;
- exact same problem IDs / gold answers across checkpoints;
- no truncated generation rate large enough to differ systematically by checkpoint.

Do not label missing checkpoints as `never_correct`.

## Frozen states

At checkpoint t:

```text
C robust-correct  pass_rate >= 0.75
W robust-wrong    pass_rate <= 0.125
U uncertain       otherwise
```

Wilson 95% intervals are reported only as diagnostics. Do not change group membership to whichever CI definition looks better.

Groups:

### F — robust forgotten

```text
final = W
at least one earlier checkpoint = C
old checkpoint = latest earlier C
```

### N — never-correct

```text
all 8 checkpoints = W
```

### S — stable-correct

```text
final = C
at least one earlier checkpoint = C
```

## G-1 pass condition

MATH-500 must yield at least:

```text
F >= 50
N >= 50
S >= 50
```

If `F < 50`:

1. do not relax `.75/.125`;
2. run OlympiadBench under the same definition if desired;
3. if still sparse, primary topic is not supported strongly enough for re-entry and is stopped.

The exploratory `analyze_state_dynamics.py` may then describe what dynamics *do* dominate, but that cannot rescue this topic.

---

# G0 setup — freeze traces and controls before final-model intervention

## Old-self trace

For each F/S item:

1. use the **latest robust-correct checkpoint** fixed by G-1;
2. among its correct sampled completions, choose the shortest valid nonempty correct trace deterministically;
3. freeze it before any re-entry result exists.

Why shortest? It is a simple deterministic rule and reduces the chance that success is caused merely by giving an extremely long solution.

## Final-wrong trace

For F:

- from final checkpoint incorrect samples, choose shortest valid nonempty wrong trace deterministically.

## Other-correct / never-correct solution

Primary other route is the canonical MATH-500 worked `solution`:

- F: `other_correct_trace`;
- N: `verified_correct_trace`.

This gives a fixed verified route that was not chosen from G0 outcomes.

## F ↔ N matching

One-to-one greedy match before G0 using:

1. same MATH subject when available;
2. level;
3. prompt length proxy.

Both groups already satisfy final pass rate `<= .125`.

Do not rematch based on which controls give nicer rescue curves.

---

# G0-A — Re-entry experiment

## Prefix levels

The partial trace is appended directly after the **assistant generation prompt**. It is not embedded as a user-provided hint. This is essential: the experiment tests continuation from a former generation state, not ordinary instruction following.

Primary fractions on **old-self reasoning-step sequence**:

```text
10%
25%
50%
```

`0%` baseline is emitted exactly once per problem.

### Step boundaries

Automatic parser uses explicit line/paragraph boundaries, then sentence-like boundaries as fallback. Before G0:

- manually audit 30 F old-self traces;
- if >10% of cut points are clearly inside malformed/math fragments, improve the deterministic splitter **before any G0 inference** and rebuild all prefixes.

Do not hand-fix individual items after seeing outcomes.

## Token-budget matching

For each F and fraction k:

1. compute tokenizer-token count of the F old-self prefix;
2. truncate `other_correct` and `final_wrong` at the nearest complete step under approximately that budget;
3. use the same F budget for its matched N `verified_correct` prefix.

Primary tolerance:

```text
absolute token-budget ratio error <= 0.30
```

Items/conditions outside tolerance are excluded before inference and reported.

## Answer leakage

Automatically reject prefixes containing:

- `\\boxed{...}` / `\\fbox{...}`;
- explicit `final answer`, `answer is`, equivalent final-answer markers containing the gold answer.

Do not ban every raw occurrence of the gold number: it can be a legitimate intermediate value.

Before G0, manually audit at least **100 generated prefixes** across sources/fractions. If any direct answer leakage is found, fix the deterministic leak detector and rebuild every prefix.

## Conditions

```text
F baseline
F oldself
F other_correct
F final_wrong
N baseline
N verified_correct
S baseline
S oldself
```

Primary sampling:

```text
8 rollouts / request
temperature 0.6
top_p 0.95
```

Given abundant GPUs, run 16 rollouts/request as a robustness pass **after** the 8-sample primary result, without changing conditions/fractions.

## Primary estimands

For condition c and fraction k:

\[
R_c(k)=P(\text{correct}).
\]

### 1. Absolute rescue

\[
R_{F,oldself}(k)-R_{F,baseline}.
\]

Necessary but not sufficient.

### 2. Route-specificity

\[
\Delta_{route}(k)=R_{F,oldself}(k)-R_{F,othercorrect}(k).
\]

### 3. Wrong-route control

\[
\Delta_{wrong}(k)=R_{F,oldself}(k)-R_{F,finalwrong}(k).
\]

### 4. History-specificity

For frozen F/N pairs with the same prefix token budget:

\[
\Delta_{history}(k)=R_{F,oldself}(k)-R_{N,verifiedcorrect}(k).
\]

All confidence intervals cluster/bootstrap over **problem IDs**; F/N comparison clusters over matched pair ID.

## Primary interpretation

### Strong lost-entry pattern

Prefer to see all of:

1. short old-self prefix strongly rescues over baseline;
2. old-self > wrong prefix;
3. old-self > other-correct or at least produces an earlier/steeper rescue curve;
4. F old-self > matched N correct-prefix rescue;
5. G0-B old-route likelihood remains favorable.

Do not require every inequality to reach a magic p-value in a 50-item pilot. Direction, effect size, bootstrap CI, and replication across fractions matter.

### Broad guidability, not old-route retention

```text
oldself ≈ other_correct
both >> baseline
F ≈ N after matched correct prefixes
```

Conclusion: robust forgotten problems can still be guided, but there is little evidence for route-specific historical access. This is scientifically useful but weakens the original "old route" story.

### Partial erosion

Only 50% prefixes rescue; old-route suffix NLL is increasingly bad.

### Genuine route loss pattern

Even 50% non-leaking prefixes fail; old-route suffix NLL resembles N controls while S control is healthy.

---

# G0-B — Old-route continuation likelihood

Run on F/S old-self traces and N verified-correct traces.

Use the **final checkpoint** and its actual chat template. Score per-token suffix NLL after prefix fractions:

```text
0, 10%, 25%, 50%
```

Important:

- score assistant tokens only;
- same user prompt formatting as generation;
- per-token NLL, never total NLL;
- plot the full fraction curve;
- do not select a favorable layer/probe/position (there are none in this measurement).

Interpretation is comparative. Low NLL alone does not prove a human-like stored memory.

---

# G1 — Relearning savings (secondary)

Only after G0 is complete; its role is triangulation, not rescue.

Start from identical final checkpoints and use equal-size F/N sets:

```text
F: old correct solutions of robust forgotten items
N: verified solutions of matched never-correct items
```

Match initial final-model solution NLL as closely as practical; otherwise easier N examples create a trivial bias.

Use LoRA first for speed, then full-parameter confirmation only if the signal is large.

Fixed exposure checkpoints, e.g.:

```text
0,1,2,4,8 corrective exposures/item
```

Measure:

- generation success;
- verified-solution NLL;
- exposures-to-recover criterion.

A savings effect is consistent with a residual learning trace, but it does not by itself localize the mechanism.

---

# Confirmation / anti-selection rules

The initial MATH-500 run can be split deterministically by stable hash:

```text
60% pipeline/discovery
40% locked confirmation
```

Discovery may fix bugs in answer checking, step splitting, or leakage detection. It may **not** be used to choose new prefix fractions or replace controls.

After frozen confirmation fails, do not rescue by:

- loosening robust-state thresholds;
- selecting a different old checkpoint because it works better;
- trying many prefix fractions and reporting the best;
- switching to hidden-state probes;
- selecting only easy-to-rescue F items;
- replacing `other_correct` with a custom teacher chosen after outcomes;
- redefining "retention" as any small hint benefit.

---

# Predetermined exploratory branch if the premise is wrong

`analyze_state_dynamics.py` is allowed to report:

- fraction of `C→W`, `W→C`, `C→W→C`, repeated flips;
- when robust flips occur;
- whether temporal training looks monotonic or state-volatile.

Possible natural follow-up:

> **Are reasoning skills learned monotonically, or do individual problems repeatedly enter and leave competence during RL?**

This is **not Topic 05**. If the state-sequence audit is striking, register it separately and perform a new collision search before running targeted experiments.
