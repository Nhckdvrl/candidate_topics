# Coverage Collapse vs. Latent Viability of Suppressed Branches

The completed paper-exact G0-A run is documented in [G0_RESULTS.md](G0_RESULTS.md). It returned `stop_or_redesign`, so the protocol correctly did not run G0-B latent probing.

## Status

**Candidate topic — falsification-first G0 implemented.**

The question is deliberately narrower than "does SFT cause representation collapse?" or "do hidden states contain alternative reasoning paths?" Both neighboring claims already have substantial prior work.

The surviving question is:

> **When SFT causes sampled reasoning coverage to shrink at a known binary fork, does a branch-viability signal learned at an earlier high-coverage checkpoint remain usable at the late checkpoint even when the native first-fork readout and the latent readout disagree?**

The key words are **same SFT trajectory**, **exact graph-ground-truth viability**, **frozen early readout**, **matched target counterfactual**, and **label-free disagreement selection**.

---

## Why this question is still potentially interesting

### Seed phenomenon

Nguyen et al., **Why Do Reasoning Models Lose Coverage? The Role of Data and Forks in the Road** (arXiv:2605.17026, 2026) show that reasoning SFT can increase `pass@1` while reducing `pass@k`. In their controlled `arithchain_2_10` Graph Branching task, the first reasoning step is an exact two-way fork. They show that SFT increasingly polarizes model confidence at this decision point, including confidently incorrect decisions, and that prefix perturbations can recover some lost coverage.

That paper already supports a **behavioral suppression** story. Therefore this project is not allowed to claim novelty from "coverage can be recovered" alone.

### Neighboring hidden-state result

Zur et al., **Are Language Models Aware of the Road Not Taken? Token-level uncertainty and hidden state dynamics** (2025) show that hidden activations can predict future outcome distributions and contain information about alternative reasoning paths.

Therefore `probe AUC > 0.5` is not enough.

### Neighboring branch-viability diagnostic

Liu et al., **When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning** (arXiv:2605.21606, 2026) use forced alternative tokens and continuation success as a behavior-level branch-viability diagnostic.

Therefore forcing a branch and observing that it can still solve the problem is also not enough.

### Probe caveat

Recent work such as **Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States** (arXiv:2606.02907, 2026), together with the older probing-control literature, makes a plain checkpoint-local probe especially weak evidence. A probe can exploit format, lexical identity, or a representation geometry that has nothing to do with the claimed computation.

This is why the revised G0 uses a matched target flip and freezes the readout learned at the reference checkpoint.

---

## Critical audit corrections

### 1. The previous `output-wrong rescue` gate was logically invalid

The old gate selected examples where the native binary branch choice was known to be wrong and then asked whether a hidden probe could identify the correct branch.

In a binary fork with exactly one viable branch:

```text
native branch is wrong
=> the opposite branch is correct
```

Conditioning on "output is wrong" therefore leaks the answer structurally. It is useful for descriptive case analysis, but it cannot be the decisive statistical gate.

**Replacement:** select cases without labels:

```text
native readout is high-confidence
AND
native branch choice != frozen hidden-probe branch choice
```

Only after this label-free disagreement subset is fixed do we reveal graph ground truth and ask which readout wins.

### 2. A matched target-flip counterfactual is now mandatory

For every graph, keep **all equations, premise, candidate letters, and formatting fixed**, and change only the queried terminal from the leaf on one branch to the leaf on the other branch.

Then the correct first branch must flip.

Example:

```text
same graph:
p -> c -> ... -> d
p -> x -> ... -> y

original query: target d  => c-branch viable
counterfactual query: target y => x-branch viable
```

A real branch-viability signal should change direction under this intervention. A target-insensitive lexical shortcut should not.

### 3. The probe and candidate basis are frozen at the early reference checkpoint

The previous code trained a separate probe at every checkpoint. That only establishes checkpoint-local decodability and lets the decoder adapt to a reorganized representation.

The revised gate:

1. selects one layer using **reference checkpoint discovery data only**;
2. trains one low-capacity probe on reference `original + target_flip` discovery pairs;
3. freezes the probe;
4. freezes the **reference checkpoint candidate-embedding basis**;
5. applies that exact readout to the late checkpoint.

This is much closer to asking whether the earlier branch-viability representation survives.

### 4. Behavior-preflight problems are excluded from latent discovery/confirmation

The cheap behavior gate uses 200 problems. Those problem IDs are then excluded from the hidden-state experiment. The remaining 800 test problems are split 60/40 into discovery and confirmation.

This prevents behavior-based checkpoint selection from contaminating the latent confirmation set.

### 5. Paper/code learning-rate mismatch is handled explicitly

Nguyen et al. Appendix A.2 reports:

```text
Qwen2.5-0.5B Graph Branching SFT learning rate = 2e-5
```

but the pinned upstream `run_sft.sh` currently sets Qwen2.5-0.5B to `1e-5`.

A failed `1e-5` reproduction is therefore **not a valid kill**.

`run_train_paper_exact.sh` calls the pinned upstream training implementation directly with `2e-5`.

---

# Fast falsification pipeline

## 0. Prepare exact upstream snapshot

```bash
./prepare_upstream.sh
```

Pinned upstream commit:

```text
NNHieu/reasoning_forks
64bf9e3e86231bc6b52f2974ca285ad8aa8fc181
```

This regenerates the official 6,400 SFT / 1,600 RLVR / 1,000 test Graph Branching split and reconstructs exact first-fork labels plus the opposite terminal needed for target-flip counterfactuals.

## 1. Train paper-exact forward SFT

```bash
TRAIN_GPU=0 ./run_train_paper_exact.sh
```

Default output:

```text
external/reasoning_forks/runs/topic03_paper_exact/
qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1
```

Checkpoints:

```text
e01 = step 200
e02 = step 400
e04 = step 800
e08 = step 1600
e16 = step 3200
```

## 2. One-command G0

```bash
GPUS=0,1,2,3 ./run_g0.sh
```

If the paper-exact checkpoint trajectory is absent, `run_g0.sh` trains it first.

---

# G0-A — behavior premise

Default:

```text
200 problems
16 samples/problem
e01, e02, e04, e16
temperature 1.0
top_p 0.95
max_tokens 512
```

The early reference checkpoint is the non-late checkpoint with the best sampled `pass@8`.

Continue only if all are true:

1. paired bootstrap 95% CI lower bound for

   ```text
   pass@8(reference) - pass@8(e16)
   ```

   is `> 0`;

2. point drop is at least `0.03`;

3. paired bootstrap 95% CI lower bound for

   ```text
   first-fork entropy(reference) - first-fork entropy(e16)
   ```

   is `> 0`;

4. entropy drop is at least `0.05` nats;

5. first-branch parser coverage is at least `90%` at both reference and late checkpoints.

If this fails under the **paper-exact 2e-5 trajectory**, stop. Do not run hidden-state probes to rescue the topic.

---

# G0-B — latent branch viability

Only runs after G0-A passes.

For the 800 problems not used by G0-A, extract five conditions:

```text
reference / original target
reference / matched target flip
late      / original target
late      / matched target flip
reference / target blind
```

The target-blind control removes only the queried target identity.

### Probe

```text
feature = h_layer * reference_embedding(candidate_A - candidate_B)

StandardScaler
-> PCA <= 32
-> LogisticRegression(C=1)
```

Layer selection uses reference discovery only.

Probe training uses reference `original + target_flip` discovery pairs only.

No late-checkpoint fitting is allowed.

### Why target flip matters

For each confirmation graph:

```text
P(A viable | original query)
```

should move in the opposite direction when only the target is switched to the other branch leaf.

This directly tests whether the hidden score is tied to the actual queried branch relation instead of static graph/letter identity.

---

# G0-B automated continue / kill conditions

Continue to full confirmation only if **every** condition passes:

```text
reference paired counterfactual hidden AUC:
    95% CI lower > 0.70

reference target-flip direction accuracy:
    95% CI lower > 0.75

late frozen-probe paired AUC:
    95% CI lower > 0.60

late target-flip direction accuracy:
    95% CI lower > 0.60

target-blind frozen-probe AUC:
    |AUC - 0.50| < 0.10

label-free committed native/probe disagreements:
    >= 30 events

on those disagreements:
    hidden-readout win-rate 95% CI lower > 0.55
```

High-confidence commitment is defined by:

```text
abs(native candidate log-prob margin) >= 2.0
```

The disagreement set is chosen **without looking at correctness labels**.

---

# Interpretation

### Strong positive: access suppression

All behavior and latent gates pass.

Interpretation:

```text
SFT narrows sampled coverage and increases commitment;
a fixed earlier viability readout still transfers to the late state;
the signal reverses under a matched target intervention;
and when the frozen latent readout conflicts with a strongly committed native readout,
the latent readout wins more often than chance.
```

This is the result worth pursuing.

### Representation erasure / reorganization

Behavior gate passes, reference probe is valid, but frozen transfer fails.

This does **not** support the suppression story. A checkpoint-local late probe may still decode viability, but that is a different and weaker question. Do not silently retrain a late probe and call it the same result.

### No fork-linked coverage collapse

Behavior gate fails.

Kill this mechanism immediately.

### Counterfactual failure

Reference hidden AUC is high but target-flip direction is weak.

Treat the probe as shortcut-prone; kill/redesign.

### Target-blind positive

Kill/redesign. The probe is likely exploiting generator or lexical structure.

### Too few label-free disagreements

If the native and latent readouts almost never disagree under strong commitment, this experiment cannot separate "suppressed latent alternative" from "same decision represented twice." Kill this version rather than inventing more gates.

---

# Full confirmation only after G0 passes

Behavior:

```bash
NUM_PROBLEMS=1000 NUM_SAMPLES=64 TAGS=e01,e02,e04,e08,e16 \
RUN_ID=full_$(date +%Y%m%d_%H%M%S) \
./run_behavior_passk_forward.sh
```

Hidden trajectory:

```bash
./run_sft_dynamics_example.sh
```

For a paper-level claim, the next requirement is replication across at least one additional training seed or backbone. Do not pay that cost before the G0 result is clearly positive.

---

# Literature boundary

The project should **not** claim any of the following as its contribution:

- SFT reduces `pass@k`;
- decision points become overconfident;
- prefix perturbations restore coverage;
- hidden states can represent alternative futures;
- forced alternative branches can remain viable;
- generic effective-rank / anisotropy representation collapse.

The only defensible remaining contribution is the narrower one:

> **A branch-ground-truth, matched-counterfactual, cross-checkpoint test of whether an early branch-viability representation remains functionally more reliable than the late native fork readout when SFT coverage collapses.**

See [`VALIDATION.md`](./VALIDATION.md) for the statistical contract and [`SERVER_HANDOFF.md`](./SERVER_HANDOFF.md) for execution instructions.
