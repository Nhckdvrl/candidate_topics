# 14 — Does Power-Law Learning Need a Persistent Head?

**Status:** VALIDATION IMPLEMENTED — RUN GATES BEFORE PROMOTION

## The question

> When a power-law training distribution makes a compositional task learnable, does the benefit come from **instantaneous asymmetry** at each moment, or must the **same skills stay frequent for long enough** to become a scaffold for the rest?

This is a temporal question about the mechanism behind the power-law effect, not another test of whether power-law data can outperform uniform data.

The cleanest version is even more concrete:

> **If two training runs receive exactly the same multiset of training minibatch blocks, can merely reordering those blocks — so that high-frequency skill identity is persistent vs rapidly rotating — change whether compositional learning succeeds?**

## Why this question exists

The seed work, [*The Power of Power Law: Asymmetry Enables Compositional Reasoning* (ICML 2026 Spotlight)](https://arxiv.org/abs/2604.22951), reports that changing skill frequencies from uniform to power law can turn difficult compositional tasks such as 4-hop state tracking on \(S_5\) from effectively unlearnable into learnable.

Its mechanistic account contains two separable ideas:

1. **Stage I — immediate landscape asymmetry.** A power-law distribution creates large gradients for a small set of head skills and helps optimization escape a flat/pathological region near initialization.
2. **Stage II — head-to-tail scaffolding.** Head skills are learned first; once learned, they strengthen the signal for tail skills and accelerate the rest of learning.

The paper carefully checks power-law exponent, granularity, lexicographic/random/reversed skill order, and explicit curriculum. But in those experiments the mapping from probability rank to skill is **static within a run**. Random order means “randomize once, then keep it fixed,” not “change which skills are the head over training time.”

That leaves a direct identification gap:

> Does the power-law advantage require a temporally persistent head, as the Stage-II scaffold story suggests, or can a rapidly moving head work just as well because local asymmetry alone is enough?

## What is deliberately *not* being asked

This project is not about:

- finding the best power-law exponent;
- discovering a special semantic ordering of permutations;
- hidden-state probes or mechanistic interpretability;
- designing a new curriculum algorithm;
- showing yet again that Zipf sampling beats uniform sampling.

The first job is only to determine whether **temporal persistence of the head is a real causal variable**.

---

# Primary testbed: 4-hop \(S_5\) state tracking

The first validation uses the seed paper's cheapest strong testbed rather than the Qwen arithmetic experiment.

Locked task details:

- group: \(S_5\), 120 permutation skills;
- each permutation: 5 tokens;
- composition depth: 4 hops;
- input length: 20 tokens;
- vocabulary size: 5;
- direct prediction of the final composed permutation;
- loss on the final 5 positions;
- online generated data;
- encoder Transformer;
- 4 layers, hidden size 256;
- AdamW, \((\beta_1,\beta_2)=(0.9,0.999)\), \(\epsilon=10^{-8}\), weight decay \(10^{-6}\);
- batch size 256;
- peak LR \(2\times10^{-4}\), 1000-step warmup, cosine decay to 0.1× peak;
- seed-paper full budget: 200k optimizer steps.

The seed paper does not fully specify attention-head count / FFN width in the text. The implementation freezes these under-specified details at **8 heads / 4× FFN / GELU / dropout 0** and does not tune them. The static-power-law prerequisite gate exists precisely so a failed reproduction cannot be misread as evidence about persistence.

The `full` profile uses **199,920 steps**, essentially matching the reported 200k budget while preserving complete balanced 120-block cycles.

---

# The causal intervention

Let the 120 skills be assigned to probability ranks \(1,\ldots,120\), with the same frozen Zipf vector

\[
p_r \propto r^{-1.5}.
\]

A training **block** contains many optimizer steps. Inside every power-law block, the integer number of samples assigned to every rank is exactly the same across conditions.

We construct four conditions.

## 1. Uniform — negative anchor

Every skill is sampled uniformly.

Purpose: reproduce the seed regime that struggles on the composition task.

## 2. Static power law — positive anchor

A single random mapping from rank to skill is sampled once and held fixed for the whole run.

Purpose: reproduce the seed paper's random-order \(\alpha=1.5\) power-law advantage while avoiding a hand-designed easy-to-hard ordering.

## 3. Balanced slow rotation — high temporal persistence

Within one 120-block cycle, use all 120 cyclic rank-to-skill shifts exactly once, in smooth order:

`0, 1, 2, ..., 119`.

A skill therefore moves gradually through adjacent frequency ranks. It stays inside the top-frequency head region for a long contiguous interval before moving through the tail.

## 4. Balanced fast rotation — low temporal persistence

Use **the exact same 120 cyclic mappings once each**, but randomly permute their temporal order, rejecting unusually autocorrelated permutations.

A skill still occupies every rank exactly once per cycle, but its high- and low-frequency periods are temporally scrambled.

---

# Why the slow-vs-fast comparison is unusually clean

The implementation enforces all of the following **exactly**, not merely in expectation:

### Same total compute

Same optimizer steps, batch size, sequence length, model, optimizer and initialization.

### Same local power-law spectrum

Every power-law block contains the same exact integer histogram over probability ranks.

### Same long-run exposure for every skill

In both balanced schedules, every skill occupies every probability rank exactly once per cycle. Therefore every skill receives exactly the same total number of training occurrences.

### Same multiset of rank-to-skill mappings

Slow and fast both use the same 120 cyclic mappings in each cycle. Only their temporal order differs.

### Same multiset of actual minibatch blocks

For balanced schedules, the random stream for a block is keyed by `(cycle, rank-to-skill shift)`, **not by temporal block index**.

Therefore the block corresponding to a given mapping contains the same ordered minibatches in slow and fast. Across one cycle the two conditions receive the **same training blocks; they are simply reordered in time**.

### Same initialization and frozen evaluation data

For a given model seed, all conditions start from identical parameters and use identical frozen uniform evaluation sets.

The primary contrast is therefore:

\[
\boxed{\text{same training-block multiset} + \text{different temporal persistence}}
\]

This is substantially stronger than comparing two independently sampled datasets with only matched marginal frequencies.

---

# What is measured

## Primary metric

**Normalized area under the learning curve (AUC) of token accuracy on a frozen uniform test distribution, over a fixed optimizer-step budget.**

Why AUC:

- the scientific question concerns learning dynamics;
- it uses the full frozen curve rather than selecting a favorable checkpoint;
- it has a direct interpretation as learning faster / spending less of the training budget near chance.

Token accuracy is seed-paper faithful; chance is 0.20.

## Secondary metrics

- final uniform-test token accuracy;
- exact 5-token sequence accuracy (unconstrained random-output chance \(1/5^5 = 1/3125\));
- uniform-test cross-entropy;
- five fixed skill-rank-bin curves.

The rank-bin curves are **diagnostics**, especially for checking the seed paper's head-first pattern under static power law. They are not used to define the primary slow-vs-fast result.

No post-hoc threshold search, hidden representation measure, or cherry-picked checkpoint is permitted.

---

# The three substantive outcomes

## A. Slow > Fast

A persistent frequency advantage matters even when both runs receive the same total examples and the same block multiset.

This supports the idea that a skill must remain privileged long enough for a head-to-tail scaffold to form:

> **the curriculum is not only in the frequency distribution; it exists in time.**

A large separation here is the primary positive hypothesis.

## B. Slow ≈ Fast > Uniform

Rapidly changing local asymmetry is sufficient even though no skill owns the head in the long run.

That would favor the Stage-I / optimization-landscape interpretation:

> **instantaneous symmetry breaking can enable compositional learning without a persistent curriculum.**

This is also a strong result because the balanced schedules have uniform long-run per-skill counts.

## C. Static > Slow ≈ Fast ≈ Uniform

Neither balanced form reproduces the static power-law benefit.

Then the crucial ingredient is a genuine long-run frequency hierarchy, not merely local asymmetry or short-lived persistence.

That is a clear negative answer to the temporal-rotation hypothesis and the project should not be rescued with arbitrary schedule sweeps.

## D. Fast > Slow

Unexpected, but scientifically meaningful if seed-stable: rapid redistribution of head status helps more than persistence.

This would directly contradict the simple scaffold picture. It should be kept only if the static prerequisite is strong and the sign replicates across seeds.

---

# One important interpretation boundary: learning-rate schedule

With the seed-faithful monotone cosine schedule, reordering training blocks also changes which block appears at which learning rate. That is part of real temporal training dynamics, but a large slow-vs-fast effect alone should initially be stated conservatively as:

> **temporal ordering / persistence of the power-law head matters.**

Only **after a strong replicated separation** should one run the single pre-registered mechanism diagnostic of a constant learning rate. If the separation survives, the persistent-scaffold interpretation becomes much stronger; if it disappears, the phenomenon is an interaction between data order and optimizer schedule.

Locked diagnostic command (only after a positive replicated signal):

```bash
LR_SCHEDULE=constant WARMUP_STEPS=0 bash run_gate.sh confirm 0,1,2
```

This diagnostic is not allowed as a rescue after a null result.

---

# Validation gates

The experiment is deliberately staged to avoid spending full compute on a broken premise.

## G-0 — zero-GPU structural audit

Run:

```bash
python self_test.py
python audit_schedule.py --cycles 2
```

Required before any training:

- all 120 \(S_5\) skills are valid and composition algebra passes identity/associativity tests;
- every slow/fast cycle contains every cyclic mapping exactly once;
- skill × rank occupancy is exactly balanced;
- realized per-skill sample counts are exactly equal;
- slow has substantially larger lag-1 frequency autocorrelation than fast;
- slow has much longer contiguous head runs;
- the balanced schedules use the same keyed block multiset.

If any check fails: **do not train**.

## G-1 — smoke, code path only

```bash
bash run_gate.sh smoke 0
```

This is only an engineering test. `analyze.py` explicitly returns `SMOKE_ONLY_DO_NOT_INTERPRET` so a 120-step run cannot accidentally kill or confirm the science.

## G-2 — cheap four-condition pilot

```bash
bash run_gate.sh pilot 0
```

`pilot` = 12,000 optimizer steps per condition. On a 4-GPU node, the four conditions are pinned one per GPU and run concurrently; no distributed training is used.

Purpose:

1. see whether the seed static-power-law advantage is already visible;
2. look for a large slow-vs-fast separation early;
3. avoid committing to 200k-step runs when there is no signal.

A pilot null is **not** enough to falsify the seed prerequisite because the published S5 setup trains for 200k steps.

## G-3a — prerequisite recovery, only if pilot anchor is weak

Do **not** run all four full-budget conditions yet. Run only the two anchors:

```bash
CONDITIONS=uniform,static bash run_gate.sh full 0
```

If static power law still fails to materially beat uniform at the near-paper 200k budget, stop:

> `KILL_PREREQUISITE_NOT_REPRODUCED`

Do not rescue by sweeping exponent, attention heads, FFN widths, rank orders or learning rates.

## G-3b — confirmation after a healthy pilot / anchor

```bash
bash run_gate.sh confirm 0,1,2
```

`confirm` = 96,000 steps per condition, three paired seeds.

The main evidence is:

- the full paired learning curves;
- sign consistency of `slow - fast` across seeds;
- normalized-AUC effect size;
- anchor validity.

## G-4 — near-paper full budget only when needed

```bash
bash run_gate.sh full 0,1,2
```

Use this only when the confirm run is scientifically ambiguous or when a full-budget statement is needed. It is **not** the default first experiment.

---

# Frozen engineering decision margins

`analyze.py` contains simple triage margins fixed before seeing real results:

- anchor mean `static - uniform` AUC ≥ **0.10**, and > **0.05** on every seed;
- strong persistence: |mean `slow - fast` AUC| ≥ **0.05** with the same non-zero sign on every seed;
- local-asymmetry signal: both balanced conditions beat uniform by mean AUC ≥ **0.05**.

These are not p-values and are not paper claims. They only automate early keep/kill triage. **They must never be swept after observing results.** Raw paired curves remain the evidence.

---

# Efficiency / environment

The implementation intentionally avoids a second heavy environment:

```text
numpy>=1.24
torch>=2.2
```

No Hugging Face, Ray, vLLM, torchrun or multi-node communication is required.

If the server environment already used by another topic has PyTorch and NumPy, reuse it. `matplotlib` is optional; if absent, plotting is skipped cleanly while all analysis still runs.

Four GPUs are used as **four independent single-GPU jobs**, one condition per GPU. With fewer GPUs, `launch_grid.py` automatically runs the conditions in waves.

---

# Files

| File | Purpose |
|---|---|
| `experiment.py` | S5 task, balanced temporal schedules, training and frozen evaluation |
| `self_test.py` | algebra + exact causal-design tests before GPU use |
| `audit_schedule.py` | human-readable schedule/persistence audit |
| `launch_grid.py` | 1–4 GPU independent-condition launcher |
| `analyze.py` | AUC contrasts, integrity checks and frozen gate report |
| `plot_results.py` | optional first learning-curve figure |
| `run_gate.sh` | one-command gate runner |
| `RUNBOOK.md` | server handoff / exact execution order |
| `requirements.txt` | minimal reusable dependencies |

---

# Kill discipline

Kill the project if the near-faithful static-power-law prerequisite cannot be reproduced.

If the prerequisite is healthy but slow and fast are indistinguishable and neither balanced schedule gives a scientifically interesting result relative to uniform, kill the temporal-persistence question.

Do **not** respond to a weak result by trying:

- many \(\alpha\) values;
- arbitrary rotation periods;
- many definitions of “head”;
- different semantic skill rankings;
- hidden-state probes;
- architecture sweeps.

A follow-up persistence-timescale sweep is justified **only after** a large frozen slow-vs-fast separation establishes that persistence is a real variable.

---

# Why a positive result would matter

The valuable statement is not “power-law data works.” The seed work already established that.

The valuable statement would be one of two much sharper principles:

> **Local distributional asymmetry can enable compositional learning even when every skill has exactly equal long-run frequency.**

or

> **The very same training data becomes more learnable when frequency advantage is temporally persistent, revealing a genuine curriculum timescale.**

Both distinguish a static property of the data histogram from a dynamical property of the learning process.
