# 14 — Does Power-Law Learning Need a Persistent Head?

**Status:** VALIDATION v3 IMPLEMENTED — exact same-data temporal-order test, second-pass audited

## Scientific question

> When a power-law skill distribution makes compositional learning possible, is it enough that the current training stream is asymmetric, or must the **same skills stay high-frequency for long enough** to become a scaffold for the rest?

The seed work, **The Power of Power Law: Asymmetry Enables Compositional Reasoning** (ICML 2026 Spotlight), shows a striking S5 state-tracking phenomenon: a power-law skill distribution can make a compositional task learnable where uniform sampling struggles. Its proposed dynamics have a natural temporal reading: head skills are learned first, then tail learning accelerates.

The missing question is not whether power law works. It is whether **persistence of head identity is causal**.

## The decisive intervention

The primary experiment deliberately uses the simplest possible identification:

```text
same learner
same optimizer state
same model architecture
same finite training minibatch multiset
same A/B power-law maps
same number of A/B batches
same constant post-branch LR
same frozen uniform evaluation panel

ONLY the temporal order changes
```

For each replication seed, every power-law minibatch is deterministically keyed by `(seed, map_id, occurrence_id)`.

Slow:

```text
A0 A1 ... A(P-1) B0 B1 ... B(P-1)
```

Fast:

```text
A0 B0 A1 B1 ...
```

Therefore Slow and Fast receive **byte-identical finite training data as a multiset**. The intervention is only how long one head identity persists before the other takes over.

Map A is a frozen random rank→skill permutation. Map B is the same ranking shifted by 60 skills, which guarantees disjoint top-20% heads without searching for a favorable pair. The base mapping is predeclared to vary across replication seeds (`1729 + 1009*seed`), so five seeds do not merely repeat one arbitrary A/B identity assignment.

## Why the clean core starts from a shared warmup

For `pilot` and `full`, each seed first trains **one** 1000-step uniform warmup checkpoint. Uniform / Static / Slow / Fast then branch from the exact same model **and AdamW state**.

This is an identification device, not a claim of paper-faithful reproduction. It prevents random initialization and optimizer-history differences from contaminating Slow-vs-Fast and avoids paying warmup four times.

The post-branch LR is constant at `2e-4`. This removes the otherwise serious confound in which reordering the same batches also changes which head is seen at high vs low cosine LR.

The seed paper's stage-wise S5 transition occurs on a tens-of-thousands-step scale, so the locked persistence windows are intentionally long enough to give the hypothesis a fair chance:

- pilot: 80k core steps, 40k per Slow phase;
- full: 160k core steps, 80k per Slow phase.

## Testbed

- group: S5, 120 permutation skills;
- 4-hop composition;
- each permutation is represented by 5 symbols;
- direct prediction of the final 5-symbol composed permutation;
- Transformer encoder, 4 layers, `d_model=256`, 8 heads, 4× FFN, dropout 0;
- AdamW, LR `2e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `1e-6`;
- batch size 256;
- power-law exponent `alpha=1.5`;
- fixed uniform evaluation panel, global eval seed `424242`.

The exact 5-token sequence accuracy is the scientific observable. Token accuracy and CE loss are diagnostics only.

## Validation ladder

```bash
python -m pytest -q
python audit_schedule.py --profile pilot --seeds 0
bash run_gate.sh smoke 0
bash run_gate.sh pilot 0
bash run_gate.sh full 0,1,2,3,4
```

### Smoke

Engineering only. It must never be interpreted scientifically.

### Pilot

One seed, 80k core steps. It can expose gross bugs or a very large effect, but the analyzer always returns `PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE`.

### Full G0

Five locked paired seeds `0,1,2,3,4`. This is the first stage allowed to answer the question.

First verify the **clean-regime prerequisite**:

- median `Static - Uniform` exact-accuracy AUC >= 0.03;
- positive in at least 4/5 seeds.

This does not exist to make a positive result harder. It asks whether the seed phenomenon is alive in the deliberately clean branch/flat-LR regime. If it is not, Slow/Fast cannot be interpreted because neither temporal schedule has been given a regime in which power-law asymmetry is known to help.

After prerequisite pass:

- median `Slow - Fast >= 0.10` and >=4/5 positive → `PASS_PERSISTENT_HEAD_HELPS`;
- median `Slow - Fast <= -0.10` and >=4/5 negative → `PASS_RAPID_SWITCHING_HELPS`;
- `|median| <= 0.03` and >=4/5 seeds individually have `|gap| <= 0.06` → `KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT`;
- otherwise → `INCONCLUSIVE_FIXED_PROTOCOL_NO_TUNING`.

A small average created by large positive/negative seed effects is **not** called a null.

## If the clean anchor is weak: do not kill the question

A weak clean-core Static-vs-Uniform anchor now returns:

```text
CORE_ANCHOR_WEAK_NO_PERSISTENCE_CONCLUSION
```

That is deliberately not a scientific kill. The clean protocol changed two things relative to the seed setup for identification quality: common uniform data warmup and constant post-branch LR.

Run the separate near-paper anchor diagnostic:

```bash
bash run_gate.sh paper_anchor 0,1,2
```

`paper_anchor` is anchor-only (`Uniform` and `Static`):

- random initialization / empty optimizer at step 0;
- each arm sees its own distribution from the first training batch;
- 200k optimizer steps;
- 1000-step **LR** warmup to `2e-4`;
- cosine decay to `0.1×` peak;
- fp16 + GradScaler on CUDA by default.

Interpretation:

- paper anchor also fails → `TECHNICAL_SEED_REPRODUCTION_FAILED_DEBUG_BEFORE_SCIENCE`; inspect implementation/testbed before making any Topic-14 claim;
- paper anchor reproduces but clean anchor fails → the clean identification regime does not preserve the seed effect; Topic 14 is **unresolved**, not falsified;
- clean anchor passes → Slow/Fast is a valid direct test of persistence.

The paper anchor cannot rescue a weak Slow/Fast result after the clean prerequisite has passed.

## Integrity gates

Scientific interpretation is forbidden unless all paired runs agree on protocol version, profile, model shape, precision, optimizer hyperparameters, alpha, effective mapping seed, stream seed, evaluation seed, branch digest, metric grid and run signature.

Slow/Fast additionally require:

- identical multiset SHA-256;
- different temporal SHA-256;
- Slow max same-map run = locked phase length;
- Fast max same-map run = 1;
- deterministic keyed batches regenerate byte-identically.

Completed/resumed outputs carry protocol signatures so stale files from older Topic-14 versions are rejected rather than silently mixed.

## Resume and GPU behavior

`RESUME=1 bash run_gate.sh ...` now performs real checkpoint resume, restoring model, AdamW and fp16 scaler state. It is not just a "skip if done" flag.

`launch_grid.py` respects an existing `CUDA_VISIBLE_DEVICES` mask. With fewer GPUs than arms it runs waves, one arm per visible GPU, rather than oversubscribing a card. No distributed training or cross-node communication is needed.

## Conditional G1

Only after a replicated G0 temporal-order effect, test intermediate persistence lengths while preserving the exact same A/B batch multiset. `h=1` is already Fast and `h=P` is already Slow, so G1 only needs intermediate points. A monotone response would estimate a genuine curriculum timescale.

G1 is not allowed to rescue G0.

## What counts as a successful topic?

The goal is not to manufacture a null. Any large, replicated outcome is scientifically useful:

- Slow > Fast: persistent head identity is part of the power-law curriculum mechanism;
- Slow ≈ Fast with a healthy power-law regime: local asymmetry is sufficient and a stable head is unnecessary;
- Fast > Slow: rapid redistribution of privileged skills is actually beneficial, contradicting the simple scaffold picture.

The topic should be abandoned only when the relevant prerequisite is healthy and the same-data/different-order contrast is genuinely small, or when the seed phenomenon cannot be technically reproduced at all.

Do not react to weak data by sweeping alpha, map pairs, head definitions, architectures, hidden-state probes, primary metrics or decision margins.
