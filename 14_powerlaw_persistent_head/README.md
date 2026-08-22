# 14 — Does Power-Law Learning Need a Persistent Head?

**Status:** VALIDATION v2 IMPLEMENTED — exact matched-multiset temporal-order test

## Scientific question

> When a power-law training distribution makes a compositional task learnable, is local asymmetry enough, or must the **same skills remain high-frequency for long enough** to become a scaffold?

The seed work, **The Power of Power Law: Asymmetry Enables Compositional Reasoning** (ICML 2026 Spotlight), shows that replacing uniform skill frequencies with a power law can make 4-hop S5 state tracking learnable. Its interpretation contains two ingredients: immediate symmetry breaking near initialization and stage-wise head-to-tail learning. The missing identification is temporal. The paper randomizes skill order once and keeps it fixed; it does not hold the finite training data fixed while changing how long head identity persists.

## Why v2 is stronger

The first implementation used 120 cyclic rank shifts. That exactly equalized long-run skill×rank occupancy, but it weakened the object we actually care about: the so-called slow schedule could still change head identity much faster than the tens-of-thousands-step learning transition in the seed paper, and a cosine LR entangled data order with optimizer time.

v2 removes both problems **before observing Topic-14 scientific data**.

For each seed:

1. Train one shared 1000-step **uniform warmup** checkpoint and save both model and AdamW state.
2. Branch Uniform / Static / Slow / Fast from that exact checkpoint.
3. Freeze two deterministic power-law rank→skill maps, A and B. B is a half-cycle shift of A, so their top-20% heads are disjoint without searching for a convenient pair.
4. Generate every power-law training batch from an immutable key `(model_seed, map_id, occurrence_id)`.
5. Give Slow and Fast the **same actual finite batch multiset** in different temporal orders.

Slow:

```text
A0 A1 ... A(P-1) B0 B1 ... B(P-1)
```

Fast:

```text
A0 B0 A1 B1 ...
```

The primary post-warmup LR is constant. Therefore the Slow/Fast contrast changes only temporal persistence/order, not batch contents, counts, optimizer state, LR, or evaluation set.

## Testbed

- group: S5, 120 permutation skills;
- 4-hop composition;
- direct prediction of the final 5-token permutation;
- encoder Transformer, 4 layers, d_model 256;
- AdamW, LR 2e-4, weight decay 1e-6;
- batch size 256;
- power-law exponent alpha=1.5;
- uniform frozen evaluation panel with a single global eval seed.

The architecture details that are under-specified in the paper are frozen, not tuned. The Static-vs-Uniform prerequisite exists so a weak seed reproduction cannot be misread as evidence about persistence.

## Primary metric

**Normalized AUC of exact 5-token sequence accuracy** on the frozen uniform test set over fixed core compute.

Token accuracy and CE loss are diagnostics only. Exact accuracy is primary because the task is to recover the entire group composition, not some output coordinates.

## Gates

```bash
python -m pytest -q
python audit_schedule.py --profile pilot
bash run_gate.sh smoke 0
bash run_gate.sh pilot 0
bash run_gate.sh full 0,1,2,3,4
```

- Smoke is engineering only.
- Pilot uses one seed and always returns `PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE`.
- Full uses five paired seeds and is the first stage allowed to keep/kill the scientific claim.

### Frozen full decision

Prerequisite first: median `Static - Uniform` exact-AUC >= 0.03 and positive in at least 4/5 seeds.

Then:

- median `Slow - Fast >= 0.10` and >=4/5 positive -> `PASS_PERSISTENT_HEAD_HELPS`;
- median `Slow - Fast <= -0.10` and >=4/5 negative -> `PASS_RAPID_SWITCHING_HELPS`;
- |median| <=0.03 and >=4/5 seeds have |gap| <=0.06 -> `KILL_NO_TEMPORAL_PERSISTENCE_EFFECT`;
- otherwise -> `INCONCLUSIVE_FIXED_PROTOCOL_NO_TUNING`.

These are locked engineering triage margins, not p-values.

## Integrity checks

Scientific interpretation is forbidden unless:

- S5 algebra tests pass;
- Slow/Fast multiset SHA-256 matches exactly;
- Slow/Fast temporal SHA-256 differs;
- Slow max same-map run equals the predeclared phase length;
- Fast max same-map run is 1;
- a representative keyed batch regenerates byte-identically;
- all four arms share the exact same `branch_digest` (model + optimizer state);
- all seeds/arms use eval seed `424242`.

## What if the prerequisite fails?

Do not tune Slow/Fast. A separate near-paper 200k/cosine **anchor-only** diagnostic may be used to distinguish an implementation/regime failure from a genuine failure to reproduce the seed phenomenon. It cannot rescue a Slow/Fast null.

## Conditional G1

Only after a replicated G0 temporal-order effect, sweep persistence chunk length using the same A/B batch multiset, e.g. `h in {1,32,256,2048,P}`. This estimates a persistence timescale. G1 is not allowed to rescue G0.

## Kill discipline

Do not react to weak data by sweeping alpha, mapping pairs, head definitions, architectures, hidden-state probes, alternate metrics, or decision thresholds.

The topic is worth keeping only if a **simple same-data/different-order intervention** produces a large, seed-stable effect.
