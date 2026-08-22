# Validation contract — Topic 14 v3

## Scientific object

Seed phenomenon:

```text
power-law skill frequency -> compositional learning advantage
```

Topic 14 asks one narrower causal question:

```text
does the identity of the high-frequency head need to persist in time?
```

The decisive G0 contrast is intentionally simple: **Slow and Fast receive the exact same finite training minibatch multiset, from the same learner/optimizer branch point, and differ only in temporal order.**

## P0 — technical identity gate

Before any scientific interpretation:

1. S5 must contain exactly 120 unique valid permutations and composition-oracle tests must pass.
2. For every replication seed, map A/B top-20% heads must be disjoint under the predeclared half-cycle relation.
3. Effective mapping seeds must be unique across the locked five replication seeds.
4. Slow/Fast schedule multiset SHA-256 must match exactly.
5. Slow/Fast temporal SHA-256 must differ.
6. Fast max same-map run must be 1; Slow max run must equal the frozen phase length.
7. A keyed batch `(seed,map_id,occurrence_id)` must regenerate byte-identical inputs and labels.
8. Within a seed, all four primary arms must share the same branch digest (model + AdamW state).
9. Paired configs must match on architecture, precision, optimizer, alpha, mapping seed, stream seed, eval seed and metric grid.
10. Protocol/run signatures must match completed or resumed outputs; stale outputs are technical failures.

Any failure => `TECHNICAL_INVALID_INTEGRITY_OR_INCOMPLETE`. Do not interpret.

## G-engineering — smoke

```bash
bash run_gate.sh smoke 0
```

Only checks the end-to-end path. Analyzer must return:

```text
SMOKE_ONLY_DO_NOT_INTERPRET
```

## G-pilot — cheap directional check

```bash
bash run_gate.sh pilot 0
```

One paired seed, 80k post-branch steps, 40k Slow phases. The output is always:

```text
PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE
```

The pilot may expose a broken premise or giant signal, but it has no authority to confirm or kill Topic 14.

## G0 — locked full test

```bash
bash run_gate.sh full 0,1,2,3,4
```

### Common clean branch

Each seed creates one 1000-step uniform-warmup checkpoint and saves model + AdamW state. Uniform / Static / Slow / Fast all branch from that exact state.

After branching the LR is constant `2e-4`. This intentionally removes learning-rate-time as a competing explanation for data-order effects.

### Slow/Fast exact matching

Matched exactly:

- initialization and 1000-step warmup history;
- AdamW state;
- model/hyperparameters;
- total optimizer steps and batch size;
- maps A/B;
- every deterministic A/B batch key;
- finite training-batch multiset;
- constant LR;
- frozen uniform evaluation panel.

Changed:

- **only temporal order / persistence of map identity**.

Full budget: 160k core steps. Slow uses 80k contiguous A then 80k contiguous B; Fast alternates A/B every step.

### Primary observable

Normalized AUC of **exact 5-token sequence accuracy** on the frozen uniform evaluation panel. Token accuracy, CE loss and final values are diagnostics only.

### Clean-regime prerequisite

Before Slow/Fast can be interpreted, the same clean regime must retain a visible Static power-law advantage over Uniform:

- median `Static - Uniform` exact-AUC >= 0.03;
- at least 4/5 seeds positive.

If this fails:

```text
CORE_ANCHOR_WEAK_NO_PERSISTENCE_CONCLUSION
```

This is **not** a Topic-14 scientific kill, because the clean shared-warmup/flat-LR regime intentionally differs from the seed training recipe.

### Primary decision after prerequisite pass

- median `Slow - Fast` exact-AUC >= 0.10 and >=4/5 positive → `PASS_PERSISTENT_HEAD_HELPS`;
- median <= -0.10 and >=4/5 negative → `PASS_RAPID_SWITCHING_HELPS`;
- `|median| <= 0.03` and >=4/5 individual gaps have `|gap| <= 0.06` → `KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT`;
- otherwise → `INCONCLUSIVE_FIXED_PROTOCOL_NO_TUNING`.

The near-zero rule is deliberately seed-level as well as median-level, so opposite large effects cannot cancel into a false null.

## Technical seed-reproduction diagnostic

Only when the clean Static-vs-Uniform prerequisite is weak, run:

```bash
bash run_gate.sh paper_anchor 0,1,2
```

This is not another version of Slow/Fast. It is an anchor-only implementation diagnostic designed to be much closer to the seed paper:

- Uniform and Static only;
- random initialization / empty optimizer at step 0;
- no shared uniform-data warmup;
- each arm sees its own distribution from the first batch;
- 200k optimizer steps;
- 1000-step LR warmup;
- cosine LR decay to `0.1x` peak;
- fp16 + GradScaler by default on CUDA.

Diagnostic pass requires a large directional Static advantage and nontrivial final Static competence. Outcomes:

- `PAPER_ANCHOR_REPRODUCED`: implementation/testbed can express the seed phenomenon; if clean anchor remains weak, the clean identification regime itself removed the effect and Topic 14 is unresolved in this setup.
- `TECHNICAL_SEED_REPRODUCTION_FAILED_DEBUG_BEFORE_SCIENCE`: debug task/model/training implementation before making scientific claims.

The paper diagnostic cannot rescue a weak Slow/Fast result after the clean prerequisite has already passed.

## Resume / stale-output contract

`--resume` restores the latest numeric checkpoint including model, AdamW and fp16 GradScaler state, then truncates/reuses metrics only up to that checkpoint step. Checkpoint protocol and branch signatures must match the requested run.

A completed output is skipped only if its frozen run signature matches the current request. Otherwise abort and use a fresh output directory rather than mixing protocols.

## G1 — persistence timescale, conditional only

Only after G0 shows a large replicated Slow/Fast effect may intermediate persistence chunks be run. Fast (`h=1`) and Slow (`h=P`) already provide the endpoints; G1 tests intermediate `h` while preserving the same A/B batch multiset.

A monotone response would estimate a curriculum timescale. G1 cannot rescue a G0 null or weak prerequisite.

## Discipline

This protocol is designed to give the hypothesis a fair chance, not to maximize rejection. Do not add post-hoc sweeps of alpha, mapping pairs, head definitions, architectures, hidden-state probes, metrics or thresholds after seeing weak data.

If an engineering bug is found, fix the bug while preserving the scientific invariants and document the pre/post-fix protocol. If a fix changes the scientific intervention itself, invalidate prior results and rerun from a fresh output root.
