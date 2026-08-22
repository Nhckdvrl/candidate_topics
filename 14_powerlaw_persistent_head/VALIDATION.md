# Validation contract — Topic 14 v2

## Scientific object

The seed phenomenon is `power-law skill frequencies -> compositional learning`. Topic 14 asks whether **temporal persistence of head identity** is causal.

## Technical identity gates

Before scientific interpretation:

1. S5 contains exactly 120 valid permutations and composition-oracle tests pass.
2. Slow/Fast schedule multiset SHA-256 must match exactly.
3. Slow/Fast temporal SHA-256 must differ.
4. Fast max same-map run must be 1; Slow max run must equal half of the core budget.
5. A representative batch key must regenerate byte-identical inputs and labels.
6. All four arms within a seed must load the same `branch_digest` (model + optimizer state).
7. Evaluation seed is fixed globally (`424242`) across seeds and arms.

Any failure => technical invalidity; do not interpret.

## G0 prerequisite

Uniform and Static power-law anchors are run from the same branch checkpoint. Full prerequisite: median Static-Uniform exact-AUC >=0.03 and at least 4/5 paired seeds positive. This is deliberately directional and coarse: it only asks whether the chosen cheap testbed reproduces enough of the seed phenomenon to make persistence interpretable.

No persistence conclusion is allowed after prerequisite failure.

## G0 primary

The only scientific contrast is Slow vs Fast.

Matched exactly:

- model initialization and common 1000-step warmup;
- AdamW state at branch point;
- batch size and optimizer steps;
- maps A/B;
- number of A/B batches;
- every deterministic batch key and therefore the finite training-batch multiset;
- frozen uniform evaluation panel;
- constant post-warmup LR.

Changed: temporal order / persistence of A/B batches.

Pilot: 80k core steps, one seed, always `PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE`.

Full: 160k core steps, 80k Slow phases, five paired seeds.

Primary: exact-sequence-accuracy AUC. Diagnostics: token accuracy, CE loss, final values.

## Full decision

After prerequisite pass:

- median Slow-Fast exact-AUC >=0.10 and >=4/5 positive => persistent head helps;
- median <=-0.10 and >=4/5 negative => rapid switching helps;
- |median| <=0.03 and >=4/5 seeds individually have |gap|<=0.06 => no meaningful temporal-persistence effect in the locked regime;
- otherwise => inconclusive; do not tune the protocol.

## G1

Only after a clear replicated G0 effect. Sweep persistence chunk `h` while retaining the exact same A/B batch multiset. A monotone response estimates a curriculum timescale. G1 cannot rescue G0.

## Paper-schedule diagnostic

If the flat-LR Static-vs-Uniform prerequisite fails, a separate near-paper 200k/cosine anchor-only run may diagnose implementation/regime mismatch. It cannot be used to rescue or reinterpret a Slow/Fast null.
