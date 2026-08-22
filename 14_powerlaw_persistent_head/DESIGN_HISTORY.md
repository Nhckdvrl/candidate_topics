# Design history

## v1 — balanced 120 cyclic maps

Initial implementation used all 120 cyclic rank shifts. Slow traversed shifts smoothly; Fast randomized their order. It exactly balanced every skill×rank occupancy and used the same keyed block multiset.

## v2 — exact A/B batch multiset, long persistence

Retired v1 **before Topic-14 scientific results were observed**.

Reasons:

1. The primary variable is persistence duration, not perfect long-run rank equality. Equalizing all 120 ranks forced the persistence interval to be short relative to the seed paper's stage-wise learning timescale.
2. A monotone cosine LR made temporal order inseparable from which mapping was seen at high vs low LR.
3. Independent arm warmups wasted compute and made branch identity an assumption rather than an audited fact.

v2 therefore uses one shared uniform warmup checkpoint, two fixed disjoint-head maps, deterministic batch-key matching, Slow/Fast reordering only, and a constant post-warmup LR.

No Topic-14 outcome data motivated this amendment.
