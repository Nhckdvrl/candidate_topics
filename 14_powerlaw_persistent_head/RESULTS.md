# Topic 14 — G0 result

Protocol: `topic14-v3-2026-08-22`, full profile, seeds `0,1,2,3,4`.

## Decision

`KILL_NO_MEANINGFUL_TEMPORAL_PERSISTENCE_EFFECT`

The clean power-law anchor was healthy: median Static−Uniform exact-sequence AUC was `0.9300079346`, with all five seeds positive. The temporal-order contrast had median Slow−Fast exact-sequence AUC `0.0095153809`; four of five seed-level gaps were within `±0.06`.

| seed | mapping seed | Static−Uniform AUC | Slow−Fast AUC |
|---:|---:|---:|---:|
| 0 | 1729 | 0.9300079346 | -0.0324844360 |
| 1 | 2738 | 0.9111183167 | 0.7105895996 |
| 2 | 3747 | 0.9416473389 | 0.0095153809 |
| 3 | 4756 | 0.9649276733 | 0.0340454102 |
| 4 | 5765 | 0.9054695129 | -0.0394935608 |

Seed 1 showed a large positive Slow−Fast gap, but the other four seeds were near zero or slightly negative. Under the pre-registered seed-level rule this is not a replicated persistence effect and is not treated as a null created by cancellation of opposing large effects.

All five seeds passed protocol/run-signature, branch-digest, metric-grid, mapping, schedule multiset, and temporal-order integrity checks. The uploaded result artifacts include metrics, configs, schedules, done metadata, branch metadata, and the analyzer decision. Large binary checkpoints remain local and are intentionally excluded.
