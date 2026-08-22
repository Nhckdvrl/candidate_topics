# G0-v3 4×4 discovery interpretation

Protocol: `g0-v3-4x4-published`, public UPO 4×4 test setting, 64 discovery puzzles, four digit-preserving spatial transforms per puzzle.

The prerequisite is instantiated. Identity exact accuracy is `44/64 = 0.6875`, same-serialization repeat Kendall tau is `1.0`, and native scheduler agreement is `1.0` because the reproduction uses the published full-vocabulary scheduler without a grammar projection.

Outcome equivariance is not supported: isomorph exact accuracy is `157/256 = 0.6133`, with `101/256 = 0.3945` solve/fail flips. Puzzle-cluster bootstrap 95% CI for the flip rate is `[0.3164, 0.4727]`. These flips are retained as scientific outcome non-equivariance, not filtered from the study.

Among 116 both-exact pairs over 40 source puzzles, mapped order tau has puzzle-cluster mean `0.1107`, bootstrap 95% CI `[0.0240, 0.1991]`. The excess over the row-major positional null is `0.0542`, CI `[-0.0519, 0.1595]`; the excess over the boundary-first positional null is `0.1298`, CI `[0.0155, 0.2472]`.

Frozen interpretation before confirmation: the model's behavior has substantial serialization/sampler dependence at the outcome level. Conditional order preservation, if present, is modest and is not yet cleanly separated from the row-major null; it is more clearly above the boundary-first null in discovery. The untouched confirmation set will test this same statement.
