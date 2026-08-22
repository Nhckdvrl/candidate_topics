# G0-v3 4×4 confirmation interpretation

The untouched 64-puzzle confirmation uses the same published UPO setting and the pre-frozen 4-transform spatial manifest.

- identity exact accuracy: `34/64 = 0.5313`
- isomorph exact accuracy: `148/256 = 0.5781`
- solve/fail flip rate: `116/256 = 0.4531`, puzzle-cluster bootstrap 95% CI `[0.3789, 0.5273]`
- both-exact pairs: `84` across `31` source puzzles
- mapped tau cluster mean: `0.1179`, bootstrap 95% CI `[0.0219, 0.2152]`
- tau excess over row-major null: `0.2074`, CI `[0.0597, 0.3556]`
- tau excess over boundary-first null: `0.1189`, CI `[-0.0148, 0.2650]`

The confirmation reproduces the discovery direction: exact spatial isomorphisms cause substantial outcome non-equivariance, while the conditional mapped order is positive and clearly above the row-major null in this setting. Evidence over the boundary-first null is directionally positive but remains compatible with zero at this sample size.

This is a valid 4×4 G0 result, not yet a 9×9 claim. The next paper-level stage should use a literature-established 9×9 competent object such as LLaDA2.0-flash-100B or a documented Sudoku-capable Dream variant, with a newly locked protocol and manifest.
