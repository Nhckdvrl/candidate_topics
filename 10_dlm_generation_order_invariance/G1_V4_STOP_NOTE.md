# G1/v4 stop note

G1/v4 is stopped before epoch 10 and before any 9x9 symmetry generation.

The fixed ordinary competence checks did not instantiate a competent 9x9 solver:

| checkpoint | exact / 100 |
| --- | ---: |
| epoch 2 / step 24 | 6 |
| epoch 5 / step 60 | 3 |

Training losses were often near zero, while held-out exact solve stayed extremely low and declined from epoch 2 to epoch 5. Raw outputs frequently reproduced prompt instructions or emitted malformed, flat, or truncated matrices. This pattern is consistent with overfitting or format memorization rather than Sudoku generalization.

This is a G1 experimental-object/prerequisite failure, not a negative result for Topic 10's representation-equivariance hypothesis. No 9x9 symmetry result is claimed.

## Final project decision

Topic 10 is now **ARCHIVED**. The valid positive result remains the published 4x4 G0 setting, but that result alone is too toy-scale for the intended significance bar. A future 9x9 attempt would require a genuinely recovered, literature-aligned competent object rather than extending or tuning this run.

See `ARCHIVE_SUMMARY.md` for the complete postmortem and reopen condition.
