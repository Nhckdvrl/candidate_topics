# Topic 12 G-0 report

**Scientific gate:** `INCONCLUSIVE_DO_NOT_TUNE`  
**Intervention informativeness:** `INFORMATIVE`

## Locked primary result

- Primary necessity I(l): **P(ablated wrong | baseline correct)**, averaged equally across MATH500 and GSM8K.
- Spearman rho(I, C_matched): **0.355**
- Paired item-bootstrap 90% CI: **[0.300, 0.402]**
- Kendall tau: **0.225**
- Spearman between residuals after removing predeclared quadratic depth trends: **-0.238**
- partial-rank depth diagnostic (descriptive): **-0.093**
- circular-shift p-value: **0.071**
- top-5 overlap: **1** (random expectation 0.89)
- MATH500-vs-GSM8K necessity-profile rho: **0.878**

## Locked robustness checks

- rho using legacy net accuracy drop as I: **0.358**
- rho against published four-task C_math: **0.315**
- rho(necessity, parser-fallback rate): **0.586**
- rho(necessity, truncation rate): **0.515**

## Task-level checks
- gsm8k: baseline acc=0.781, baseline solved n=200, rho(conditional I_task,C_task)=0.293, rho(net drop,C_task)=0.295
- math500: baseline acc=0.586, baseline solved n=150, rho(conditional I_task,C_task)=0.279, rho(net drop,C_task)=0.271

## How to read this without fooling ourselves

The primary necessity measure is conditional on baseline-correct items. It asks
whether a layer is required for competence that demonstrably exists before the
intervention. This prevents chance wrong->correct flips from cancelling real
damage. Net accuracy drop is still reported because it matches the older layer-
ablation literature, but it is no longer the identification target.

The locked depth-shape gate correlates deviations after each raw curve has had
a quadratic function of normalized depth removed. This deliberately asks whether
neighboring layers line up beyond a broad middle-layer profile. A separate true
partial-rank diagnostic is reported, but it is descriptive because a quadratic
model of ranks does not perfectly absorb a nonlinear U-shaped rank profile.

Ablation-induced parser failure and runaway generation are outcomes, not rows to
filter. But if >=25% of layers lose >=90% of baseline-solved items, or >=25% of
layers have >=50% parser/truncation failure, full deletion is declared too
destructive to rank functional necessity cleanly. That is a measurement failure,
not a negative scientific result; run the predeclared alpha=0.5 full sweep.

`INCONCLUSIVE_DO_NOT_TUNE` means do not search layer subsets, task weights,
metrics, or new ablation definitions. Fix only genuine engineering/protocol
mismatches, or stop. `DISSOCIATION_CANDIDATE` is not yet a paper-level law: it
needs independent-model replication because the published RL curve itself is a
finite experimental estimate.
