# Validation audit — Topic 15

This file records the scientific audit performed before running G0. The purpose is to prevent a later engineering agent from accidentally restoring an easier but non-identifying version of the experiment.

## Target claim

The identifiable first claim is:

> training-time world-model supervision may leave a predictive policy state, and the native pathway carrying that state may be causally relevant for action.

G0 is a screen for a clean native Light-WAM route. It is **not** a proof that the particular decodable future coordinates are themselves the causal action code.

---

## Audit pass 1 — representation / intervention semantics

### Problem found: action-pooler bias in the representation comparison

The initial implementation measured future decodability after passing both normal and pre-adapter states through the action expert's learned-query pooler.

That pooler is trained on adapted states. Therefore a normal > counterfactual probe difference could partly reflect the learned pooler's distribution preference rather than a difference in the underlying WAM state.

### Fix

The representation probe now uses the same **parameter-free per-layer token mean** on normal and bypass states.

The trained action expert is still used unchanged for the causal action measurement.

This keeps the two questions separate:

1. did the WAM pathway add future information to the state?
2. does removing that WAM pathway hurt action?

### Problem found: verdict overclaim

The old verdict name `PROMISING_NATIVE_MEDIATOR` was too strong.

Adapter bypass does not selectively remove future information. It removes the adapter transformation, which contains information learned from both future and action objectives.

### Fix

The positive verdict is now:

`PROCEED_TO_MATCHED_TRAINING`

A positive G0 means only that the same native adapter pathway is both more future-predictive and action-relevant.

---

## Audit pass 2 — exact Light-WAM computation graph

### Problem found: wrong future target resolution

Released Light-WAM trains its future/video objective in a spatially downsampled latent space (`video_latent_spatial_downsample_factor=2` in the released config).

The original G0 used the full-resolution cached VAE latent as the probe target.

That could create a false negative by scoring high-frequency details the world-model objective was never trained to predict.

### Fix

G0 now applies the checkpoint's own future-training latent preprocessing/downsampling before constructing the clean future-change target.

### Problem found: same-pass pre-adapter substitution was not a full adapter ablation

At layer 16, the cached `backbone_tokens` already contain the propagated effect of the layer-8 adapter. At layer 24 they already contain earlier adapter effects as well.

Therefore replacing each final readout slot with its same-layer pre-adapter tensor removes only the local residual; it does not remove the complete WAM-adapter computation.

### Fix

G0 now performs a second action-backbone pass on the identical observation with **all WAM adapter scales set to zero**.

This removes every adapter residual and its downstream propagation, then restores all original scales after the pass.

The code asserts that every bypassed adapter output equals its local pre-adapter input.

---

## Reviewer audit — what can still attack a positive result?

A positive released-checkpoint G0 can still be explained by action supervision, because the released WAM adapters are trained jointly by future loss and action loss. Backbone LoRA is also trainable.

Therefore G0 cannot answer the training-time mediation question by itself.

The decisive experiment must manipulate future supervision across matched trainings.

## Decisive matched-training identification

For a clean first causal test, construct two otherwise identical trainings:

```text
future-on:   lambda_video = 1
future-off:  lambda_video = 0
```

with:

- pretrained video backbone frozen;
- backbone LoRA disabled;
- proprio encoder initialized identically and frozen (or future gradients explicitly stopped into it);
- identical adapter/action-expert initialization;
- identical data order;
- identical optimizer / LR / step count;
- global gradient clipping inactive.

Why the proprio requirement matters: upstream `build_inputs` appends the learned proprio token to the context used by **both** future and action branches, so a trainable proprio encoder is otherwise another path from future loss to deployed action.

Why the clipping requirement matters: upstream training clips the gradient norm over all trainable parameters, so future-head gradients could otherwise rescale action-expert gradients even when no parameter is shared.

Why the initialization requirement matters: upstream currently applies `cfg.seed` inside `Wan22Trainer`, after model instantiation. Equal CLI seeds alone do not guarantee identical randomly initialized adapters/action expert.

Under the isolated matched configuration, the WAM adapters are the only trainable representation module shared by future and action objectives.

The decisive pattern is:

1. future-on > future-off in deployed policy performance;
2. future-on produces more future-predictive action state;
3. adapter bypass costs substantially more under future-on than future-off;
4. preferably, adapter bypass removes a substantial fraction of the future-on policy advantage.

The key interaction is:

```text
bypass_cost(future-on) - bypass_cost(future-off)
```

This supports the route-level mechanism claim:

> future supervision causally shapes a predictive adapter state that the policy relies on for action.

It still does not prove that a complete counterfactual world model exists inside the policy, nor that the particular linearly decoded future coordinates are the exact causal code. A content-selective intervention would be required for that stronger statement.

---

## Kill semantics

A weak G0 is a **project-level kill of the simple Light-WAM adapter route**, not a universal proof that no predictive state exists anywhere in the architecture.

That distinction is intentional.

If G0 is weak, do not rescue the topic with SAE / PCA / CCA / learned projectors / layer sweeps.

If G0 is strong, run exactly the matched future-on / future-off mechanism experiment above before making any mediation claim.
