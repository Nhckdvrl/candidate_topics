# Archive Summary — Topic 15: Does Training-Time World Modeling Act Through a Predictive Policy State?

**Final status: ARCHIVED / KILLED AT MATCHED-TRAINING MEDIATION GATE — predictive state formed but was not used for action, and giving action extra non-shared capacity did not change that.**

Archived 2026-08-23 after two full matched future-on/future-off training runs (30,000 steps
each, 4 arms total) completed and were evaluated on held-out, episode-disjoint samples at
multiple checkpoints each. Both matched designs were run to their pre-registered or
user-authorized stopping points; neither was cut short favorably.

---

## 1. Scientific question

> When training-time future prediction helps a deployed robot policy, does it leave a
> predictive internal state that the policy actually relies on for action — or is future
> prediction only a training-time regularizer whose benefit does not route through a legible
> predictive representation?

Causal chain under test: `training-time future supervision (T) → predictive policy state (M) →
better action (Y)`.

## 2. What was run

| Stage | Design | Result |
| --- | --- | --- |
| G0 | Released Light-WAM checkpoints (`libero_spatial`, `libero_object`); WAM-adapter bypass vs. normal, episode-disjoint held-out probe | Adapters strongly causal for action (+135%, +8.8% action-loss increase on bypass) but add **zero** measurable held-out future-predictive information (CI crosses zero both times) |
| Isolated G1 | Matched future-on/off training, backbone LoRA **disabled**, 2.37M WAM adapters the sole trainable module shared by future and action losses | `T → M` replicated cleanly and stably (4/4 checkpoints, CI excludes zero). `M → Y` failed and got monotonically worse (relative action penalty 2.0%→27.6%). `ΔC` wrong-signed at all 4 points, CI never crosses zero |
| Capacity-restored G1 | Same design, LoRA **restored** to released spec but restricted to action-loss gradient only via exact gradient routing, giving action 87.5M non-shared parameters | Oscillated early (10k), converged back to the same failure by 20k–30k: no policy gain, `ΔC` wrong-signed, and the predictive-state effect that was stable in the isolated run essentially vanished (both arms statistically insignificant by 30k) |

Full numeric detail, every checkpoint, every CI: [`RESULTS.md`](RESULTS.md).

## 3. Why this counts as a real (not premature) kill

The isolated-G1 negative had one live, unresolved confound: with LoRA off, the 2.37M-parameter
adapters were the *only* trainable representation, forcing future and action objectives to
compete for one small bottleneck. That alone could explain a predictive-state-forms-but-doesn't-
help result without saying anything about whether such a state is ever useful. The
capacity-restored run exists specifically to remove that confound — action was given ~37× more
dedicated capacity than the bottleneck it was competing for — and the result did not change in
the hoped-for direction. If anything, restoring LoRA displaced the residual future-predictive
signal rather than letting it become useful, consistent with LoRA's larger capacity now doing
the representational work the adapters used to be forced into.

## 4. What this project does establish

`T → M` is real and robust across two independent designs: future/video supervision reliably
makes a small adapter pathway's state more linearly predictive of the real future in the
checkpoint's own future-training latent space, on a parameter-free probe, with an
episode-disjoint held-out split, and with a genuine module-level bypass intervention. "Future
information is decodable" and "future information is useful to the deployed policy" are
demonstrably different claims — that gap is the one clean, general finding this project
produced, independent of the final negative.

## 5. What was not attempted, per the project's standing rule

No SAE, PCA/CCA, learned projector, adapter-width/layer/rank/threshold sweep, additional λ
values, or alternative probe was used to try to rescue the negative. No claim is made about
whether an action-conditioned world model exists anywhere in Light-WAM, or about closed-loop
success (all action measurements are offline demonstration action loss).

## 6. Where the full record lives

- [`README.md`](README.md) — original scientific design and decision rules
- [`VALIDATION_AUDIT.md`](VALIDATION_AUDIT.md) — pre-run audit of the G0/G1 identification
- [`RESULTS.md`](RESULTS.md) — full numeric results for G0, isolated G1, and capacity-restored G1
- `g0_lightwam.py`, `g0_core.py`, `g0_feature_delta_check.py` — G0 implementation
- `g1_make_init.py`, `g1_make_config.py`, `g1_train.py`, `g1_evaluate.py` — matched-training
  infrastructure shared by both G1 designs
- `g1cap/`, `patches/lightwam_freeze_proprio.patch` — capacity-restored design and the local
  Light-WAM trainer edits it required (proprio freeze, LoRA action-only gradient routing,
  grad-norm logging, DeepSpeed-plugin guard)
- `tests/test_lora_gradient_routing.py` — algorithmic verification of the LoRA gradient routing
- `results/` — machine-readable results for every evaluated checkpoint
