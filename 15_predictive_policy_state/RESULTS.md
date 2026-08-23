# Topic 15 — Final results

**Question.** Does training-time world modeling act through a predictive policy state?

**Stages run.** G0 (released-checkpoint screen) → isolated G1 (matched future-on/future-off
training, LoRA off) → capacity-restored G1 (matched training, LoRA restored but restricted to
action-loss gradient). All three stages executed to their pre-registered or user-authorized
stopping points. No stage was cut short favorably and no stage was extended past its rule.

**Final verdict.**

```text
MECHANISM NOT SUPPORTED — ARCHIVE
```

Training-time future/video supervision reliably makes the shared WAM-adapter state carry more
linearly decodable future information. That predictive state does not help, and in the two
matched-training experiments where it was cleanly isolated, it either provided no significant
benefit or made the deployed policy measurably worse. The reversed mediation sign — action
depending on the adapters *more* when future supervision is off, not on — held from the very
first matched run, survived removing the shared-adapter-bottleneck confound, and had not
reversed direction by the last authorized checkpoint of the last authorized experiment.

---

## 1. Timeline

| Experiment | Design | Steps | Outcome |
| --- | --- | --- | --- |
| G0 | Released Light-WAM checkpoints, adapter bypass, held-out episode-disjoint probe | n/a (inference only) | `libero_spatial`: adapters cause +135% action loss on bypass, 0 future gain. `libero_object`: same pattern, +8.8% / 0 future gain. `libero_10`: never completed — HF cache download stalled indefinitely and was abandoned; does not change the verdict, since two suites already independently replicated the same pattern. |
| Isolated G1 | Matched future-on/off training. Backbone frozen, **LoRA disabled**, WAM adapters (2.37M) the only trainable module shared by future and action losses, proprio frozen, clipping inactive, shared init checkpoint | 30,000 (both arms) | Action-loss gap and ΔC both stably wrong-signed at every one of 4 evaluation points (2.5k/10k/20k/30k). Relative action penalty grows monotonically 2.0%→27.6%. |
| Capacity-restored G1 | Same matched design, **LoRA restored** to the released spec (30 layers, rank 64) and given exact-gradient routing so it receives action-loss gradient only — the WAM adapters remain the sole route by which future supervision can reach the deployed policy | 30,000 (both arms) | Oscillated at 10k/20k, but by 30k all three gates fail again: no policy gain, no significant predictive-state gain in either arm, ΔC wrong-signed. |

---

## 2. G0 — released-checkpoint screen

Full detail in the G0 section retained below (§9). Summary: on two independently downloaded
released checkpoints (`libero_spatial` step 55000, `libero_object` step 12500), the explicit
WAM-adapter pathway has a large, statistically clear causal effect on action prediction, but
adds no measurable held-out future-predictive information relative to full adapter bypass.
Verdict both times: `ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN`.

This screen alone could not settle the training-time question, because the released
checkpoint's bypass condition still contains backbone LoRA, itself trained by the future
objective — a second, unidentified route. That is why G1 was run rather than archiving at G0.

---

## 3. Isolated G1 — matched training, LoRA off

### Design

Two runs from one shared initialization checkpoint, differing in exactly one hyperparameter:

```text
future-on:   lambda_video = 1.0
future-off:  lambda_video = 0.0
```

Held fixed in both arms: frozen pretrained video backbone, **backbone LoRA disabled**
(`use_backbone_lora=false`), proprio encoder frozen (`LIGHTWAM_FREEZE_PROPRIO=1`), global
gradient clipping threshold set to `1e9` (never activates; observed grad norms ~0.6–10),
identical seed/batch/LR/schedule/data order, identical action-expert and adapter
initialization (verified by state-dict hash). Under this configuration the 2.37M-parameter
WAM adapters are the **only** trainable representation module shared by the future loss and
the action loss.

Evaluated at four checkpoints on 256 held-out, episode-disjoint samples (same protocol as G0:
fixed per-layer token-mean probe, full adapter bypass, checkpoint-native future-training latent
space).

### Result

| step | L_on | L_off | gap (on−off) | gap 95% CI | relative penalty | pg_on | pg_off | C_on | C_off | ΔC | ΔC 95% CI |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2500 | 0.04855 | 0.04757 | +0.00097 | [−0.0084, +0.0083] | 2.0% | +4.28% | +1.93% | 0.070 | 0.340 | −0.270 | [−0.348, −0.190] |
| 10000 | 0.03206 | 0.02744 | +0.00462 | [+0.0010, +0.0093] | 16.8% | +5.69% | +2.70% | 0.087 | 0.287 | −0.200 | [−0.258, −0.143] |
| 20000 | 0.01849 | 0.01482 | +0.00367 | [+0.0018, +0.0060] | 24.7% | +5.18% | +2.25% | 0.103 | 0.308 | −0.205 | [−0.267, −0.144] |
| 30000 | 0.01727 | 0.01353 | +0.00373 | [+0.0021, +0.0056] | 27.6% | +5.02% | +2.36% | 0.105 | 0.317 | −0.212 | [−0.274, −0.150] |

(`gap`, `L_on`, `L_off` are the checkpoint's demonstration action loss; `pg` is the relative
MSE gain in held-out future decodability from enabling adapters; `C` is the mean paired
adapter-bypass action-loss cost; `ΔC = C_on − C_off`.)

**What held at every point:**

1. **Predictive-state gate passed cleanly and stably.** `pg_on` was consistently about **2×**
   `pg_off` (≈5% vs ≈2–3%), with the interaction CI excluding zero throughout. Future
   supervision reliably made the shared adapter state carry more decodable future information
   — this is the one link in the causal chain (`T → M`) that this project set out to test and
   that actually replicated.
2. **Policy gate failed and got worse, not better.** The relative action penalty from future
   supervision rose monotonically from 2.0% to 27.6% and never came close to reversing. By
   10,000 steps the gap's CI had already left zero and stayed there.
3. **Interaction gate failed in the wrong direction.** `ΔC` was negative at all four points
   with a CI that never touched zero: the future-**off** policy depended on the shared
   adapters *more* than the future-on policy did, the opposite of the mediation prediction.
   The sharpest single number: under full adapter bypass, future-on's disadvantage of
   −0.0037 (worse than off) flips to an **advantage of +0.208** (better than off). The shared
   adapters are a net burden to the future-on policy, not the source of any benefit.

**Interpretation at this point.** The isolated design cleanly confirmed `T → M` and cleanly
falsified `M → Y` in the direction the topic requires — but with an unresolved confound: LoRA
disabled meant the 2.37M adapters were the *only* trainable representation, forcing future and
action objectives to compete for one small bottleneck. That confound alone could produce this
exact pattern (predictive state formed, but action starved of capacity) without saying anything
about whether a predictive adapter state is *ever* useful to the policy. This is why the user
authorized one further experiment rather than archiving here.

---

## 4. Capacity-restored G1 — the decisive experiment

### Design

Same matched-training design as §3, with one addition: backbone LoRA is restored to the
released specification (30 layers, rank 64, alpha 128, **identical in both arms**) and given
+87.5M parameters of action-only capacity via exact gradient routing implemented in the
trainer:

```text
per step:
  action_loss.backward()                    # populates LoRA + adapter grads from action only
  snapshot LoRA grads
  video_loss.backward()                     # adds future grads to adapter AND LoRA
  restore LoRA grads to the action-only snapshot   # strip future's contribution from LoRA
```

This gives the action path non-shared capacity while the WAM adapters remain the **sole** route
by which future supervision can reach the deployed policy representation. The routing was
verified twice:

1. **Algorithmically**, in `tests/test_lora_gradient_routing.py`, on a toy graph where a
   "shared" module and an "action-only" module are both present; confirms LoRA-equivalent
   gradient equals a pure action-loss backward, differs from a naively combined backward, and
   that the shared module still receives both objectives. A second test confirms the routing is
   a no-op when the video loss weight is zero (the future-off arm's code path).
2. **On the real model**, before launch: `LoRA: routed vs action_only maxdiff = 0.000e+00`,
   `LoRA: routed vs combined maxdiff = 2.168e-01` (confirms the two differ, i.e. the test is not
   trivially passing), `WAM adapters: routed vs combined maxdiff = 0.000e+00` (adapters still
   see both objectives).

Everything else matched §3: frozen backbone, frozen proprio, clipping threshold `1e9` (observed
grad norms ~0.6–10.3), shared initialization checkpoint (rebuilt for this LoRA-present
architecture), identical seed/data order. Only `lambda_video` differs between arms. Batch size
was reduced from 32 to 24 for memory (LoRA adds ~87.5M parameters and their activations);
matched identically across both arms, so pairing is unaffected.

### Result

| step | L_on | L_off | gap (on−off) | gap 95% CI | pg_on | pg_on 95% CI | pg_off | pg_off 95% CI | C_on | C_off | ΔC | ΔC 95% CI |
| ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| 10000 | 0.03224 | 0.03328 | −0.00104 | [−0.0058, +0.0029] | −1.01% | [−0.0027, −0.0002] | +0.84% | [+0.0003, +0.0018] | 0.0007 | 0.0004 | +0.0003 | [−0.0008, +0.0015] |
| 20000 | 0.01600 | 0.01405 | +0.00195 | [+0.0002, +0.0039] | −0.50% | [−0.0015, +0.0003] | −0.38% | [−0.0016, +0.0005] | 0.0012 | 0.0046 | −0.0034 | [−0.0058, −0.0014] |
| 30000 | 0.01304 | 0.01125 | +0.00179 | [+0.0005, +0.0032] | −0.72% | [−0.0020, +0.0000] | −0.40% | [−0.0016, +0.0005] | 0.0016 | 0.0051 | −0.0035 | [−0.0056, −0.0016] |

At step 30000 (the last authorized checkpoint):

```text
gates:
  policy_gain              = false   (gap +0.00179, CI [+0.0005,+0.0032]; on is WORSE)
  predictive_state_gain    = false   (pg_on −0.72% CI crosses zero; pg_off −0.40% CI crosses zero)
  adapter_dependence_gate  = false   (ΔC −0.0035, CI [−0.0056,−0.0016]; wrong-signed)
verdict: NO_POLICY_GAIN_FROM_FUTURE_SUPERVISION
future_on advantage under normal adapters:  −0.00179  (worse)
future_on advantage under adapter bypass:   +0.00169  (better)
```

**Reading the trend.** The 10k point oscillated to the opposite sign on both `gap` and `ΔC`,
but every one of those 10k values had a CI crossing zero, and — tellingly — the
predictive-state gate itself was internally inconsistent at 10k (`pg_on` significantly
*negative*, `pg_off` significantly positive), the opposite of the stable pattern seen
throughout §3. That is early-training noise from a freshly restored representation, not a
different regime: by 20k and 30k the pattern converges back to the same failure mode as the
isolated run — `gap` significant and positive (future-on worse), `ΔC` significant and negative
(future-off depends on the adapters more), and by 30k neither arm shows a significant adapter
future-gain at all.

**What this rules out.** The capacity-restored design gives the action path 87.5M parameters of
non-shared capacity — roughly 37× the size of the shared adapter bottleneck it was competing
for in §3. If the isolated-G1 negative had been an artifact of forcing future and action to
share a 2.37M-parameter bottleneck, restoring dedicated action capacity should have let a
useful predictive-state effect emerge. It did not. The predictive-state effect that was so
stable in §3 (pg_on ≈ 2× pg_off, CI excluding zero at every point) essentially **disappeared**
here — both arms show statistically insignificant, slightly negative adapter future-gain by
20k–30k. Restoring LoRA did not just fail to produce a policy benefit; it appears to have
displaced most of what little future-predictive signal the adapters had been carrying under the
bottlenecked configuration, plausibly because LoRA's larger, unconstrained action-relevant
capacity now does more of the representational work the adapters used to be forced into,
leaving the adapters comparatively undertrained on both objectives.

---

## 5. Engineering notes retained for any future reader

- Multi-GPU NCCL failed with an unrecoverable illegal-memory-access / `SIGABRT` on this host's
  Blackwell cards, reproduced under four separate NCCL environment-variable configurations
  (default, `NCCL_P2P_DISABLE`, `+NCCL_SHM_DISABLE`, `NCCL_P2P_LEVEL=SYS`). All isolated and
  capacity-restored arms therefore ran single-GPU (`world_size=1`), which is scientifically
  immaterial to a paired comparison but capped wall-clock throughput to ~0.9–1.15 step/s.
- Upstream applies `cfg.seed` inside `Wan22Trainer.__init__`, **after** `instantiate(cfg.model,
  ...)`; matched initialization therefore used one shared checkpoint built by seeding before
  model construction (`g1_make_init.py`), loaded into both arms via upstream `resume=`.
- Upstream's global grad-clip threshold was set to `1e9` (functionally inactive) rather than
  disabled outright, and `grad_norm` was mirrored into the console log (a one-line trainer
  patch) specifically so the "clipping never activates" claim is auditable from the run logs
  rather than assumed. Observed values (0.6–10.3) never approached the threshold.
- `LIGHTWAM_FREEZE_PROPRIO=1` (an opt-in trainer flag, default off) freezes `proprio_encoder`;
  required because upstream `build_inputs` appends the proprio token to context consumed by
  both the future and action branches, otherwise a second, unidentified trainable route.
- The exact-gradient LoRA routing (`_backward_with_lora_action_only` in
  `src/lightwam/trainer.py`, opt-in via `LIGHTWAM_LORA_ACTION_ONLY=1`) initially used
  `retain_graph=True` for the action backward; this caused an OOM once LoRA's ~87.5M parameters
  and activations were added. Verifying that the action and video forward passes build fully
  disjoint computation graphs (they share only parameters, no activations) let `retain_graph`
  be dropped, which resolved the OOM without changing the routing semantics (re-verified against
  the real model afterward).
- G0 on the third LIBERO suite (`libero_10`) was never completed: the required 15.7GB offline
  latent-cache shard stalled indefinitely on the HF Xet backend. Two independently completed G0
  suites (`libero_spatial`, `libero_object`) already agreed, so this was not chased further; it
  does not affect the final verdict, which rests on the two matched-training experiments.

---

## 6. What this project now supports, and does not

**Supported, with reasonably strong evidence across two independent matched-training designs
and 4+3 evaluation checkpoints:**

> Training-time future/video supervision, when forced through a small dedicated adapter
> pathway, reliably makes that pathway's state more linearly predictive of the real future.
> "Future information is decodable" and "future information is useful to the deployed policy"
> are different claims, and the gap between them is not a measurement artifact — it survives a
> parameter-free probe, an episode-disjoint held-out split, a real module-level bypass
> intervention, and a second design that removes a specific confound (shared-capacity
> competition) the first design could not rule out.

**Not supported, despite being given two clean chances:**

> That predictive state is used by the deployed policy for better action. In both matched
> designs tested, the opposite held: the future-off policy depended on the shared adapter
> pathway *more*, and in the design with the bottleneck confound removed, giving action extra
> non-shared capacity, if anything, weakened the residual future-predictive signal rather than
> letting it become useful.

**Explicitly not addressed by anything in this project:**

- Whether some other architecture, adapter width, layer placement, or supervision schedule
  would produce a different result. Per the project's stated principle, that space was not
  searched — no SAE, no PCA/CCA, no learned projector, no rank/layer/threshold sweep, no
  additional λ values, no different probe.
- Whether an action-conditioned world model exists anywhere in Light-WAM. This project never
  attempted to identify that stronger claim.
- Closed-loop success. All action measurements here are offline demonstration action loss.

---

## 7. Verdict

```text
KILL — ARCHIVE TOPIC 15
```

Both the required matched-training tests were run to their authorized stopping points. Neither
supported the mediation claim; the second, more permissive design (extra non-shared action
capacity) did not rescue the first. Per the project's standing rule, no further representation
machinery, architecture variant, or hyperparameter sweep will be added to try to recover a
positive result. See `ARCHIVE_SUMMARY.md` for the short-form record.

---

# Appendix — G0 detail (retained from the original screen)

## G0 part A — future-predictability effect

`libero_spatial`, held-out episodes (256 samples, 192 train / 64 test, episode-disjoint):

| condition | probe R² | probe MSE |
| --- | --- | --- |
| normal (adapters enabled) | 0.16560 | 0.118735 |
| full adapter bypass (all scales = 0) | 0.16496 | 0.118826 |
| mean-target baseline | — | 0.142300 |

```text
relative MSE gain from adapters = +0.077 %,  95% CI [-7.9e-04, +1.05e-03]  (crosses zero)
```

`libero_object` (step 12500, weaker checkpoint):

```text
relative MSE gain from adapters = +0.36 %,  95% CI [-2.4e-04, +9.2e-04]  (crosses zero)
```

A fit-free validity check (`g0_feature_delta_check.py`) confirms the null is a property of the
adapters, not the measurement: adapters move the exact pooled probe feature by ~10% relative
norm at every layer, carrying ~4.7% of across-sample variance.

## G0 part B — adapter causal action effect

| checkpoint | normal action loss | bypass action loss | relative increase | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `libero_spatial` | 0.002528 | 0.005948 | +135.3% | [+2.50e-03, +4.46e-03] |
| `libero_object` | 0.009590 | 0.010435 | +8.8% | [+1.5e-04, +1.7e-03] |

Both: `ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN`.
