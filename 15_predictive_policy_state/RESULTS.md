# Topic 15 — G0 results

**Question.** Does training-time world modeling act through a predictive policy state?

**Stage run.** G0 only — the pre-registered released-checkpoint screen of the native
Light-WAM WAM-adapter route, exactly as defined in [README.md](README.md) and
[VALIDATION_AUDIT.md](VALIDATION_AUDIT.md).

**Headline.** The native route is **strongly action-causal but not more future-predictive**.
Pre-registered verdict: `ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN`.
Per the pre-registered decision table this is a **negative** for the topic's mechanism and a
**kill of this Light-WAM route**. G1 was therefore not run.

---

## 0. What was actually executed

The topic had never been executed before this session: no Light-WAM checkout, checkpoint,
dataset or latent cache existed on this machine. The scientific contrast was not modified.
Only the engineering environment was built:

| item | value |
| --- | --- |
| Light-WAM | `https://github.com/L1ziang/Light-WAM` @ `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae` |
| released checkpoints | `l1ziang/lightwam-checkpoints` |
| datasets | `yuanty/LIBERO-fastwam` |
| latent / text caches | `l1ziang/lightwam-offline-cache` |
| backbone | `Wan-AI/Wan2.1-T2V-1.3B` (frozen) |
| env | conda `lightwam`, python 3.10, torch 2.7.1+cu128 |
| hardware | 1 GPU per run, no training, no simulator |

The only change to `g0_lightwam.py` was one engineering line that defaults
`DIFFSYNTH_MODEL_BASE_PATH` to `<lightwam-root>/checkpoints`, because upstream resolves
`model_id` against that variable. No scientific parameter was touched.

The architecture audit inside the script passed against the real released checkpoint and
confirmed every assumption the README makes:

```text
adapter layers                       = [8, 16, 24]
action readout feature sources       = [adapted] on all three layers
video hidden dim                     = 1536   (pooled probe feature = 3 x 1536 = 4608)
video_latent_spatial_downsample_factor = 2
backbone LoRA                        = enabled (layers 0..29, rank 64)
```

Design as pre-registered: 256 samples, **one non-padded window per episode**,
**episode-disjoint** split (192 train episodes / 64 test episodes, intersection empty),
parameter-free per-layer token mean (the trained learned-query pooler is *not* used for the
probe), one fixed linear ridge probe, future target = future-minus-first clean VAE latent
change **after the checkpoint's own future-training spatial downsampling** (12,544 dims, no PCA),
and a full second backbone pass with **every** adapter scale set to 0.

---

## 1. G0 part A — future-predictability effect

`libero_spatial`, held-out episodes:

| condition | probe R² | probe MSE |
| --- | --- | --- |
| normal (adapters enabled) | 0.16560 | 0.118735 |
| full adapter bypass (all scales = 0) | 0.16496 | 0.118826 |
| mean-target baseline | — | 0.142300 |

```text
relative MSE gain from adapters = +0.077 %
paired episode bootstrap mean   = +9.1e-05
95 % CI                         = [-7.9e-04, +1.05e-03]      <- crosses zero
continuation floor              = +5 %
```

**The trained WAM adapters add essentially no linearly decodable future information.**
The future *is* partly decodable — R² ≈ 0.166 against a real 12,544-dim future-change target —
but that decodability is already present without the adapters, i.e. it lives in the frozen
Wan backbone (plus its LoRA), not in the adapter residual.

The effect is not merely below the 5 % floor; it is ~65× smaller than the floor and its
confidence interval contains zero.

### Validity of the negative

A null part-A result would be meaningless if the fixed token-mean summary were blind to the
adapters. It is not. `g0_feature_delta_check.py` measures how far the adapters move the exact
feature the probe consumes (`results/libero_spatial_g0/feature_delta_check.json`):

```text
relative ||normal - bypass|| / ||normal||   = 0.103   (test split)
per-layer:  layer8 0.100   layer16 0.101   layer24 0.110
across-sample variance carried by the delta = 32.8  of  703.8   (~4.7 %)
```

The adapters visibly and consistently change the measured state at every adapter layer. The
probe simply finds no *additional future information* in that change.

---

## 2. G0 part B — adapter causal action effect

Same two forward passes, same unchanged deployed action expert, same checkpoint action-loss
definition and temporal weighting, same held-out episodes:

| condition | demonstration action loss |
| --- | --- |
| normal | 0.0025284 |
| full adapter bypass | 0.0059484 |

```text
relative action-loss increase   = +135.3 %
paired episode bootstrap mean   = +3.42e-03
95 % CI                         = [+2.50e-03, +4.46e-03]     <- far from zero
action RMS shift                = 0.0899
```

**Bypassing the explicit WAM adapters more than doubles action error.** This is a genuine
module-level intervention in the computation graph, and the dependence is large and
unambiguous.

---

## 3. Is G0 worth taking to G1?

**No.** The pre-registered decision table gives:

| future predictability improves? | bypass hurts action? | verdict |
| --- | --- | --- |
| **no** | **yes** | `ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN` |

README §"G0 decision table" and the project principle "G0 弱 → 停" both resolve this to stop.
The matched `lambda_video = 1` vs `lambda_video = 0` training was **not** launched.

The reason this is the right call and not premature: the adapters matter enormously for
action, so the route is not weak in general — it is weak *in the specific way the topic
requires*. The topic's causal chain is

```text
training-time future supervision -> predictive policy state -> better action
```

G0 finds a large `adapter state -> action` link and no `adapter state -> future information`
link. The middle term of the chain is the thing that failed to appear.

---

## 4. What these results DO establish

1. The released Light-WAM deployed action state carries real, non-trivial future information
   (held-out R² ≈ 0.166 in the checkpoint's own future-training latent space, on
   episode-disjoint data with one window per episode).
2. That future information is **not** contributed by the explicit WAM-adapter pathway; it is
   already present when all adapter scales are zero.
3. The explicit WAM-adapter pathway is nevertheless **strongly causal for action**: removing
   it (including all downstream propagation through later layers) more than doubles
   demonstration action error.
4. Therefore the adapters carry something the action expert depends on heavily, and that
   something is not captured by linear future-change decodability from the state.

---

## 5. What these results explicitly DO NOT establish

- **Not** a proof that no predictive state exists anywhere in Light-WAM. The bypass condition
  still contains the trained backbone LoRA, which is also updated by the future objective.
  G0 isolates the explicit adapter route only — by design, as stated in the README.
- **Not** a statement that future supervision is useless during Light-WAM training. G0 never
  manipulated `lambda_video`; it inspects one released checkpoint.
- **Not** a statement that the adapters contain no future information in any form. It shows
  they add none that a fixed linear probe can read from a fixed token-mean summary — the
  pre-registered measurement.
- **Not** a claim about closed-loop success. The action effect is an offline demonstration
  action loss.
- **Not** mediation. Even the part-B positive would not have licensed a mediation claim; the
  README removed `PROMISING_NATIVE_MEDIATOR` for exactly this reason.

The negative is a **project-level kill of the clean Light-WAM adapter route**, not a universal
scientific falsification. That distinction was pre-registered and is preserved here.

---

## 6.–9. G1 (matched future-on / future-off training)

**Not triggered.** G0 did not license it, so there is nothing to report for:

6. whether future-on/off were truly matched;
7. whether future supervision produced a policy gain;
8. whether predictive state strengthened with future supervision;
9. whether an adapter-bypass interaction supported mediation.

The matched-training design in README §"The decisive next experiment" remains correct and was
verified against the real upstream code during this session (all three implementation hazards
it names are real in `Light-WAM@b2785f6`):

- `Wan22Trainer.__init__` calls `set_global_seed(self.seed)` **after** `instantiate(cfg.model, ...)`
  in `runtime.run_training`, so equal CLI seeds do not guarantee equal initialization.
  Upstream `resume=<weights.pt>` loads weights without optimizer/step state, so it is a usable
  common-init mechanism.
- `trainer.py` clips the gradient norm over `self.model.parameters()` with `max_grad_norm=1.0`,
  i.e. globally, exactly the confound the README describes.
- `configure_trainable_modules` leaves `proprio_encoder` trainable, and `build_inputs` appends
  the proprio token to the context used by **both** the future and action branches.

These notes are recorded so a future project does not have to re-derive them. They are not a
plan to continue this topic.

---

## 10. Final verdict

```text
KILL THIS LIGHT-WAM MECHANISM ROUTE
```

No rescue was attempted and none should be. Per README §"Do not rescue a weak result", the
following were **not** run and should not be run to revive this topic: SAE feature search,
PCA / CCA / Procrustes, learned causal projectors, rank search, layer search, threshold search,
or substituting the trained learned-query pooler into the probe.

The one thing that *was* added is a validity diagnostic (`g0_feature_delta_check.py`) whose
only purpose is to confirm that the negative is a property of the adapters and not of the
measurement. It reports magnitudes; it fits nothing and searches nothing.
