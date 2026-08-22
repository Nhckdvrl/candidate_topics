# 15 — Does Training-Time World Modeling Act Through a Predictive Policy State?

## Question

Training-time future prediction can improve a deployed robot policy even when explicit future generation is absent at deployment. The mechanism question is:

> **When future prediction helps during training, does it leave a predictive internal state that the deployed policy actually relies on for action?**

This is deliberately narrower than claiming that a complete action-conditioned world model lives inside the policy.

The important distinction is:

```text
future information is decodable
            !=
that future information is the causal code used for action
```

The validation is therefore split into two stages. **G0 is only a clean released-checkpoint screen. It is not allowed to claim mediation.** A positive G0 licenses one matched-training experiment; a weak G0 kills this simple Light-WAM route rather than triggering projector / SAE / layer-sweep rescue work.

---

## Why Light-WAM is a useful first platform

The released Light-WAM state-fusion configuration exposes a simple native pathway:

- WAM adapter layers are `8, 16, 24`;
- the deployed `StateFusionActionExpert` reads `feature_sources: [adapted]`;
- the action path is a single-observation backbone pass;
- the explicit WAM adapters are residual modules with a native scalar `scale`;
- future/video training uses the same adapter parameters.

The released model also uses backbone LoRA and action supervision. Therefore **released-checkpoint G0 can isolate only the explicit WAM-adapter route**, not every parameter altered during training.

That limitation is intentional and is enforced in the interpretation.

---

# G0 — clean native-route screen

G0 asks only two questions.

## A. Do the trained WAM adapters add future-predictive information to the deployed state?

For every sample we run the ordinary single-observation action backbone twice:

```text
normal:          trained WAM adapters enabled
adapter bypass:  every WAM adapter scale = 0
```

The second pass is important. Merely substituting each layer's cached pre-adapter tensor at the final readout is **not** a complete adapter ablation, because an earlier adapter already affects the input to later layers. Setting all adapter scales to zero and rerunning the backbone removes both each local residual and all downstream propagation of those residuals.

### Representation measurement

The probe does **not** use the action expert's learned-query pooler.

That pooler was trained specifically on adapted states, so applying it to bypass states would bias the comparison toward the normal/adapted distribution.

Instead both sides use the identical parameter-free summary:

```text
for each adapter layer:
    mean over spatial/video tokens
concatenate layer summaries
```

Then one fixed linear ridge probe predicts the real future latent change.

### Future target

The target is constructed in the **same clean spatial latent space used by the checkpoint's future/video training objective**.

Light-WAM's released configuration uses spatial latent downsampling, so using full-resolution cached latents would incorrectly ask the probe to predict high-frequency details that the world-model objective was never trained on.

The target is:

```text
future-change target
= future clean VAE latents - first clean latent
```

after applying the checkpoint's own future-training latent preprocessing/downsampling.

Using change-from-first-frame suppresses the trivial static-scene component and focuses the measurement on what changes into the future.

The train/test split is episode-disjoint, with one full non-padded window per episode, so overlapping trajectory windows cannot leak into both sides of the probe.

Primary representation quantity:

```text
future gain
= MSE(adapter-bypass probe) - MSE(normal-adapter probe)
```

Positive means enabling the trained WAM adapters makes the deployed state more linearly informative about the held-out future.

---

## B. Does action depend on the explicit WAM-adapter pathway?

The same normal and adapter-bypass passes are sent through the **unchanged deployed action expert**.

We measure:

```text
action effect
= action_loss(adapter bypass) - action_loss(normal)
```

against the demonstration action chunk using the checkpoint's own action-loss definition and temporal weighting.

Positive means the explicit WAM-adapter transformation has a causal effect on action prediction.

This is a genuine module intervention, but it is **not a selective intervention on future-predictive content**. Therefore it cannot by itself prove that the particular future-decodable dimensions are what the action expert uses.

---

# G0 decision table

| Future predictability improves with adapters? | Adapter bypass hurts action? | Correct interpretation |
| --- | --- | --- |
| yes | yes | **PROCEED_TO_MATCHED_TRAINING** — a simple native route exists and is worth testing causally across training conditions |
| yes | no | **FUTURE_GAIN_WITHOUT_CLEAN_ADAPTER_ACTION_EFFECT** — predictive information is added, but this native adapter route is not clearly necessary for action |
| no | yes | **ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN** — adapters matter for action, but not in the predictive way required by the topic |
| no | no | **NO_CLEAN_ADAPTER_ROUTE** — stop this Light-WAM route |

The default `5%` relative-effect floor plus paired episode bootstrap is only a continuation rule. It is not a publication significance threshold.

## What a positive G0 means

A positive G0 supports only:

> **The same explicit native WAM-adapter pathway both increases held-out future decodability and improves action prediction.**

It does **not** establish:

- that future loss, rather than action loss, created the relevant adapter state;
- that the decodable future content itself is the causal action code;
- that future supervision explains the policy-performance gain;
- that the state is an action-conditioned dynamics/world model;
- that the offline action effect changes closed-loop success.

The old wording `PROMISING_NATIVE_MEDIATOR` was removed after audit because it overstated what the intervention identifies.

## What a negative G0 means

A negative G0 does not mathematically prove that no predictive state exists anywhere in Light-WAM, because the released checkpoint also contains backbone LoRA and other trained components.

For **this project**, however, it is a valid stop signal:

> if the architecture's simplest explicit WAM-adapter pathway does not show a large predictive + action effect, do not rescue the topic with PCA / SAE / CCA / projector / layer search.

That is a project-level kill of this clean Light-WAM route, not a universal scientific falsification.

---

# The decisive next experiment if G0 is positive

To answer the actual training-time mechanism question, the next experiment must manipulate **future supervision itself**, not merely inspect one released checkpoint.

The cleanest Light-WAM version is a matched pair of trainings in which the WAM adapters are the only trainable representation module shared between future loss and action loss.

Use in **both** runs:

```text
freeze pretrained video backbone = true
backbone LoRA                   = false
same architecture
same data order
same initialization
same optimizer / LR / steps
same action loss
same action expert
```

Then vary only:

```text
future-on:   lambda_video = 1
future-off:  lambda_video = 0
```

Why disable backbone LoRA?

With LoRA disabled and the pretrained backbone frozen:

- WAM adapters are shared by future and action paths;
- future head is future-only;
- state-fusion action expert is action-only;
- proprio encoder is action-only.

Therefore any effect of `lambda_video` on action-relevant representation parameters must pass through the WAM adapters.

### Two implementation requirements are mandatory

#### 1. Identical initialization

Upstream Light-WAM currently calls its configured random seed inside `Wan22Trainer`, **after the model has already been instantiated**. Two separately launched runs therefore are not guaranteed to start from identical randomly initialized adapters/action expert merely because `seed=` is equal.

Before matched training, the local agent must either:

- seed Python / NumPy / PyTorch **before model instantiation**, or
- create one common initialization checkpoint and load it into both runs.

Do not compare separately randomized initializations and call the difference a future-supervision effect.

#### 2. Prevent global grad clipping from creating an indirect coupling

Upstream training clips the gradient norm over **all trainable parameters** before every optimizer step.

If future-on includes large future-head gradients while future-off does not, global clipping can rescale action-expert gradients differently even though the future head is not on the action path.

For the matched mechanism run, either disable clipping or set the clipping threshold high enough that clipping never activates, and verify this from logged gradient norms.

Otherwise `future-on vs future-off` is not a clean representation-routing intervention.

### What pattern would justify the mechanism claim?

The matched experiment should establish all of the following:

1. **future-on improves the deployed action policy over future-off** under otherwise matched training;
2. the future-on action state has a substantially larger future-predictive gain than future-off;
3. bypassing the shared WAM adapters removes a substantially larger amount of action performance in future-on than in future-off;
4. ideally, the future-on policy advantage itself shrinks strongly under adapter bypass.

The clean causal quantity is the interaction:

```text
adapter-bypass cost under future-on
        -
adapter-bypass cost under future-off
```

combined with the direct future-on vs future-off policy gain.

This tests whether future supervision causally creates an action-used predictive adapter state.

It still does **not** justify the stronger statement that a complete counterfactual world model lives inside the policy. That would require a later action-conditioned consequence intervention.

---

# Run G0

```bash
cd candidate_topics/15_predictive_policy_state

CUDA_VISIBLE_DEVICES=0 python g0_lightwam.py \
  --lightwam-root /path/to/Light-WAM \
  --checkpoint /path/to/lightwam/checkpoint.pt \
  --dataset-dir /path/to/matching_libero_lerobot_dataset \
  --latent-cache-dir /path/to/matching_lightwam_latent_cache \
  --text-cache-dir /path/to/text_embeds_cache/libero \
  --num-samples 256 \
  --batch-size 4 \
  --output-dir ./results/libero_g0
```

`config.yaml` and `dataset_stats.json` are searched near the checkpoint. They can also be supplied explicitly.

A shell wrapper is provided as `run_g0.sh`.

The primary output is `g0_result.json`.

---

# Resource policy

G0 requires:

- no training;
- no simulator;
- one GPU;
- released offline latent cache;
- 100–500 episode-disjoint samples;
- **two** single-observation backbone passes per sample: normal and full adapter bypass.

The second pass is intentional. The previous one-pass local substitution was cheaper but did not fully remove earlier adapters' downstream effects.

---

# Do not rescue a weak result

Do not add:

- SAE feature search;
- PCA / CCA / Procrustes;
- learned causal projectors;
- rank search;
- layer search;
- threshold search.

If G0 is weak, stop the Light-WAM route.

If G0 is strong, go directly to the one matched future-on / future-off training test above.
