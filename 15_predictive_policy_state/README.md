# 15 — Does Training-Time World Modeling Act Through a Predictive Policy State?

## Question

Training-time future prediction can improve a deployed robot policy even when explicit future generation is absent at deployment. The mechanism question is:

> **When future prediction helps the policy during training, does the deployed policy actually act through information about the future that was left in its internal state?**

The first claim is intentionally narrower than “there is a world model inside the policy.” A representation can predict the future without implementing an action-conditioned transition model. The first thing worth establishing is therefore a **causally action-used predictive state**.

## Why this is a real mechanism question

Recent WAM work has already established the prerequisites separately:

- training-time future/video supervision can improve direct action policies;
- future-oriented information can be decoded from WAM representations;
- released models make the relevant internal states accessible.

The missing link is the arrow:

```text
training-time future supervision
            ↓
   predictive policy state
            ↓ ?
           action
```

G0 does **not** try to prove the full mediation chain. It asks whether Light-WAM contains a clean native candidate for the middle arrow before we invest in training ablations or simulator rollouts.

## Why Light-WAM is a useful first platform

The released Light-WAM state-fusion configuration gives an unusually direct intervention point:

- WAM adapter layers are `8, 16, 24`;
- the deployed `StateFusionActionExpert` is configured with `feature_sources: [adapted]`;
- at every selected layer, the upstream code stores both the tensor immediately **before** the residual WAM adapter (`backbone_tokens`) and immediately **after** it (`adapted_tokens`);
- the action expert consumes the adapted tensors through its learned-query pooler.

So the primary causal intervention requires no learned subspace and no arbitrary direction:

```text
normal action readout:       adapted state  -> action expert -> action
native counterfactual:       backbone state -> action expert -> action
```

Both tensors come from the same input, same layer, same forward pass, and have identical shape. The action expert and all of its weights are held fixed.

The released model also uses backbone LoRA. Therefore this intervention isolates the **native adapter-state route into the action readout**, not every parameter changed by world-model training. A null result is a null for this clean Light-WAM route, not a proof that no predictive state exists anywhere in the model.

## G0: one clean offline experiment

G0 has only two measurements.

### 1. Is future information actually stronger in the action-fed adapted state?

For each sample, run the ordinary single-observation Light-WAM action backbone pass. At layers `8, 16, 24`, extract the exact pooled features used by the deployed action expert.

Two feature sets are evaluated with the **same fixed linear ridge probe**:

```text
X_adapted  = action expert's pooled adapted states
X_backbone = the same learned-query pooler applied to same-layer backbone states
```

The target is the real cached future-video latent change:

```text
Y = VAE_latents[t > 0] - VAE_latents[t = 0]
```

Every future latent position is retained and flattened. There is no PCA, SAE, CCA, rank search, or target projection.

The probe split is **episode-disjoint**, with one full non-padded window per episode. This is essential: a random split of overlapping trajectory windows can make future prediction look strong simply because nearly identical windows appear in train and test.

The primary quantity is held-out future-target error:

```text
future gain = MSE(backbone probe) - MSE(adapted probe)
```

Positive means the actual state delivered to the action path contains additional linearly accessible future information relative to its same-layer pre-adapter state.

### 2. Does action causally depend on that native adapted state?

Using the **same forward pass**, feed the action expert either:

```text
normal:        adapted_tokens
intervention:  backbone_tokens substituted into the adapted input slots
```

Then measure the checkpoint's own per-sample action loss against the demonstration action chunk:

```text
action effect = loss(backbone intervention) - loss(normal adapted state)
```

Positive means removing the local adapter contribution from the deployed readout makes action prediction worse.

The intervention never uses the fitted future probe. This is deliberate: the probe only measures whether future information is present; the causal test remains architecture-native.

## The entire decision table

| Future information added by adapted state? | Action worsens under adapted→backbone? | G0 interpretation |
| --- | --- | --- |
| yes | yes | **PROMISING_NATIVE_MEDIATOR** |
| yes | no | **PREDICTIVE_BUT_NOT_ACTION_USED** |
| no | yes | **ACTION_RELEVANT_BUT_NOT_PREDICTIVE** |
| no | no | **NO_CLEAN_NATIVE_SIGNAL** |

The script uses one practical continuation floor (`5%` relative effect by default) plus a paired bootstrap interval. This is not a publication threshold; it only enforces the project rule that G0 should reveal a **large, clean effect without analytic rescue work**.

If the result is weak, do **not** add PCA/SAE/CCA/projector sweeps. Either move to a platform with a cleaner bottleneck or stop this Light-WAM instance.

## What G0 can and cannot establish

A positive G0 supports:

> A native state shaped inside the WAM pathway both carries extra future information and is causally required for accurate action readout.

It does **not** yet establish:

1. that the future loss, rather than action supervision, uniquely created that state;
2. that this state mediates the full performance gain from world-model co-training;
3. that the state implements action-conditioned dynamics rather than goal-conditioned predictive progress;
4. that the intervention changes closed-loop task success.

Those are later questions. They should only be attempted if G0 produces a large direct signal.

## Why there are no extra controls in G0

The purpose of G0 is not to publish a causal representation paper in one script. It is to decide whether the scientific object exists in a form simple enough to study.

We intentionally do **not** run:

- random-direction ablations;
- SAE feature search;
- PCA/CCA/Procrustes alignment;
- layer/rank/threshold sweeps;
- simulator rollouts;
- retraining.

If the architecture-native contrast is not already informative, the project should not be rescued by increasingly elaborate representation machinery.

## Run

Use the official Light-WAM environment and released checkpoint/cache. The upstream repository is expected to be installed normally.

Minimal example:

```bash
cd candidate_topics/15_predictive_policy_state

CUDA_VISIBLE_DEVICES=0 python g0_lightwam.py \
  --lightwam-root /path/to/Light-WAM \
  --checkpoint /path/to/lightwam/checkpoint.pt \
  --dataset-dir /path/to/libero_suite_lerobot \
  --latent-cache-dir /path/to/latent_cache_for_same_suite \
  --text-cache-dir /path/to/text_embeds_cache/libero \
  --num-samples 256 \
  --batch-size 4 \
  --output-dir ./results/libero_g0
```

`config.yaml` and `dataset_stats.json` are automatically searched for in checkpoint parent directories. If the released checkpoint layout does not place them nearby, pass:

```bash
  --training-config /path/to/config.yaml \
  --dataset-stats /path/to/dataset_stats.json
```

A shell wrapper is also provided:

```bash
LIGHTWAM_ROOT=/path/to/Light-WAM \
CKPT=/path/to/checkpoint.pt \
DATASET_DIR=/path/to/libero_suite_lerobot \
LATENT_CACHE_DIR=/path/to/latent_cache_for_same_suite \
TEXT_CACHE_DIR=/path/to/text_embeds_cache/libero \
bash run_g0.sh
```

## Output

The primary output is:

```text
g0_result.json
```

It records:

- exact Light-WAM git revision when available;
- checkpoint/config paths;
- architecture audit;
- episode-disjoint sample IDs;
- future-probe effect;
- causal action-readout effect;
- the four-way G0 verdict.

Set `--save-tensors` only when debugging; future-latent targets are large.

## Resource policy

G0 intentionally requires:

- **no training**;
- **no simulator**;
- **one GPU**;
- one single-observation backbone pass per sample;
- released offline future-latent cache;
- 100–500 samples (default `256`).

The backbone is run once per sample. Baseline and counterfactual actions reuse the same cached native layer states, so the causal readout test does not require a second video-backbone pass.

## If G0 is strongly positive

Only then move to the two experiments needed for a publication-level mechanism claim:

1. **closed-loop native intervention** — check whether the same adapted→backbone intervention degrades task success, not only offline action loss;
2. **matched training mediation** — compare otherwise matched action-only vs future-supervised training and test whether the predictive-state effect grows with, and explains a substantial fraction of, the policy gain.

If we later want to claim an actual *implicit world model*, add an action-conditioned counterfactual test. Until then the correct object is a **predictive policy state**.

## Upstream implementation points audited

Light-WAM repository: <https://github.com/L1ziang/Light-WAM>

Relevant upstream files:

- `configs/model/lightwam.yaml`
- `src/lightwam/models/wan22/wan_video_dit.py`
- `src/lightwam/models/wan22/lightwam.py`
- `src/lightwam/models/wan22/state_fusion_action_expert.py`
- `src/lightwam/datasets/lerobot/robot_video_dataset.py`

The G0 script fails fast if the loaded checkpoint/config no longer exposes the adapted-only state-fusion path assumed above.
