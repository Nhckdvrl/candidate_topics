# 13 — Does Repetition Hurt Because It Repeats, or Because It Repeats Too Soon?

**Status:** VALIDATION CODE READY — locked G-0 pilot + confirmation

## Scientific question

> If a learner sees the exact same documents the exact same number of times, can changing only the temporal distance between duplicate exposures materially change pretraining damage?

The seed paper, **Internal Data Repetition Destroys Language Models** (2026), establishes a strong exact-document repetition phenomenon: at fixed compute and a fixed 10% repeated-token budget, held-out loss is worst at an intermediate per-document repeat count. For the 34M Qwen3-style model, the reported peak is around `R ≈ 1400`; the paper randomly interleaves repeated copies into the training stream.

That leaves one clean missing variable:

> is the damage determined by the repeated **multiset**, or by the temporal **stream** in which that multiset is presented?

This repository implements the shortest experiment that can answer that question without representation probes, memorization thresholds, document-category controls, or a repeat-count sweep.

---

## Why this implementation is stricter than the original proposal

A naive clustered-vs-random-vs-spaced experiment can accidentally change several things at once:

- the positions of unique documents;
- which positions are occupied by the repeated-data component;
- the total number of repeated tokens;
- document length and therefore token-time;
- the repeated pool itself;
- initialization or optimizer trajectory.

The G-0 below removes all of these degrees of freedom.

### Fixed-length document atoms

The seed paper truncates documents to 2048 tokens and appends EOS. For this causal spacing test we make the timing control even cleaner: we keep only documents that are long enough, take exactly the first **2047 Qwen3 tokens + EOS**, and treat that 2048-token block as one training atom.

Therefore every slot has exactly the same token duration. Moving a repeated identity from one slot to another cannot shift the token positions of any later unique document.

This is an intentional identification choice. It is not claimed to be the only natural data distribution. The random-repetition gate below must first show that repetition damage still exists under this fixed-length subset; if it does not, the setup is invalid and the spacing comparison is not interpreted.

---

## Paper-matched minimum model

The G-0 uses the smallest architecture actually studied in the seed paper:

| item | locked value |
|---|---:|
| total parameters | 34,061,856 |
| layers | 3 |
| hidden size | 96 |
| FFN size | 256 |
| attention heads | 32 |
| KV heads | 32 |
| head dimension | 128 |
| vocabulary | 151,670 |
| context | 2048 |
| max position | 32768 |
| embeddings | untied input/output |
| norm / FFN | RMSNorm / SwiGLU |
| optimizer | fused AdamW |
| betas | `(0.9, 0.95)` |
| weight decay | 0.01 |
| grad clip | 1.0 |
| precision | BF16 |
| LR schedule | 20% warmup + cosine |

The implementation instantiates `Qwen3ForCausalLM` from scratch and **asserts the exact 34,061,856 parameter count** before training. If a Transformers version silently changes the architecture, the run stops.

The seed paper says its peak LR is derived from a base LR of `1e-6` and the optimizer-step token count. Rather than tune LR after seeing results, this G-0 freezes the natural square-root rule explicitly:

```text
peak_lr = 1e-6 * sqrt(tokens_per_optimizer_step)
```

With the default `4 × 16 × 2048 = 131,072` tokens per optimizer step, this is about `3.62e-4`.

---

## Why `R = 1386`

The pilot uses the cheapest training-duration cell actually covered by the seed paper:

```text
N  = 34,061,856 parameters
OT = 0.25
T  = 20 * OT * N = 170,309,280 target tokens
```

With fixed 2048-token atoms:

```text
total_blocks = floor(T / 2048) = 83,158
10% repeat slots ≈ 8,316
8,316 = 6 * 1,386
```

So we lock:

```text
repeat_count R = 1386
repeat_pool = 6 documents = 12,288 unique repeated tokens
realized repeated-slot fraction ≈ 0.1000024
```

This is almost exactly the seed paper's reported `R_peak ≈ 1400` for 34M, while avoiding a repeat-budget rounding artifact. There is **no R sweep** in G-0.

---

## Four conditions

For each matched trial, `schedule.py` draws one set of repeated-component slots and one fixed assignment of unique documents to every other slot.

### 1. `fresh`

All slots contain distinct documents.

This is the seed-phenomenon gate. It shares the same 90% unique documents at the exact same positions as the repetition conditions; only the 10% component uses fresh documents instead of six repeated documents.

### 2. `random`

The six repeated document identities each occur exactly 1,386 times and are randomly assigned to the fixed repeated slots.

This is the closest analogue of the seed paper's random interleaving.

### 3. `clustered`

All 1,386 copies of one repeated document occupy consecutive positions **in repeat-slot order**, then all copies of the next document, etc.

The repeated slots themselves are unchanged; only identity assignment changes.

### 4. `even`

The schedule cycles through all six repeated identities once per round, rotating their order between rounds. Each document is therefore revisited at nearly uniform temporal intervals across the run.

---

## What is exactly identical across `clustered`, `random`, and `even`

The schedule builder asserts, before training:

- identical total number of blocks;
- identical repeated fraction;
- identical repeated slots;
- identical non-repeated document IDs;
- identical non-repeated IDs at the **same exact positions**;
- identical six repeated document IDs;
- identical multiplicity `R=1386` for every repeated document;
- identical complete repeated multiset;
- identical model architecture and optimizer settings.

Only this mapping changes:

```text
fixed repeated slot -> repeated document identity
```

An `audit.json` contains SHA-256 hashes for the schedules, shared unique positions, repeat-slot positions, and repeated multiset.

The code also computes the realized gap distribution. In the locked pilot geometry, a smoke construction gives roughly:

```text
clustered mean gap ≈ 10 blocks
random mean gap    ≈ 60 blocks
six-way even gap   ≈ 60 blocks, but with much lower gap variance
```

The scientifically decisive contrast is **clustered vs even**, because that produces the largest controlled change in exposure spacing while preserving the exact multiset.

`random` anchors the experiment to the seed paper and tells us where ordinary shuffling lies between the two schedules.

---

## Two gates, in the correct order

### Gate A — reproduce the seed phenomenon

First require:

```text
L_random - L_fresh > max(
    5 * held-out evaluation SE,
    0.5% * L_fresh
)
```

The 0.5% relative-loss floor is frozen in `configs/g0.json` before real scores are observed. The seed paper reports loss bumps on the order of a few percent, so this is deliberately conservative.

If this gate fails:

> `PILOT_SETUP_FAIL_SEED_DAMAGE_NOT_REPRODUCED`

Stop. Do not discuss spacing. A setup that cannot reproduce repetition damage is not evidence that spacing is irrelevant.

### Gate B — does spacing materially change that damage?

Primary causal effect:

```text
Delta_spacing = L_clustered - L_even
```

The pilot calls the effect practically large only if:

```text
|Delta_spacing| > max(
    5 * held-out evaluation SE,
    25% * (L_random - L_fresh)
)
```

This scale is intentionally tied to the magnitude of the seed phenomenon. We are not interested in a statistically detectable but scientifically tiny schedule effect.

Both directions are scientifically valid:

- `L_clustered > L_even`: seeing the same example again too soon is more damaging;
- `L_clustered < L_even`: concentrated duplication is surprisingly less damaging than persistent spaced replay;
- large but non-monotonic `clustered/random/even`: spacing matters, but the simple spaced-repetition story is wrong.

---

## Pilot and confirmation

The seed paper itself reports one training run per experimental cell. Topic 13 is stricter.

### Pilot

One matched trial:

```text
seed = 20260822
fresh / clustered / random / even
```

All four conditions use the same initialization fingerprint and are intended to run simultaneously on four GPUs.

Possible verdicts:

- `PILOT_SETUP_FAIL_SEED_DAMAGE_NOT_REPRODUCED`
- `PILOT_WEAK_DO_NOT_TUNE`
- `PILOT_PROMISING_RUN_CONFIRMATION`

Only the last verdict justifies confirmation.

### Frozen confirmation

Exactly three matched seeds are predeclared:

```text
20260822
20260823
20260824
```

Each trial independently rebuilds the fixed slots/order from its seed, while preserving the exact factorial constraints within that trial.

The confirmation requires:

1. seed repetition damage in at least 2/3 trials;
2. a practically large clustered-vs-even effect in at least 2/3 trials;
3. spacing-effect direction stable in at least 2/3 reproduced trials.

Only then does `analyze.py` emit:

```text
GO_SPACING_IS_CAUSAL
```

Otherwise the topic is not rescued by more schedules, more repeat counts, or cherry-picked seeds.

---

## Why no CEG/CEL in G-0

The seed paper's Compute-Equivalent Gain/Loss requires fitting a no-repetition scaling law across multiple model sizes. Repeating that full scaling-law grid would massively increase cost without helping identify the question here.

Topic 13 therefore uses the most direct observable:

> **final held-out next-token loss at fixed model, fixed token budget, and fixed training multiset.**

If spacing does not visibly separate final loss under this design, the topic is not worth a large scaling-law campaign.

CEG/CEL can be added later only after the phenomenon is established.

---

## Data and evaluation

Source corpus:

```text
HuggingFaceTB/smollm-corpus
subset: fineweb-edu-dedup
```

Tokenizer:

```text
Qwen/Qwen3-0.6B-Base
```

`prepare_corpus.py`:

1. streams FineWeb-Edu-Dedup;
2. makes a deterministic document-level train/eval split from document IDs using split seed 0;
3. tokenizes with Qwen3;
4. keeps only documents with at least 2047 tokens;
5. stores exactly `2047 tokens + EOS` in a NumPy memmap;
6. asserts train/eval document-ID disjointness and no duplicate IDs.

The default held-out set is 2,048 fixed blocks, about 4.2M tokens. Final loss is evaluated once at the final checkpoint, matching the seed paper's final-checkpoint reporting rule. Intermediate evaluation is monitoring only and never used to choose a checkpoint.

---

## Files

```text
13_repetition_temporal_spacing/
├── README.md
├── requirements.txt
├── configs/
│   └── g0.json
├── preflight.py          # exact Qwen3 config / CUDA / LR fail-fast audit
├── prepare_corpus.py     # fixed-length FineWeb-Edu-Dedup corpus
├── schedule.py           # exact fixed-multiset temporal schedules + hashes
├── train.py              # paper-matched 34M pretraining run
├── analyze.py            # matched gates and frozen verdict
├── run_g0.py             # corpus -> schedules -> 4 parallel GPU conditions
└── tests/
    ├── test_schedule.py
    └── test_analysis.py
```

Generated data/results live under `runs/` and are gitignored.

---

## Environment

Do not create a separate environment if an existing modern pretraining environment already has compatible PyTorch/Transformers/Datasets/FlashAttention.

Required capabilities:

- PyTorch with CUDA and fused AdamW;
- Transformers with Qwen3 support (`>=4.51`);
- `datasets` for streaming FineWeb-Edu-Dedup;
- NumPy;
- FlashAttention-2 recommended and locked by default.

Unlike Topic 10/11, this experiment trains from scratch; it downloads only the Qwen3 tokenizer, not pretrained model weights.

---

## Cheap local structural tests

These require no model and no dataset download:

```bash
cd 13_repetition_temporal_spacing
python -m unittest discover -s tests -v
```

To inspect the schedule geometry only:

```bash
python schedule.py \
  --out-dir /tmp/topic13_schedule \
  --total-blocks 83158 \
  --repeat-fraction 0.1 \
  --repeat-count 1386 \
  --seed 20260822
```

Read `/tmp/topic13_schedule/audit.json` before spending GPU compute.

---

## Run G-0

### 1. Pilot — four GPUs, one condition per GPU

```bash
cd 13_repetition_temporal_spacing
NUM_GPUS=4 python run_g0.py --mode pilot --num-gpus 4
```

No DDP is used. Each GPU trains one independent condition, so there is no cross-node or cross-GPU communication bottleneck.

If you only have one GPU:

```bash
python run_g0.py --mode pilot --num-gpus 1
```

The four conditions run sequentially.

### 2. Only if pilot says `PILOT_PROMISING_RUN_CONFIRMATION`

```bash
NUM_GPUS=4 python run_g0.py --mode confirm --num-gpus 4
```

The first seed is automatically reused if its metrics already exist, so confirmation does not waste the pilot compute.

Outputs:

```text
runs/g0/
├── corpus/
├── schedules_seed_*/
├── seed_*/
│   ├── fresh/metrics.json
│   ├── clustered/metrics.json
│   ├── random/metrics.json
│   └── even/metrics.json
├── summary_pilot.{json,md}
└── summary_confirm.{json,md}
```

---

## Non-negotiable kill rules

After the first real pilot, do **not**:

- sweep `R` until a spacing effect appears;
- change the repeated fraction;
- search many model sizes;
- select document categories with a stronger result;
- redefine “spacing” based on observed losses;
- choose a best checkpoint instead of the final checkpoint;
- change the practical-effect thresholds;
- report a spacing conclusion if random repetition did not first beat the fresh baseline;
- add memorization probes or hidden-state analyses to rescue a null result.

A failed pilot is useful: it says either this small-scale seed phenomenon did not transfer to the cleaner fixed-length design, or spacing is not a first-order effect worth pursuing here.

---

## What would make the result worth being excited about?

The desired result is not “one schedule is 0.01% better.” It is:

> **same repeated documents, same exact multiplicities, same 90% unique examples at the same positions, same compute, same model initialization — but materially different final held-out loss solely because duplicate copies reappear on a different timescale.**

That would show that duplicate-data damage is not only a property of a dataset multiset. The temporal organization of pretraining exposure would itself be a causal efficiency variable.

If that separation does not appear cleanly in this experiment, stop.
