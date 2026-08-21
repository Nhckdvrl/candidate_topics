# Validation Contract — 07 Memory-Update Architecture × PI/RI

This document freezes the first validation logic before GPU results are inspected.

## 0. Claim under test

Primary claim candidate:

> The update rule of a sequence memory systematically changes the direction/magnitude of proactive-versus-retroactive interference under the same conflicting information stream.

The first pilot does **not** test a representation-level mechanism. It tests only whether an architecture-level behavioral interaction exists strongly enough to justify further work.

## 1. Why the observable identifies the question

For each episode, RI and PI receive exactly the same ordered assignments. The only manipulated variable is which temporal endpoint is queried:

- RI target = first value for the key;
- PI target = most recently presented value for the key.

Therefore, within an episode/key pair, a change in `Accuracy_RI - Accuracy_PI` cannot be attributed to a different stimulus stream.

Across the primary model family, the same generated episodes, queried keys and candidate sets are reused. The M-A-P study further reduces training-family confounding by providing the compared architectures at the same 1.3B/100B-token scale under one pretraining study.

Remaining limitation: architecture and learned parameters are still jointly changed. Even matched training does not isolate a single algebraic term. A later controlled interpolation/ablation is needed only **after** a robust architecture interaction exists.

## 2. Frozen data source

Public source:

```text
zhuangziGiantfish/Unable-to-Forget
 testing_data/dict_category_double-word_46-400_v1-1.json
```

Pinned Git blob:

```text
15442a4cd50a7af5b9362620bbf43f6a0365965a
```

The downloader recomputes the Git blob hash and aborts if upstream bytes change.

### Episode construction

For each `(seed, episode_id, U)`:

1. sample `num_keys` categories;
2. sample `U + 1` distinct values for every category;
3. present one initial binding for every category in randomized order;
4. for update round `r = 1..U`, present exactly one new value for every category, randomizing category order inside that round;
5. select frozen query keys from the episode;
6. reuse this exact stream for both RI and PI.

The update ordering is round-balanced rather than a global shuffle. This prevents one key's latest update from accidentally occurring much earlier than another key's latest update solely because all updates were globally permuted.

## 3. Frozen discovery configuration

File: `configs/pilot.yaml`

```text
seed                  = 20260821
num_keys              = 8
U                     = {1, 3, 7, 15} later updates/key
episodes_per_level    = 6
queries_per_episode   = 4
candidate_metric      = mean_logprob
candidate_batch_size  = 16
context_safety_margin = 64 tokens
```

Per model and level this yields 24 paired RI/PI query cells before skips.

Do not change these values after looking at pilot outcomes.

## 4. Frozen primary model family

Primary models:

```text
Transformer    m-a-p/transformer_1.3B_baseline
GLA            m-a-p/1.3B-100B-GLA-pure
DeltaNet       m-a-p/1.3B-100B-DeltaNet-pure
GatedDeltaNet  m-a-p/1.3B-100B-GatedDeltaNet-pure
```

Why this family is preferred:

- the source study reports all 1.3B models trained on 100B FineWeb-Edu tokens;
- the study uses the same broad optimization recipe and evaluates the variants as one controlled architecture family;
- the public collection contains pure linear variants and the Transformer baseline;
- tokenizer compatibility can be checked before loading weights.

Mamba-2 is **not** part of the primary gate because the most obvious public Mamba-2 checkpoint comes from a different pretraining family. GDN2 is also not forced into stage 1: its main value is as a follow-up erase/write architecture after the basic interaction survives.

## 5. Preflight gates — failure means INVALID, not a negative scientific result

Run:

```bash
python -m memory_interference.preflight --config configs/pilot.yaml
```

The preflight must pass all of the following before full inference:

### P1 — dataset integrity

The public JSON downloads and matches the pinned Git blob.

### P2 — checkpoint/config availability

Every registered checkpoint must resolve via Hugging Face config/tokenizer loading.

### P3 — shared tokenizer audit

The primary family must produce the same fingerprint on fixed probe strings. If not, stop and document the discrepancy before inference; do not silently compare different tokenizations.

### P4 — context safety

Sampled prompts at all frozen update levels must stay below each checkpoint's configured context limit minus the safety margin.

### P5 — code tests

```bash
python -m pytest -q
```

All tests must pass. The tests cover deterministic data construction, round-balanced updates, shared-stream prompts, metric sign convention, decision logic, and checksum logic.

Any P1–P5 failure is an engineering/setup problem. It must not be counted as evidence for or against the research hypothesis.

## 6. Scoring protocol

### 6.1 Candidate set

For a queried key, candidates are **only values actually assigned to that same key in the episode**.

This deliberately turns the first pilot into temporal-selection measurement rather than open-ended generation. It removes parse failures and most instruction-following noise from base models.

### 6.2 Teacher-forced continuation score

For every candidate `c`:

```text
prompt + c
```

is tokenized as a whole. Candidate tokens are scored from causal next-token logits.

Primary candidate score:

```text
mean_logprob(c)
```

The full sum log probability is also recorded.

Why mean log probability is frozen here: candidate values have variable token lengths, while the primary architecture family shares a tokenizer. Token-normalization avoids making longer category values mechanically less likely solely because they contain more pieces. Since the exact same candidate set is used for RI and PI in each pair, any residual length effect is paired.

### 6.3 Tokenization boundary guard

The code compares tokenization of `prompt` with the common prefix of `prompt + candidate` and aborts if the boundary changes by any token. This catches BPE/SentencePiece delimiter pathologies rather than silently scoring different prefixes.

## 7. Primary metrics

For model `m` and update count `U`:

```text
RI_acc(m,U) = P(top candidate == initial value)
PI_acc(m,U) = P(top candidate == latest value)
I(m,U)      = RI_acc(m,U) - PI_acc(m,U)
```

Equivalent error form:

```text
I = Error_PI - Error_RI
```

Sign convention:

```text
I > 0  => PI worse than RI (primacy-biased)
I < 0  => RI worse than PI (recency/overwrite-biased)
```

The summary also integrates accuracy over a log-scaled **later-update count** `log10(U+1)` for a compact within-pilot AUC. This quantity is source-inspired but **must not be called the source paper's RIES/PIES**, because the source paper uses a different interference-level definition and much larger levels.

### Primary architecture contrast

Frozen primary comparison:

```text
Delta_I = mean_U I(Transformer,U) - mean_U I(GatedDeltaNet,U)
```

The bootstrap is paired on common:

```text
(episode_id, query_key, U)
```

cells across the two models.

GLA and DeltaNet are predeclared architecture-family context. They are not searched post hoc to choose whichever gives the largest result.

## 8. Diagnostics that do not redefine the primary endpoint

The following are mandatory audits but are not alternative primary metrics:

1. `token_audit.json`
   - RI/PI mean target token count;
   - maximum boundary shift;
   - maximum prompt length.
2. `intrusions.json`
   - where wrong predictions fall in the queried key's history (`0 = first`, `1 = latest`).
3. target-rank information in `results.jsonl`.
4. skipped-row rate.

If diagnostics expose a coding artifact, stop and classify the run as INVALID. Do not simply choose a different metric that makes the desired result appear.

## 9. Frozen discovery decision

Run:

```bash
python -m memory_interference.decide outputs/architecture_pi_ri_pilot --bootstrap 5000
```

The decision file uses these rules.

### 9.1 INVALID

If more than 5% of result rows were skipped for context/setup reasons, the pilot is invalid. Fix engineering before interpreting it.

### 9.2 PARADIGM_FAIL

If:

```text
mean_U I(Transformer,U) <= 0
```

then the motivating Transformer PI>RI phenomenon did not reproduce under this base-model constrained-scoring setup.

Stop the architecture claim. Do **not** interpret a linear-model sign difference as the desired result because the measurement itself has not reproduced the starting anomaly.

A later redesign would be a new registered measurement, not an informal rescue of this run.

### 9.3 STRONG_GO

Require both:

```text
Delta_I >= 0.10
bootstrap 95% lower bound > 0
```

and at least three of four update levels satisfy:

```text
I_Transformer > 0
I_GatedDeltaNet < 0
```

This is the clean primacy-to-recency sign-transition case.

### 9.4 GO_TO_LOCKED_CONFIRMATION

Require:

```text
Delta_I >= 0.10
bootstrap 95% lower bound > 0
```

without requiring a sign reversal. A robust magnitude change is enough to justify confirmation; the scientific claim should then be about a changed interference regime, not necessarily a categorical sign flip.

### 9.5 KILL

If:

```text
abs(Delta_I) < 0.05
```

kill the architecture-separation hypothesis at the cheap pilot. Do not start adding GDN2, Mamba, prompt variants, layer probes or gate interventions to find a cell that works.

### 9.6 INCONCLUSIVE_DO_NOT_TUNE

Everything else is inconclusive. Preserve the run and do not tune the metric/prompt/model roster against it.

This status is deliberately uncomfortable: it prevents a moderately noisy result from becoming an invitation to search over degrees of freedom.

## 10. Locked independent confirmation

Only discovery outcomes `GO_TO_LOCKED_CONFIRMATION` or `STRONG_GO` authorize:

```bash
./run_confirmation.sh
```

`configs/confirm.yaml` is already frozen:

```text
seed               = 20260822
episodes_per_level = 18
same update levels
same query count
same models
same prompt
same candidate metric
same decision function
```

No discovery examples are reused because episode construction changes with the seed.

### Confirmation pass criterion

Use the same positive architecture-gap criterion:

```text
Delta_I >= 0.10
bootstrap 95% lower bound > 0
```

If confirmation fails this criterion, archive the strong architecture claim. Do not average discovery and confirmation to rescue it.

## 11. What happens after confirmation?

Only after a locked positive confirmation should a mechanistic/causal follow-up begin.

Preferred next experiment is **not** an unrestricted probe search. It should manipulate the update rule along a pre-specified axis, for example:

- add GDN2 as an erase/write-decoupled architecture;
- use a matched update-rule interpolation/ablation if code permits it without retraining confounds;
- train small matched models only if the intervention cannot be implemented on an existing controlled family.

The next claim would be stronger:

```text
change update dynamics -> change PI/RI asymmetry
```

rather than merely:

```text
different named checkpoints have different I
```

## 12. Hard stop against complexity creep

The following are explicitly **not allowed as pilot rescue operations** after a negative result:

- choosing a different layer;
- probing hidden states;
- clamping erase/write gates;
- changing `INITIAL`/`LAST` wording after inspecting outcomes;
- selecting only convenient semantic categories;
- filtering token lengths based on observed architecture effects;
- replacing mean log probability with whichever score gives a larger effect;
- adding many model families and reporting the best-separated pair;
- moving to longer contexts solely because short contexts failed.

A genuine external observation can motivate a new registered experiment. A negative pilot cannot be rescued by expanding researcher degrees of freedom.
