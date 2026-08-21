# 07 — Old Blocks New, or New Erases Old?

**Status:** registered candidate / cheap architecture pilot ready to run  
**Primary question:** does the memory-update rule of a sequence model systematically change whether conflicting associations produce more **proactive interference (PI)** or **retroactive interference (RI)**?

## Natural question

When old and new memories conflict, why does one system become **primacy-biased** (old blocks new) while another becomes **recency-biased** (new erases old)?

The classical distinction is:

- **PI:** older information interferes with retrieval of the new association;
- **RI:** newer information interferes with retrieval of the old association.

This question is interesting before naming any neural architecture. Modern sequence models are useful because they instantiate materially different ways of storing and updating past information.

## External anomaly that motivates the topic

Chattaraj & Raj, *Transformers Remember First, Forget Last: Dual-Process Interference in LLMs* (arXiv:2603.00270, 2026) apply an AB–AC interference paradigm to LLMs. Their current arXiv version reports that all 39 models with complete paired data show PI > RI, with a large paired effect (Cohen's d = 1.73). RI and PI endurance scores are weakly related (R² = 0.044), and parameter count predicts RI resistance much more strongly than PI resistance. The paper explicitly lists the fact that only Transformer architectures were tested as a limitation and asks whether the asymmetry is architectural or learned.

Source: <https://arxiv.org/abs/2603.00270>

That creates a direct architecture question rather than a mechanism invented after looking inside one model.

## Why linear/recurrent memories are a natural axis

The relevant architecture distinction is not simply “Transformer versus RNN.” The update operators form a more informative progression:

1. **softmax Transformer:** past KV entries remain available as an append-only retrieval history within the context;
2. **GLA-like recurrent memory:** a fixed-size state is continually updated with learned decay/gating;
3. **DeltaNet:** the state uses an error-corrective delta update that can explicitly replace a key–value association;
4. **Gated DeltaNet:** the delta update adds learned forgetting/gating;
5. **Gated DeltaNet-2 (follow-up only):** erase and write are explicitly decoupled.

Recent work already studies collision, overwriting and erase/write dynamics in these memories. Therefore **“DeltaNet can overwrite” is not a publishable claim by itself**. The candidate survives only if the classical PI/RI asymmetry changes systematically with the update operator under a controlled model family.

Closest work is summarized in [LITERATURE.md](./LITERATURE.md).

## The one clean contrast

For a category/key `A`, the model sees one initial binding followed by conflicting rebindings:

```text
A -> B0
...
A -> B1
...
A -> B2
...
```

The **stimulus stream is identical** for both probes. Only the final query changes:

```text
RI: What was the INITIAL value of A?
PI: What was the LAST (most recent) value of A?
```

The primary per-level asymmetry is

```text
I = Error_PI - Error_RI
  = Accuracy_RI - Accuracy_PI
```

Interpretation:

- `I > 0`: PI is worse; primacy protection / old blocks new;
- `I < 0`: RI is worse; recency overwrite / new erases old;
- `I ~= 0`: balanced interference.

No representation probe, layer search, gate clamp or mechanistic intervention is part of the first pilot.

## Important measurement choice

Most available recurrent checkpoints are base LMs, not instruction-tuned chat models. Free generation would therefore mix memory with instruction-following and parsing failures. The pilot instead performs **constrained teacher-forced candidate scoring** over every value that was actually assigned to the queried key.

For each candidate, the code computes full continuation log probabilities under the same prompt. The primary ranking metric is token-normalized mean log probability. Because RI and PI for a given episode use the **same candidate set**, value identity and tokenization are paired rather than condition-specific. The pipeline also logs:

- candidate token counts;
- prompt token counts;
- tokenizer boundary shifts;
- target rank;
- predicted historical position;
- all candidate scores.

This makes tokenization and context-length artifacts auditable rather than hidden.

## Primary pretrained family: matched rather than arbitrary checkpoints

The first pilot uses the open M-A-P family from *A Systematic Analysis of Hybrid Linear Attention* rather than unrelated model families:

| Pilot name | Checkpoint | Memory type |
| --- | --- | --- |
| `transformer_1.3b` | `m-a-p/transformer_1.3B_baseline` | full softmax attention |
| `gla_1.3b` | `m-a-p/1.3B-100B-GLA-pure` | gated linear attention |
| `deltanet_1.3b` | `m-a-p/1.3B-100B-DeltaNet-pure` | delta-rule recurrent memory |
| `gated_deltanet_1.3b` | `m-a-p/1.3B-100B-GatedDeltaNet-pure` | gated delta-rule memory |

The accompanying study reports a controlled 1.3B setting trained on 100B FineWeb-Edu tokens with a shared optimization setup, and the checkpoints are published in one collection. This is substantially cleaner than comparing unrelated Transformer/Mamba/Delta checkpoints.

Paper: <https://arxiv.org/abs/2507.06457>  
Collection: <https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research>

The preflight script checks that the selected checkpoints expose a compatible tokenizer fingerprint and that the sampled prompts stay safely within their configured context windows.

## Data

The pilot reuses the public 46-category × 400-value pool from the predecessor PI benchmark *Unable to Forget*:

<https://github.com/zhuangziGiantfish/Unable-to-Forget/blob/main/testing_data/dict_category_double-word_46-400_v1-1.json>

`python scripts/download_data.py` downloads the exact file and verifies its Git blob SHA:

```text
15442a4cd50a7af5b9362620bbf43f6a0365965a
```

No new annotation is required.

## Pilot design

`configs/pilot.yaml` freezes the first screen:

```text
8 categories per episode
later updates per key U = {1, 3, 7, 15}
6 episodes per level
4 queried keys per episode
RI and PI both evaluated on every queried key
4 matched 1.3B architectures
```

This gives 24 paired RI/PI observations per level per model before any skips. Updates are interleaved **round-by-round**: every category receives exactly one update in each round, with the category order reshuffled within that round. This avoids accidentally giving some categories many more post-update distractors than others.

The short pilot deliberately stays far below the extreme interference levels in the source paper. Its purpose is not to reproduce the source paper's absolute endurance scores; it is the cheapest test of whether update-rule architecture creates a measurable interaction.

## What counts as a real signal?

The primary preregistered architecture contrast is:

```text
Delta_I = mean_level(I_Transformer) - mean_level(I_GatedDeltaNet)
```

The paired bootstrap resamples the same `(episode, query_key, update_level)` cells across architectures.

The automated decision script implements the frozen screen:

- **PARADIGM_FAIL:** matched Transformer does not show positive mean `I`; do not interpret cross-architecture differences before the motivating effect is reproduced in this base-model scoring setup.
- **STRONG_GO:** `Delta_I >= 0.10`, bootstrap 95% lower bound > 0, and Transformer/GatedDeltaNet show opposite signs at at least 3 of 4 update levels.
- **GO_TO_LOCKED_CONFIRMATION:** `Delta_I >= 0.10` and bootstrap 95% lower bound > 0, without requiring a sign flip.
- **KILL:** `|Delta_I| < 0.05`; the cheap screen says the architecture separation is practically small.
- **INCONCLUSIVE_DO_NOT_TUNE:** everything between those cases. Do not rescue it by changing prompts, metrics, models, thresholds, or update levels after seeing the result.

Full details and the second-stage rules are in [VALIDATION.md](./VALIDATION.md).

## Run

Use a fresh CUDA environment. Current FLA installation requires the full `flash-linear-attention` package rather than `fla-core` alone because model classes live in `fla.models`.

```bash
cd 07_memory_interference_architecture
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

First run the smoke path:

```bash
./run_smoke.sh
```

Then the frozen discovery pilot:

```bash
./run_pilot.sh
```

To run only one registered model while debugging:

```bash
./run_pilot.sh --model transformer_1.3b
```

If and only if the discovery decision is `GO_TO_LOCKED_CONFIRMATION` or `STRONG_GO`, run the independent seed/config:

```bash
./run_confirmation.sh
```

See [SERVER_RUNBOOK.md](./SERVER_RUNBOOK.md) for the exact validation order and failure handling.

## Outputs

Each run directory contains:

```text
resolved_config.json
results.jsonl
summary.csv
intrusions.json
token_audit.json
pairwise_bootstrap.json
decision.json       # pilot / confirmation runs
```

`results.jsonl` retains the full candidate score vectors, so later audit does not require re-running inference.

## What this pilot can and cannot claim

A positive result in the matched M-A-P family would support:

> PI/RI asymmetry depends on the sequence memory update architecture under a controlled pretraining family.

It would **not yet prove** that a particular learned gate internally “represents forgetting,” nor that recurrent memory is human-like, nor that all Transformers are primacy-biased and all writable memories are recency-biased.

If the effect survives locked confirmation, follow-up can add GDN2 or a controlled update-rule interpolation. If it fails, the topic should be archived rather than rescued with a mechanistic search.
