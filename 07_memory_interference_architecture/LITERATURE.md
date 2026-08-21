# Literature and Collision Audit

Search date: **2026-08-21**.

The purpose of this document is to distinguish the direct open question from nearby work that already covers “overwriting” or “memory collision.” The target novelty is **classical PI-vs-RI asymmetry across controlled sequence-memory update rules**, not the generic observation that linear memories can forget or overwrite.

## 1. Seed anomaly: PI > RI in LLMs

### Chattaraj & Raj (2026), *Transformers Remember First, Forget Last: Dual-Process Interference in LLMs*

- arXiv:2603.00270
- <https://arxiv.org/abs/2603.00270>
- AB–AC paired-associate paradigm; RI and PI use identical stimulus sequences and different temporal queries.
- Current arXiv text reports 39 complete models, universal PI > RI, Cohen's d = 1.73, RI/PI R² = 0.044, and much stronger model-size predictability for RI than PI.
- The limitation section explicitly states that only Transformer architectures were tested.
- The paper itself asks whether the asymmetry is architectural or learned.

**Relevance:** this is the direct empirical tension. Our topic is a response to an explicit architecture limitation, not a mechanism invented around one linear-attention paper.

## 2. Public data / predecessor code

### Wang & Sun (2025), *Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length*

- public repository: <https://github.com/zhuangziGiantfish/Unable-to-Forget>
- public category/value pool used here:
  <https://github.com/zhuangziGiantfish/Unable-to-Forget/blob/main/testing_data/dict_category_double-word_46-400_v1-1.json>

The repository supplies a 46-category × 400-value pool and PI benchmark infrastructure. The current candidate reuses the value pool but implements its own paired RI/PI scoring so that base recurrent checkpoints can be evaluated without open-ended output parsing.

## 3. Mamba already shows both primacy and recency

### Airlangga et al. (2025), *Emergence of Primacy and Recency Effect in Mamba: A Mechanistic Point of View*

- arXiv:2506.15156
- <https://arxiv.org/abs/2506.15156>
- reports a U-shaped recall profile in Mamba, with mechanisms linked to early-token persistence and recurrent decay.

**Collision implication:** we must not preregister “all recurrent models are recency-biased.” A recurrent state can support both primacy and recency. The hypothesis is therefore a systematic change in PI/RI regime with update rule, not a forced binary Transformer-versus-RNN sign claim.

## 4. Controlled architecture family used in the pilot

### Wang et al. (2025), *A Systematic Analysis of Hybrid Linear Attention*

- arXiv:2507.06457
- <https://arxiv.org/abs/2507.06457>
- open-sources 72 models across 340M/20B and 1.3B/100B settings.
- reports that all models in the study are pretrained on FineWeb-Edu; 1.3B models receive 100B tokens with a common optimization setup.
- includes pure recurrent/linear variants and a standard Transformer baseline.
- model collection: <https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research>

Primary checkpoints used here:

- `m-a-p/transformer_1.3B_baseline`
- `m-a-p/1.3B-100B-GLA-pure`
- `m-a-p/1.3B-100B-DeltaNet-pure`
- `m-a-p/1.3B-100B-GatedDeltaNet-pure`

**Importance:** this substantially improves the first-stage identification over arbitrary public checkpoints because scale/data/token budget belong to one controlled study.

## 5. Linear-memory collision / overwrite work: close but not the same question

### Dowling et al. (2026), *Memory by Design: Probabilistic Sequence Layers*

- arXiv:2605.31163
- <https://arxiv.org/abs/2605.31163>
- derives Linear Attention, GLA, Mamba-2/SSD and DeltaNet-family recurrences from probabilistic memory assumptions;
- studies retrieval dynamics and controlled collisions;
- explicitly connects different update assumptions to different memory behavior.

**Collision risk:** high. A paper that merely says “update rules affect overwriting” is already too close to existing work.

**Remaining gap:** their target is not the classical *directional asymmetry* between old→new and new→old interference measured on an identical information stream.

### Hatamizadeh, Choi & Kautz (2026), *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*

- arXiv:2605.22791
- <https://arxiv.org/abs/2605.22791>
- official code: <https://github.com/NVlabs/GatedDeltaNet-2>
- separates erase and write gates in the recurrent update and reports 1.3B/100B FineWeb-Edu comparisons.

**Collision risk:** very high for any claim phrased as “separating erase and write improves overwriting.”

**Use here:** follow-up architecture only after the behavioral PI/RI interaction survives; not needed to manufacture stage-1 novelty.

### Gupta et al. (2026), *The Query Knows What to Forget: A Second Erase Direction for Linear Attention*

- arXiv:2608.13668, submitted 2026-08-13
- <https://arxiv.org/abs/2608.13668>
- extends GDN2 with a query-derived erase direction to reduce retrieval interference.

**Collision implication:** the overwrite/interference space is moving quickly. Novelty cannot be “better erasing.” The PI/RI directional question needs to remain central.

## 6. FLA implementation status

Official FLA:

- <https://github.com/fla-org/flash-linear-attention>
- current installation guide: <https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md>

As of the 2026 install guide, FLA is split into:

- `fla-core`: kernels/modules/utilities;
- `flash-linear-attention`: layers and **model classes**, depending on `fla-core`.

Therefore this topic intentionally installs the full CUDA package:

```bash
pip install 'flash-linear-attention[cuda]'
```

Using only `fla-core` is insufficient for `fla.models` checkpoint registration.

## 7. Direct-gap search summary

Searches were run for combinations of:

```text
proactive interference retroactive interference Mamba
proactive interference retroactive interference DeltaNet
PI RI asymmetry linear attention
AB-AC interference state space model
Gated DeltaNet proactive interference
linear attention retroactive interference
```

The closest results found were the Transformer-only dual-process paper, Mamba primacy/recency work, and collision/overwrite studies above. No retrieved work directly performs the same classical paired PI-vs-RI asymmetry comparison over a controlled softmax/GLA/Delta/GatedDelta architecture family.

This is a **search result, not a proof of absence**. Collision must be rechecked before a full paper-scale experiment because the area is active in 2026.

## 8. Novelty line we are willing to defend

Potentially novel:

> Under a controlled pretraining family and an identical AB–AC stream, changing the sequence-memory update operator produces a reproducible change in the PI/RI asymmetry, possibly a graded or sign-changing transition.

Not novel enough:

- “DeltaNet overwrites old key–value associations.”
- “GDN2 has an erase gate.”
- “Linear attention suffers memory interference.”
- “Mamba has recency effects.”
- “Gated memories recall better than ungated memories.”

If the experiment reduces to one of those statements, archive or reframe before scaling.
