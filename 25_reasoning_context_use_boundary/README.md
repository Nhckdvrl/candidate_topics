# 25 — When Does Test-Time Reasoning Help Context Use, and When Does It Hurt?

**Status: REGISTERED / REPRODUCTION RECEIPT NEXT / G0 FROZEN BUT BLOCKED ON RECEIPT.**

## Natural scientific question

> **What computation requirement determines whether test-time reasoning improves or degrades a model's use of long-context evidence?**

The target is not generic `when should a model think?`, routing, prompt engineering, or another reasoning benchmark. The motivating object is a published sign tension on the same Qwen3 model family/interface:

- retrieval-heavy long-context evaluation can become worse with thinking;
- noisy multi-hop evidence use can become substantially more robust with thinking.

The project asks whether that tension can be localized inside one matched experimental object by changing only the requested computation while holding the evidence/context fixed.

---

## Why this numbered topic is allowed despite the later search-log downgrade

`advisor_topic_search/ACTIVE_CANDIDATES.md` later downgraded the Round-09 lead for **advisor fit / external crowding**, not because a local frozen experiment falsified it.

That matters under this repository's status policy:

- Topic 13/20/21-style scientific negatives cannot be silently reopened;
- a search-ranking downgrade with no local scientific result can be elevated for a direct falsification run;
- once numbered, this README and actual local results outrank the old search-log status.

Registration therefore does **not** assert that the paper story is alive. It only authorizes the following fixed sequence:

```text
official Weakest-Link receipt
        ↓ PASS only
matched atomic-vs-composed G0
        ↓ PASS only
characterization / mechanism
```

No receipt rescue and no G0 subset/model/prompt fishing are authorized.

---

## External anchor A — Weakest-Link seed

ACL 2026 Main:

**Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck**

Official repository:

`cambridgeltl/weakest-link-effect`

Frozen upstream commit:

```text
9b01abaad354208a6a8fb26c58eb5c330036fb94
```

The official artifact provides:

- deterministic 18-document MuSiQue banks;
- Qwen3-8B thinking/non-thinking support on the same checkpoint;
- official prompt construction and answer extraction;
- official Exact Match scorer;
- spread/cross/gold-only runners;
- paper-number verification utilities.

Published MuSiQue gold-only anchors:

```text
Qwen3-8B         42.46 EM
Qwen3-8B-Think   44.70 EM
```

The paper additionally reports that Qwen3-8B-Think can match or exceed its gold-only baseline under the full noisy 18-document setting.

### Important artifact audit

Do **not** use the current `scripts/infer/musique_gold-ablation.sh` as the scientific contract. At the frozen commit its orchestration has drifted: the tail section labeled as the 8B thinking run points at a 4B thinking model and an earlier completion check is commented while the shell still reads its variable.

The receipt therefore calls the official Python entrypoint directly:

```text
src.infer.entity.run_ablation
```

This is an engineering bypass only. It does not change dataset/model/prompt/scorer/seed.

---

## External anchor B — retrieval-side tension

The Qwen3 technical evaluation reports lower RULER performance for Qwen3-8B in thinking mode than non-thinking mode (Round 09 records 84.4 vs 89.1 average) and notes that unnecessary reasoning can interfere with retrieval-oriented tasks.

A local exact RULER receipt is desirable if a clean official execution contract is available, but it is **not** required to run G0. The mandatory local prerequisite is the Weakest-Link seed because G0 is built directly on that artifact.

---

# Phase R — mandatory reproduction receipt

Run the official Qwen3-8B Weakest-Link contract before any novel experiment.

The receipt generates four objects from the same 18-doc bank:

1. Qwen3-8B gold-only;
2. Qwen3-8B-Think gold-only;
3. Qwen3-8B no-MFAI (`na`) Spread over all 3 buckets × 5 distances;
4. Qwen3-8B-Think on the exact same `na` Spread cells.

Frozen seed settings:

```text
model             Qwen/Qwen3-8B
upstream commit   9b01abaad354208a6a8fb26c58eb5c330036fb94
bank              processed MuSiQue 18-doc bank, seed 42
spread prompt id  22
gold-only prompt  upstream default prompt id 0
temperature       0.0
top_p             1.0
seed              42
no-think max out  3000
think max out     10000
```

### Receipt gate

Round 09 explicitly forbids inventing a numerical closeness threshold for a reproduction receipt. Therefore `seed_receipt.py` records the exact published gold-only reference values but gates only on the paper's qualitative relations plus complete support:

```text
all expected files complete on the exact same item IDs
think gold-only >= non-think gold-only
think noisy pooled >= think gold-only
think noisy pooled > non-think noisy pooled
```

Verdict:

```text
SEED_RELATION_REPRODUCED
or
SEED_RELATION_NOT_REPRODUCED
```

If the latter occurs, stop Topic 25. Do not change model, prompt, sampling, subset, seed, or context layout to make the seed work.

---

# G0 — matched atomic retrieval vs composed integration

G0 uses the same Weakest-Link 18-document MuSiQue bank and the same Qwen3-8B checkpoint.

The source dataset `Shahar6000/MoreDocsSameLen` already exposes `question_decomposition`, including for each step:

```text
question
answer
paragraph_support_idx
```

The G0 code pins the source dataset revision and joins bank examples by exact ID, falling back only to unique exact question equality. No fuzzy matching is used.

## Matched object

For each selected 2-hop item and each bucket placement:

```text
same item
same 18 documents
same distractor order
same two supporting documents
same evidence positions
same model checkpoint
same prompt template
same decoder settings
```

Only the requested computation changes:

```text
atomic_0      first single-hop decomposed question
atomic_1      second decomposed question with earlier #k placeholders
              resolved using the released gold intermediate answer
composed      original 2-hop final question
```

Thinking is independently toggled on/off using the upstream Qwen3 API contract.

### Why resolve `#k` placeholders?

MuSiQue decompositions often encode a later step as e.g. `#1 >> headquarters location`. Leaving `#1` unresolved would make the atomic condition a malformed task; asking the model to infer `#1` would silently reintroduce composition.

G0 therefore substitutes only the **released earlier intermediate answer** into later atomic questions. This defines the intended atomic/local-evidence task rather than a second hidden multi-hop task.

---

## Frozen G0 panel

```text
model               Qwen/Qwen3-8B
items               256 exact eligible 2-hop items
selection seed      20260825
buckets             beginning, midsection, tail
within-bucket dist  1
query types          atomic_0, atomic_1, composed
thinking             off, on
prompt id            22
temperature          0.0
top_p                1.0
no-think max out     3000
think max out        10000
```

Item selection is by a deterministic SHA-256 rank of `seed:item_id`, not first-N order and not outcome filtering.

`atomic_both_correct` is the item-level atomic endpoint: both released decomposed steps must be answered correctly. This gives one paired binary endpoint per item, matching the composed endpoint's item-level granularity.

---

# Primary observable

For each bucket `b`:

```text
D_atomic(b)   = Acc(atomic_both, think) - Acc(atomic_both, no-think)
D_composed(b) = Acc(composed, think)    - Acc(composed, no-think)

Interaction(b) = D_composed(b) - D_atomic(b)
```

The pooled interaction averages the same item across the three frozen bucket placements.

A paired bootstrap resamples **item IDs**, not individual prompt rows, so the dependence among bucket placements and query conditions stays inside the resampled unit.

---

# Frozen G0 gates

These are paper-worthiness / meaningful-regime gates for the new experiment, not reproduction thresholds:

```text
selected eligible items                    == 256
all expected model calls                    complete
no-think pooled atomic_both accuracy        >= 0.30
no-think pooled composed accuracy           >= 0.15
pooled composed thinking gain               >= 0.08
pooled atomic thinking gain                 <= 0.03
pooled interaction                          >= 0.08
paired bootstrap 90% CI lower(interaction)  > 0
positive interaction in buckets             >= 2 / 3
```

Primary verdict:

```text
GO_MATCHED_BOUNDARY
or
STOP_MATCHED_BOUNDARY
```

A stronger diagnostic is reported but is **not a separate rescue gate**:

```text
atomic thinking gain < 0
and
composed thinking gain > 0
```

If true, call it a matched sign reversal. If false but all preregistered gates pass, the result is a computation-selective benefit rather than a literal sign flip.

If G0 fails, do not select particular decomposition types, buckets, item families, prompts, or alternative Qwen models to recover the interaction.

---

# What G0 can and cannot establish

A positive G0 supports:

> Under matched long/noisy evidence, test-time reasoning benefits composed evidence integration substantially more than the corresponding atomic evidence queries.

It does **not** establish:

- why thinking helps;
- that attention is the mechanism;
- that generated reasoning tokens themselves are causal;
- that hop count is the universal boundary;
- that an adaptive router is already justified.

No hidden-state or attention mechanism work is authorized before G0 passes.

---

# Mother-topic branch map after a positive G0

## Branch A — boundary characterization

Atomic retrieval → 1-step inference → 2-hop → 3/4-hop; position and noise can be crossed only with predeclared panels.

## Branch B — computation dynamics

Evidence re-reading, reference order, revision, and the seed repo's attention pipeline become legitimate after the matched behavioral interaction exists.

## Branch C — reasoning-budget dose response

Ask whether atomic tasks enter overthinking earlier while composition requires a minimum budget.

## Branch D — evidence topology

Test whether vertical chains vs horizontal evidence aggregation determine the value of thinking more strongly than nominal context length.

## Branch E — method

Only then consider retrieval/integration phase separation or reasoning allocation conditioned on integration structure.

A failed branch does not authorize post-hoc feature search; a new branch requires either this predeclared map or a new external observation.

---

# Files

```text
README.md
VALIDATION_AUDIT.md
seed_receipt.py
run_seed_receipt.sh
g0_atomic_vs_composed.py
run_g0.sh
requirements.txt
tests/test_helpers.py
```

Outputs are written under `artifacts/` and are intentionally not committed by default.

---

# Execution summary

With the frozen Weakest-Link repository checked out and a vLLM OpenAI-compatible server serving `Qwen/Qwen3-8B`:

```bash
cd 25_reasoning_context_use_boundary
pip install -r requirements.txt

export UPSTREAM_REPO=/path/to/weakest-link-effect
export API_URL=http://localhost:8000/v1

bash run_g0.sh
```

`run_g0.sh` refuses to run if the upstream commit differs from the pinned receipt commit. It runs unit tests, completes/reuses the official receipt, checks the receipt verdict, and only then launches the novel G0.

No model weights, HF cache, 12-GB upstream raw result archive, or generated CoT traces should be committed.