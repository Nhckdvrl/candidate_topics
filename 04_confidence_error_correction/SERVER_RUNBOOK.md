# Server runbook — Topic 04

Goal: **decide quickly whether the research question is scientifically alive**.

The server environment is assumed to have many independent GPUs/nodes. Use that to parallelize **independent jobs**, not cross-node training.

---

# 0. Setup

From repository root:

```bash
cd 04_confidence_error_correction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install into an existing environment.

Set caches to fast local storage:

```bash
export HF_HOME=/path/to/local/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

---

# 1. Prepare primary candidates

```bash
python code/prepare_candidates.py \
  --dataset mmlu_pro \
  --split test \
  --require-k 10 \
  --output data/mmlu_pro_k10.jsonl
```

Check:

```bash
wc -l data/mmlu_pro_k10.jsonl
head -n 1 data/mmlu_pro_k10.jsonl
```

Expected: roughly ten thousand exact-10-option items.

---

# Optional aggressive path when GPUs are abundant

To reduce the chance that the whole decision is peculiar to one scale, run the **entire G-1 measurement** in parallel on:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
```

This is safe because model choice is evaluated **before correction outcomes exist**. If both models pass G-1, it is scientifically preferable to run G0 discovery on both rather than select the one with the prettier effect.

Do not use 1.5B/3B disagreement as permission to search many more models. Report the disagreement as a scale-dependence clue.

---

# 2. G-1 base scoring — parallelize aggressively

Use 8 GPUs/nodes as independent shards.

On worker `i = 0..7`:

```bash
CUDA_VISIBLE_DEVICES=0 python code/score_mcq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input data/mmlu_pro_k10.jsonl \
  --output results/g1/shard_${i}.jsonl \
  --num-shards 8 \
  --shard-index ${i} \
  --batch-size 16 \
  --prompt-template primary
```

No inter-node communication is required.

Merge:

```bash
python code/merge_jsonl.py \
  --inputs results/g1/shard_*.jsonl \
  --output results/g1/base_scores.jsonl \
  --sort-key id
```

Sanity:

```bash
python code/build_matched_pairs.py \
  --input results/g1/base_scores.jsonl \
  --pairs-output results/g1/matched_pairs.jsonl \
  --eligible-output results/g1/eligible_wrong.jsonl \
  --report-output results/g1/matching_report.json \
  --require-k 10 \
  --min-stability 0.80 \
  --p-caliper 0.02 \
  --high-quantile 0.70 \
  --low-quantile 0.30 \
  --discovery-fraction 0.70 \
  --seed 20260821
```

**STOP here and inspect `matching_report.json`.**

Do not start G0 unless G-1 meets at least the minimal criteria in `VALIDATION.md`.

---

# 3. Prompt-robustness audit

Take a deterministic 20% audit sample:

```bash
python - <<'PY'
import json, hashlib
src='data/mmlu_pro_k10.jsonl'
dst='data/mmlu_pro_k10_audit20.jsonl'
with open(src) as f, open(dst,'w') as g:
    for line in f:
        r=json.loads(line)
        h=int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)
        if h % 5 == 0:
            g.write(json.dumps(r,ensure_ascii=False)+'\n')
PY
```

Score using alternate prompt, again sharded if desired:

```bash
CUDA_VISIBLE_DEVICES=0 python code/score_mcq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input data/mmlu_pro_k10_audit20.jsonl \
  --output results/g1/alt_prompt.jsonl \
  --batch-size 16 \
  --prompt-template alternate
```

Compare with the frozen primary scores:

```bash
python code/audit_prompt_robustness.py \
  --primary results/g1/base_scores.jsonl \
  --alternate results/g1/alt_prompt.jsonl \
  --output results/g1/prompt_audit.json
```

Required audit criteria are in `VALIDATION.md`.

If prompt robustness is poor, stop and fix the measurement before any training.

---

# 4. Build correction cycles

Generate exactly one corrective exposure / semantic item / cycle.

```bash
python code/build_sft_data.py \
  --pairs results/g1/matched_pairs.jsonl \
  --output data/correction_cycles.jsonl \
  --cycles 10 \
  --seed 20260821
```

The file contains both `discovery` and `confirmation` rows.

---

# 5. G0-D discovery training

Run seeds on independent GPUs/nodes.

Example seed 17:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 code/train_correction.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-data data/correction_cycles.jsonl \
  --split discovery \
  --output-dir runs/discovery_seed17 \
  --seed 17 \
  --cycles 10 \
  --learning-rate 1e-5 \
  --per-device-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --max-length 1024
```

Repeat for:

```text
seed 29
seed 43
```

Each run saves:

```text
cycle_01/
...
cycle_10/
train_log.jsonl
```

---

# 6. Evaluate discovery checkpoints

For each seed:

```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluate_checkpoints.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --run-dir runs/discovery_seed17 \
  --pairs results/g1/matched_pairs.jsonl \
  --split discovery \
  --output results/g0/discovery_seed17.jsonl \
  --batch-size 16
```

Repeat for seeds 29 and 43.

Analyze:

```bash
python code/merge_jsonl.py \
  --inputs results/g0/discovery_seed*.jsonl \
  --output results/g0/discovery_all.jsonl

python code/analyze_correction.py \
  --input results/g0/discovery_all.jsonl \
  --split discovery \
  --output results/g0/discovery_report.json
```

At this point:

1. verify aggregate learning is neither broken nor cycle-1 saturated;
2. read the primary paired `auc_gain` high-minus-low effect;
3. read early/late, suppression, interaction diagnostics;
4. freeze interpretation and recipe before opening confirmation results.

---

# 7. G0-C locked confirmation

Only after the discovery recipe/analysis is frozen.

Train confirmation from the **same base model**, not from the discovery checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 code/train_correction.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-data data/correction_cycles.jsonl \
  --split confirmation \
  --output-dir runs/confirmation_seed17 \
  --seed 17 \
  --cycles 10 \
  --learning-rate 1e-5 \
  --per-device-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --max-length 1024
```

Run all three seeds if possible, evaluate exactly as above, then:

```bash
python code/analyze_correction.py \
  --input results/g0/confirmation_all.jsonl \
  --split confirmation \
  --output results/g0/confirmation_report.json
```

Apply the locked decision rules from `VALIDATION.md`.

---

# 8. What to do if the main direction is not supported

Do **not** start metric/model shopping.

Check only the predeclared alternatives:

1. equivalence-style null within ±0.02 AUC;
2. early-vs-late reversal;
3. commitment × target-accessibility interaction;
4. correct-target growth vs old-error suppression dissociation;
5. replicated domain heterogeneity.

If one is strong, phrase the new natural phenomenon in one sentence before adding any new machinery.

Examples:

```text
"Strong misconceptions are not harder to replace, but they are slower to suppress."
```

or

```text
"Conviction matters only when the learner already has partial access to the correction."
```

Register a new hypothesis before running follow-up experiments.

---

# 9. Parallelization strategy for many nodes

Best use of many nodes:

- node/GPU group A: G-1 scoring shards;
- after G-1:
  - node 1: discovery seed 17
  - node 2: discovery seed 29
  - node 3: discovery seed 43
- confirmation seeds on separate nodes after recipe freeze;
- optional Qwen2.5-3B measurement replication on spare nodes.

Avoid multi-node DDP over slow interconnect. These are independent experiments and scale embarrassingly well.

---

# 10. Artifacts to return to the research agent

Always return:

```text
results/g1/matching_report.json
results/g0/discovery_report.json
results/g0/confirmation_report.json
runs/*/train_log.jsonl
```

plus:

```text
git commit hash
exact model revision if pinned
transformers/datasets/torch versions
GPU type
```

Do not return only screenshots or verbal summaries.
