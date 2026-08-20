# Server runbook — Topic 04 G-1v2

Goal: **decide as quickly as possible whether Topic 04 has a valid measurement. Do not run G0 until G-1v2 passes.**

---

# 0. Sync and tests

```bash
git pull
cd 04_confidence_error_correction
python -m unittest tests/test_g1v2_math.py -v
```

All three tests must pass.

---

# 1. Zero-GPU reaggregation of existing 1.5B results

The existing v1 file already contains all 10 mapped permutation distributions.

```bash
mkdir -p results/g1v2

python code/reaggregate_g1v2.py \
  --input results/g1/base_scores.jsonl \
  --output results/g1v2/base_scores_reaggregated.jsonl
```

Now match with the repaired measurement:

```bash
python code/build_matched_pairs.py \
  --input results/g1v2/base_scores_reaggregated.jsonl \
  --pairs-output results/g1v2/matched_pairs.jsonl \
  --eligible-output results/g1v2/eligible_wrong.jsonl \
  --report-output results/g1v2/matching_report.json \
  --require-k 10 \
  --p-caliper 0.02 \
  --question-length-ratio 1.35 \
  --answer-length-ratio 1.50 \
  --high-quantile 0.70 \
  --low-quantile 0.30 \
  --discovery-fraction 0.70 \
  --seed 20260821
```

Read `matching_report.json`.

### Immediate decision

```text
n_pairs < 200
    => KILL Topic 04. Stop.

200 <= n_pairs < 300
    => borderline; continue only to reliability audit, do not start G0.

n_pairs >= 300
    => continue to reliability audit.
```

Do not alter quantiles/calipers/K to increase pair count.

---

# 2. Deterministic 20% audit subset

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

---

# 3. Fresh v2 scoring on the audit subset

Use a conservative batch size (`8` or lower). The old shard 5/6 logs showed allocator warnings, so do not push memory merely to save minutes.

## A — primary prompt + primary balanced family

```bash
CUDA_VISIBLE_DEVICES=0 python code/score_mcq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input data/mmlu_pro_k10_audit20.jsonl \
  --output results/g1v2/audit_A_primary_cyclic.jsonl \
  --batch-size 8 \
  --prompt-template primary \
  --permutation-scheme cyclic
```

## B — primary prompt + independent balanced family

```bash
CUDA_VISIBLE_DEVICES=0 python code/score_mcq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input data/mmlu_pro_k10_audit20.jsonl \
  --output results/g1v2/audit_B_primary_hashed.jsonl \
  --batch-size 8 \
  --prompt-template primary \
  --permutation-scheme hashed_cyclic
```

## C — alternate prompt + primary balanced family

```bash
CUDA_VISIBLE_DEVICES=0 python code/score_mcq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --input data/mmlu_pro_k10_audit20.jsonl \
  --output results/g1v2/audit_C_alternate_cyclic.jsonl \
  --batch-size 8 \
  --prompt-template alternate \
  --permutation-scheme cyclic
```

---

# 4. Reliability reports

Family A vs B:

```bash
python code/audit_prompt_robustness.py \
  --primary results/g1v2/audit_A_primary_cyclic.jsonl \
  --alternate results/g1v2/audit_B_primary_hashed.jsonl \
  --audit-kind permutation_family \
  --output results/g1v2/permutation_family_audit.json
```

Prompt A vs C:

```bash
python code/audit_prompt_robustness.py \
  --primary results/g1v2/audit_A_primary_cyclic.jsonl \
  --alternate results/g1v2/audit_C_alternate_cyclic.jsonl \
  --audit-kind prompt \
  --output results/g1v2/prompt_audit.json
```

Both must satisfy:

```text
wrong_concentration Spearman >= .70
p_correct Spearman           >= .90
median semantic JS           <= .05
```

If either fails: **KILL Topic 04.**

Exact top-wrong agreement is diagnostic only.

---

# 5. Response-channel audit

From the fresh audit A file, summarize:

```bash
python - <<'PY'
import json, statistics
rows=[json.loads(x) for x in open('results/g1v2/audit_A_primary_cyclic.jsonl') if x.strip()]
m=[r['mean_label_mass'] for r in rows if r.get('mean_label_mass') is not None]
g=[r['greedy_is_allowed_label_rate'] for r in rows if r.get('greedy_is_allowed_label_rate') is not None]
print('n',len(rows))
print('median_mean_label_mass',statistics.median(m))
print('median_greedy_is_allowed_label_rate',statistics.median(g))
PY
```

Warning thresholds:

```text
median mean_label_mass < .50
median greedy_is_allowed_label_rate < .80
```

If either is poor, manually inspect a random/hash-fixed sample before deciding. Do not filter low-mass items after seeing outcomes.

---

# 6. Optional predeclared 3B replication

If 1.5B passes or is borderline-but-not-hard-failed, run the same audit / full G-1v2 on:

```text
Qwen/Qwen2.5-3B-Instruct
```

Use many independent shards. Do not move to additional model sizes.

For a full 3B scoring pass:

```bash
CUDA_VISIBLE_DEVICES=0 python code/score_mcq.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/mmlu_pro_k10.jsonl \
  --output results/g1v2_3b/shard_${i}.jsonl \
  --num-shards 8 \
  --shard-index ${i} \
  --batch-size 8 \
  --prompt-template primary \
  --permutation-scheme cyclic
```

Merge, then run `build_matched_pairs.py` with exactly the same settings.

---

# 7. Only if G-1v2 passes: G0

Do not start corrective SFT before the above decision.

If v2 passes, existing G0 code remains the route:

```text
build_sft_data.py
train_correction.py
evaluate_checkpoints.py
analyze_correction.py
```

`evaluate_checkpoints.py` has been updated to use the same v2 log-space semantic scorer as G-1.

The discovery/confirmation rules in `VALIDATION.md` remain binding.

---

# 8. What to return

Always return:

```text
results/g1v2/matching_report.json
results/g1v2/permutation_family_audit.json
results/g1v2/prompt_audit.json
response-channel summary
exact git commit
model revision
torch / transformers versions
GPU
```

Final verdict must be one of:

```text
G1V2_PASS
G1V2_BORDERLINE
KILL_TOPIC04
```

Do not run G0 after `KILL_TOPIC04`.
