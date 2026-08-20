# Topic 05 — Fast Validation Runbook

This runbook is optimized for **fast falsification with abundant independent GPUs/nodes**, not for minimizing compute. Cross-node communication is unnecessary: checkpoint sampling, re-entry inference, and most scoring are embarrassingly parallel.

Read `VALIDATION.md` first. If this runbook and `VALIDATION.md` ever disagree, the validation contract wins.

---

## 1. Hardware strategy

Recommended cluster layout:

- one Qwen2.5-7B model replica per GPU;
- if eight nodes are available, assign one public UWNSL checkpoint to each node and run all checkpoints concurrently;
- otherwise use `scripts/run_4gpu_sharded.sh` to shard requests across four GPUs on a node;
- avoid tensor parallelism for 7B unless a particular GPU cannot hold the model; independent replicas give better throughput and require no inter-GPU communication;
- reserve one 4-GPU node for the Qwen2.5-32B judge if using the hybrid scorer.

Public RL checkpoints:

```text
UWNSL/Qwen2.5-7B-deepscaler_4k_step_32
UWNSL/Qwen2.5-7B-deepscaler_4k_step_64
UWNSL/Qwen2.5-7B-deepscaler_4k_step_96
UWNSL/Qwen2.5-7B-deepscaler_4k_step_128
UWNSL/Qwen2.5-7B-deepscaler_4k_step_160
UWNSL/Qwen2.5-7B-deepscaler_4k_step_192
UWNSL/Qwen2.5-7B-deepscaler_4k_step_224
UWNSL/Qwen2.5-7B-deepscaler_4k_step_256
```

Primary G-1 budget:

```text
500 MATH-500 problems × 8 checkpoints × 16 samples = 64,000 generations
```

This is intentionally larger than the minimum needed for a smoke test because the goal is to classify robust temporal states, not save inference tokens.

---

## 2. Environment

```bash
git clone https://github.com/Nhckdvrl/candidate_topics.git
cd candidate_topics/05_temporal_forgetting_reentry
pip install -r requirements.txt

# Seed repo: used for official PRIME/MATH grading code and reference assets.
git clone https://github.com/uw-nsl/Temporal_Forgetting.git external/Temporal_Forgetting
```

Before a long launch:

```bash
python -m py_compile code/*.py
python -m unittest discover -s tests -v
```

---

# Phase 0 — Optional zero-GPU/low-GPU smoke test on official 64-response release

The seed repo contains `sampling_64_responses.zip` for AIME24/AIME25/AMC.

```bash
cd external/Temporal_Forgetting
unzip -n sampling_64_responses.zip
cd ../..

python code/convert_official_release.py \
  --root external/Temporal_Forgetting \
  --output results/official64.jsonl

python code/validate_dataset.py --input results/official64.jsonl
python code/build_forgotten_set.py \
  --input results/official64.jsonl \
  --output results/official64_groups.jsonl \
  --min-samples 16
python code/analyze_state_dynamics.py \
  --groups results/official64_groups.jsonl \
  --output-json results/official64_state_dynamics.json
```

Purpose: check that the strict repeated-sampling definition is not obviously empty and inspect file-format compatibility. These tiny competition sets are **not** the primary feasibility gate.

---

# Phase 1 — Prepare MATH-500

```bash
python code/prepare_math500_requests.py \
  --output data/math500_requests.jsonl
```

Expected: 500 rows with stable problem IDs, prompt, gold answer, worked solution, subject, and level.

---

# Phase 2 — Sample all eight checkpoints

Primary generation parameters are fixed to match the seed sampling regime:

```text
n=16
temperature=0.6
top_p=0.95
max_tokens=8192 initially
```

## Four GPUs on one node

Example for `step_256`:

```bash
MODEL=UWNSL/Qwen2.5-7B-deepscaler_4k_step_256 \
INPUT=data/math500_requests.jsonl \
OUTDIR=results/raw_step256 \
N=16 \
NUM_SHARDS=4 \
bash scripts/run_4gpu_sharded.sh
```

Repeat independently for every checkpoint. Since the eight checkpoints do not communicate, launch them across nodes simultaneously.

## More than one four-GPU node for the same checkpoint

Set global shard count to total GPUs and give every node a distinct offset. Example: three four-GPU nodes = 12 shards.

Node 0:

```bash
NUM_SHARDS=12 SHARD_OFFSET=0 ... bash scripts/run_4gpu_sharded.sh
```

Node 1:

```bash
NUM_SHARDS=12 SHARD_OFFSET=4 ... bash scripts/run_4gpu_sharded.sh
```

Node 2:

```bash
NUM_SHARDS=12 SHARD_OFFSET=8 ... bash scripts/run_4gpu_sharded.sh
```

After all nodes finish, concatenate exactly one copy of every shard. Do not concatenate each node's already-created `all.jsonl` into another `all.jsonl` without checking duplicates.

## Truncation audit

For each checkpoint:

```bash
python - <<'PY'
import json,glob,collections
for fn in glob.glob('results/raw_step*/all.jsonl'):
    rows=[json.loads(x) for x in open(fn) if x.strip()]
    c=collections.Counter(str(r.get('finish_reason')) for r in rows)
    print(fn, len(rows), c)
PY
```

If >5% of generations end because of length or if truncation differs visibly by checkpoint, rerun only affected requests with `MAX_TOKENS=16000` and replace those rows before scoring.

---

# Phase 3 — Score all checkpoint samples

The primary scorer mirrors the seed repository's documented logic:

1. official PRIME/MATH rule+sympy scorer;
2. Qwen2.5-32B-Instruct judge fallback for rule-negative cases with extractable answers.

Example:

```bash
python code/score_math_samples.py \
  --input results/raw_step256/all.jsonl \
  --output results/scored_step256.jsonl \
  --temporal-repo external/Temporal_Forgetting \
  --method hybrid \
  --judge-model Qwen/Qwen2.5-32B-Instruct \
  --judge-tp 4 \
  --checkpoint step_256 \
  --checkpoint-order 7
```

Checkpoint order mapping:

```text
step_32  -> 0
step_64  -> 1
step_96  -> 2
step_128 -> 3
step_160 -> 4
step_192 -> 5
step_224 -> 6
step_256 -> 7
```

If the 32B judge is temporarily unavailable, use `--method prime` for an early screen, but **before freezing F/N/S membership** re-score all primary candidate/control items with the hybrid procedure.

Concatenate:

```bash
cat results/scored_step32.jsonl \
    results/scored_step64.jsonl \
    results/scored_step96.jsonl \
    results/scored_step128.jsonl \
    results/scored_step160.jsonl \
    results/scored_step192.jsonl \
    results/scored_step224.jsonl \
    results/scored_step256.jsonl \
    > results/checkpoint_samples.jsonl

python code/validate_dataset.py --input results/checkpoint_samples.jsonl
```

Expected primary geometry:

```text
500 problems × 8 checkpoints × 16 samples = 64,000 rows
```

## Mandatory scorer audit

Before classification, manually inspect a stratified random sample:

```text
25 rule-positive
25 judge-rescued
25 final-negative
```

If >5% are wrong, repair scorer behavior and rescore all checkpoint data. Do not patch labels by hand for only the F/N/S candidates.

---

# Phase 4 — Freeze robust temporal states

```bash
python code/build_forgotten_set.py \
  --input results/checkpoint_samples.jsonl \
  --output results/groups_raw.jsonl \
  --min-samples 16 \
  --correct-threshold 0.75 \
  --wrong-threshold 0.125

python code/analyze_state_dynamics.py \
  --groups results/groups_raw.jsonl \
  --output-json results/state_dynamics.json
```

Record counts of:

```text
forgotten (F)
never_correct (N)
stable_correct (S)
late_acquired_or_recovered
other
```

### Hard feasibility gate

Proceed on MATH-500 only if:

```text
F >= 50
N >= 50
S >= 50
```

If F < 50, **do not change thresholds**. Optionally repeat the same procedure on OlympiadBench. If robust forgetting remains sparse, stop Topic 05.

Still inspect `state_dynamics.json`. If another natural pattern dominates—e.g. repeated robust `C→W→C` oscillation—record it as a candidate *new* topic, not as a rescue of Topic 05.

---

# Phase 5 — Freeze traces and F/N controls

```bash
python code/select_traces.py \
  --samples results/checkpoint_samples.jsonl \
  --groups results/groups_raw.jsonl \
  --output results/groups_with_traces.jsonl

python code/match_controls.py \
  --groups results/groups_with_traces.jsonl \
  --output results/fn_pairs.jsonl
```

Inspect the pair count and distribution over subject/level. Do not rematch after seeing re-entry results.

The deterministic trace rules are:

- old-self = shortest correct trace from the **latest robust-correct old checkpoint**;
- final-wrong = shortest valid wrong trace from final checkpoint;
- other/never-correct correct route = canonical MATH-500 worked solution.

---

# Phase 6 — Build and audit re-entry requests

```bash
python code/build_reentry_prompts.py \
  --groups results/groups_with_traces.jsonl \
  --pairs results/fn_pairs.jsonl \
  --output results/reentry_requests.jsonl \
  --tokenizer UWNSL/Qwen2.5-7B-deepscaler_4k_step_256
```

The script:

- emits one baseline/problem;
- appends partial reasoning as **assistant prefix**, not user hint;
- uses old-self 10/25/50% step fractions;
- matches control prefixes to old-self token budget;
- assigns frozen 60/40 discovery/confirmation splits.

## Mandatory prefix audit

Before final-model inference, manually inspect at least 100 requests stratified across:

```text
oldself / other_correct / final_wrong / verified_correct
10% / 25% / 50%
```

Audit:

- direct final-answer leakage;
- malformed step cuts;
- token-budget mismatch;
- whether assistant-prefix continuation looks syntactically natural.

If >10% of cuts are malformed, fix the deterministic splitter **before any G0 inference**, rebuild all requests, and document the change. Do not hand-edit individual items.

---

# Phase 7 — G0-A final-checkpoint re-entry

Primary run:

```bash
MODEL=UWNSL/Qwen2.5-7B-deepscaler_4k_step_256 \
INPUT=results/reentry_requests.jsonl \
OUTDIR=results/reentry_raw_n8 \
N=8 \
NUM_SHARDS=4 \
bash scripts/run_4gpu_sharded.sh
```

Score:

```bash
python code/score_math_samples.py \
  --input results/reentry_raw_n8/all.jsonl \
  --output results/reentry_scored_n8.jsonl \
  --temporal-repo external/Temporal_Forgetting \
  --method hybrid
```

Analyze:

```bash
python code/analyze_reentry.py \
  --input results/reentry_scored_n8.jsonl \
  --output-json results/reentry_analysis_n8.json
```

Read **confirmation** separately from discovery. Do not report a discovery-only pattern as the conclusion.

Primary contrasts are already locked in `VALIDATION.md`.

---

# Phase 8 — G0-B old-route likelihood

Prepare a simple trace JSONL containing F/S old-self traces and N verified-correct traces, keeping original user prompt and group/source fields. Then shard across GPUs/nodes:

```bash
CUDA_VISIBLE_DEVICES=0 python code/trace_likelihood.py \
  --model UWNSL/Qwen2.5-7B-deepscaler_4k_step_256 \
  --input results/likelihood_traces.jsonl \
  --output results/nll_shard0.jsonl \
  --num-shards 4 --shard-index 0 &
# repeat shard-index 1,2,3 on GPUs 1,2,3
wait
cat results/nll_shard*.jsonl > results/trace_nll.jsonl

python code/analyze_trace_likelihood.py \
  --input results/trace_nll.jsonl \
  --pairs results/fn_pairs.jsonl \
  --output-json results/trace_nll_analysis.json
```

Use 0/10/25/50% fractions. Do not search for a favorable prefix point.

---

# Phase 9 — High-power robustness run

If G0 shows an interpretable pattern, rerun **the same frozen requests** with 16 samples/request:

```bash
MODEL=UWNSL/Qwen2.5-7B-deepscaler_4k_step_256 \
INPUT=results/reentry_requests.jsonl \
OUTDIR=results/reentry_raw_n16 \
N=16 \
NUM_SHARDS=4 \
bash scripts/run_4gpu_sharded.sh
```

No new controls, prefix fractions, thresholds, or old-checkpoint selection rules are allowed.

---

# Phase 10 — Optional G1 relearning savings

Only after G0 is complete.

Compare F old solutions against matched N correct solutions from the exact same final checkpoint, with equal data/exposure budget. Prefer a cheap LoRA screen at exposures:

```text
0, 1, 2, 4, 8 per item
```

Match initial final-model solution NLL as closely as practical. If F reacquires markedly faster than N, this is consistent with a residual learning trace. It does not by itself prove a specific storage mechanism.

---

# Final result decision

Write `RESULTS.md` and choose exactly one headline status:

```text
PROCEED
KILL
NEW PHENOMENON — REGISTER SEPARATELY
```

## PROCEED

Needs an interpretable re-entry/likelihood pattern that survives the locked confirmation split and is not reducible to generic correct-hint benefit.

## KILL

Examples:

- robust forgetting too sparse under repeated sampling;
- old-self has no special advantage and old-route likelihood is not distinctive;
- positive controls fail, making the measurement uninterpretable;
- exact adjacent work is found that already answers this distinction.

## NEW PHENOMENON — REGISTER SEPARATELY

Use only when the predeclared state audit exposes a different striking dynamic, e.g. repeated competence oscillations. It must receive a new literature collision check and validation protocol.

---

# Results checklist

`RESULTS.md` must contain:

1. exact model IDs/revisions and software environment;
2. sampling parameters and row counts;
3. truncation statistics;
4. scorer audit counts/disagreements;
5. F/N/S counts and robust-state sequence histogram;
6. number of frozen F/N pairs;
7. trace and prefix audit statistics;
8. raw rates for every re-entry condition/fraction;
9. all predeclared bootstrap contrasts, separately for discovery and confirmation;
10. old-route NLL curves and matched F-N differences;
11. high-power n=16 replication if run;
12. final decision and precise reason;
13. any alternate natural phenomenon, clearly labeled exploratory and separate from Topic 05.
