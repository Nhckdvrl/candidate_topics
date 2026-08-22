# Topic 14 server runbook

## 1. Enter the project

```bash
cd candidate_topics/14_powerlaw_persistent_head
python -m pip install -r requirements.txt
```

Reuse an existing PyTorch environment if available; this project only needs NumPy, PyTorch and pytest.

## 2. Preflight before spending GPU time

```bash
python -m pytest -q
python audit_schedule.py --profile pilot --seeds 0
python audit_schedule.py --profile full --seeds 0,1,2,3,4
```

Do not train if algebra, batch identity, schedule multiset/order, mapping uniqueness or head-disjointness checks fail.

## 3. Engineering smoke

```bash
bash run_gate.sh smoke 0
```

Expected decision: `SMOKE_ONLY_DO_NOT_INTERPRET`.

## 4. Cheap pilot

```bash
bash run_gate.sh pilot 0
```

Expected decision wrapper: `PILOT_SIGNAL_ONLY_DO_NOT_CONCLUDE`. Use this to catch implementation/runtime problems and estimate whether the learning curves move; do not confirm or kill the topic from one seed.

## 5. Locked full G0

```bash
bash run_gate.sh full 0,1,2,3,4
```

This is the actual scientific validation. Read `outputs/full/decision.json` and the per-seed curves. The first requirement is a healthy clean-regime Static-vs-Uniform anchor; only then interpret Slow-vs-Fast.

If interrupted:

```bash
RESUME=1 bash run_gate.sh full 0,1,2,3,4
```

Resume restores model, optimizer and fp16 scaler state and rejects stale/mismatched protocol outputs.

## 6. If and only if the clean anchor is weak

Run the near-paper reproduction diagnostic:

```bash
bash run_gate.sh paper_anchor 0,1,2
```

This uses random initialization, no shared data warmup, 200k steps, 1000-step LR warmup and cosine decay. It is only for deciding whether the implementation/testbed can reproduce the seed phenomenon.

- paper anchor fails: debug implementation/task/training before scientific interpretation;
- paper anchor passes but clean anchor fails: the clean identification regime removed the seed effect; Topic 14 remains unresolved in this setup;
- clean anchor passes: paper anchor is unnecessary for the main Slow/Fast claim.

## 7. G1 only after a replicated G0 temporal effect

```bash
bash run_g1_persistence.sh full 0,1,2,3,4
```

G1 probes intermediate persistence lengths while preserving the same A/B batch multiset. It cannot rescue a G0 null.

## GPU / node usage

Each arm is an independent single-GPU run. On a four-GPU node, the four clean-core arms run concurrently. On fewer GPUs the launcher runs waves, one process per visible GPU. Existing `CUDA_VISIBLE_DEVICES` masks are respected.

There is no benefit to cross-node distributed training. If multiple nodes are available, the cleanest parallelism is one or more complete replication seeds per node; later collect all `seed*/` directories under the same `outputs/full/` root and run `analyze.py` once on the complete locked seed set.

## When fixing bugs

Preserve the scientific invariants: same Slow/Fast batch multiset, same branch point, same LR, fixed eval set and frozen primary metric. Engineering fixes are allowed and should be documented. If a fix changes the scientific intervention or training contract, invalidate older outputs and rerun in a fresh output directory.
