# Server handoff — validation v2

Work from `topic-09-vla-own-limits` until PR #24 lands, then from `main`.

## Question

Do VLA success/reliability representations track **this policy's own competence**, or mostly generic state difficulty?

The entire first validation is one same-state paired contrast. Do not add mechanism work before it passes.

## 1. Environment / checkpoint preflight

Use the official `Physical-Intelligence/openpi` LIBERO stack pinned in `LOCKED_CONFIG.json`. Obtain the released pi0.5 LIBERO 2k / 3k / 9k checkpoints; convert them with OpenPI's official JAX->PyTorch script if the release is not already PyTorch.

Run the Topic-09 controlled server from inside the OpenPI environment:

```bash
python -m src.openpi_instrumented_server \
  --config pi05_libero \
  --checkpoint-dir <checkpoint> \
  --port <port> --device cuda:<gpu>
```

Then run:

```bash
python -m src.preflight --port <port> --out results/preflight_<ckpt>.json
```

Do not collect science data until preflight passes. Also verify checkpoint success is in a sensible range relative to the released evaluation rather than silently accepting a broken normalization/action stack.

## 2. G0 discovery behavior panel

Use every LIBERO-10 task, init indices `0-14`, and eight behavior seeds `110000-110007` for each of 2k / 3k / 9k.

One checkpoint shard looks like:

```bash
python -m src.collect_behavior \
  --port <port> --checkpoint 2k \
  --suite libero_10 --task-ids 0-9 --init-indices 0-14 \
  --policy-seeds 110000-110007 \
  --out results/g0_disc_2k.csv
```

Split task IDs / states across idle GPUs if useful; merged CSVs are accepted. **Never give different policy-seed sets to different checkpoints.**

Analyze only after all three panels are complete:

```bash
python -m src.analyze_disagreement \
  --csv results/g0_disc_*.csv \
  --min-trials 8 --rate-gap 0.5 --min-bidirectional 15 \
  --out results/g0_discovery.json
```

If the verdict is `STOP_NO_NATURAL_CROSSOVER`, stop the topic. Do not manufacture hard states.

If it passes, freeze the selected pair.

## 3. G1 discovery features for the frozen pair

For each checkpoint in the frozen pair, extract only the predeclared layer-11 representation on discovery states with feature seeds `310000-310003`:

```bash
python -m src.collect_features \
  --port <port> --checkpoint <A> \
  --suite libero_10 --task-ids 0-9 --init-indices 0-14 \
  --feature-seeds 310000-310003 \
  --out results/features_disc_A.npz
```

The server hook is observational: full action-expert layer-11 output -> mean action tokens -> mean 10 denoise steps. Four common-noise feature repeats are then averaged by analysis.

## 4. Independent confirmation

Use init indices `15-29`, behavior seeds `210000-210007`, and feature seeds `410000-410003`. Only the already frozen pair is run.

Do not reuse discovery states or noise-seed families.

Run G1 once:

```bash
python -m src.run_g1 \
  --train-behavior results/g0_disc_A.csv results/g0_disc_B.csv \
  --test-behavior results/g0_conf_A.csv results/g0_conf_B.csv \
  --train-features results/features_disc_A.npz results/features_disc_B.npz \
  --test-features results/features_conf_A.npz results/features_conf_B.npz \
  --checkpoint-a <A> --checkpoint-b <B> \
  --out results/g1_confirmation.json
```

The script first checks that bidirectional robust crossover replicates, then fits one shared ridge readout on discovery success rates and applies the frozen AUROC gate on confirmation states.

## Resources

Idle GPUs may be used on:

```text
fvcrc10 fvcrc11 fvcrc12 fvcrc13 fvcrc15 fvcrc20 fvcrc21
```

Use independent checkpoint/task shards, not cross-node distributed training. There is no large-model training here. Prefer existing local environments; create an isolated one if dependencies conflict.

## Stop rule

After a clean negative result, do not add perturbations, nonlinear probes, separate checkpoint probes, representation alignment, layer sweeps, SAE, or new confidence metrics. The point of this candidate is that the core contrast should be simple.
