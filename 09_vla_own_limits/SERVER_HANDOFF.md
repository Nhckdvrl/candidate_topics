# Server handoff — validation v3

Work directly on `main`. Environment setup is in [`ENVIRONMENT.md`](./ENVIRONMENT.md);
the scientific contract and every kill line is in [`VALIDATION.md`](./VALIDATION.md).

## Question

Do VLA success/reliability representations track **this policy's own competence**, or
mostly generic state difficulty?

The entire first validation is one same-state paired contrast. Do not add mechanism work
before it passes.

## 0. Checkpoints

The released pi0.5 LIBERO checkpoints are JAX/orbax, ~12.4 GB each:

```bash
huggingface-cli download brandonyang/openpi-libero-2000 --local-dir $CKPT_ROOT/pi05_jax_2k
huggingface-cli download brandonyang/openpi-libero-3000 --local-dir $CKPT_ROOT/pi05_jax_3k
huggingface-cli download brandonyang/openpi-libero-9000 --local-dir $CKPT_ROOT/pi05_jax_9k
```

Feature capture needs PyTorch forward hooks, so convert each one:

```bash
$OPENPI/.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir $CKPT_ROOT/pi05_jax_2k \
  --config_name pi05_libero \
  --output_path $CKPT_ROOT/pi05_pt_2k
```

> **The directory name is load-bearing.** `convert_jax_model_to_pytorch.py` decides
> whether to emit pi0.5 adaptive-RMSNorm (`Dense_0`) parameters or plain pi0 RMSNorm
> (`scale`) parameters by testing `if "pi05" in checkpoint_dir` — a substring test on the
> *path you pass it*. A directory named `jax_2000` converts down the wrong branch. Every
> checkpoint path must contain `pi05`.

Behavior rollouts and feature extraction both go through the same PyTorch stack, so that
G0 and G1 never describe two different policies.

## 1. Preflight — never skip

```bash
CUDA_VISIBLE_DEVICES=0 $OPENPI/.venv/bin/python -m src.openpi_instrumented_server \
  --config pi05_libero --checkpoint-dir $CKPT_ROOT/pi05_pt_2k --port 8100 --device cuda:0

MUJOCO_GL=egl $CLIENT_PY -m src.preflight --port 8100 --out results/preflight_2k.json
```

Preflight must pass for **every** checkpoint before any science data is collected:
reproducible settled sim-state hash, bit-identical actions and features under a repeated
noise seed, different actions under a different noise seed, and exactly ten captured
denoising activations.

Then check that each checkpoint's LIBERO success is in a sane range. A broken
normalization or action stack shows up here, not in the analysis.

## 2. G0 behavior panel

```bash
PHASE=discovery bash run_g0_fleet.sh 2k 3k 9k
```

This starts one policy server plus one collector per (checkpoint, task shard) across the
node's idle GPUs and shards LIBERO-10 tasks between them. Nothing is distributed; each
stream is an independent process pair. `collect_behavior --resume` makes a restart cheap.

Discovery uses init indices `0-14` and behavior seeds `110000-110007`.
**Never give different policy-seed sets to different checkpoints.**

Analyze only after all three panels are complete:

```bash
$CLIENT_PY -m src.analyze_disagreement \
  --csv results/g0_discovery_*.csv \
  --min-trials 8 --rate-gap 0.5 --min-bidirectional 15 \
  --out results/g0_discovery.json
```

Three ways this stops the topic:

```text
STOP_NO_NATURAL_CROSSOVER                  not enough bidirectional support
STOP_CROSSOVER_EXPLAINED_BY_SAMPLING_NOISE  support does not beat the relabeling null
G0_PASS_FREEZE_PAIR                        freeze the pair, continue
```

The middle one matters. Eight rollouts and a 0.5 rate gap manufacture roughly six
spurious wins per direction in a 150-state panel even when both checkpoints are equally
competent everywhere, so the report always carries the null comparison.

Do not manufacture hard states, and do not lower the rate gap after seeing data.

## 3. G1 discovery features for the frozen pair

```bash
MUJOCO_GL=egl $CLIENT_PY -m src.collect_features \
  --port <port> --checkpoint <A> \
  --suite libero_10 --task-ids 0-9 --init-indices 0-14 \
  --feature-seeds 310000-310003 \
  --out results/features_disc_A.npz
```

One capture per `(state, feature_seed)` at the settled initial state, so this is minutes
rather than hours. The hook is observational: action-expert layer-11 output -> mean over
action tokens -> mean over the ten denoising steps.

## 4. Independent confirmation

```bash
PHASE=confirmation bash run_g0_fleet.sh <A> <B>
```

Init indices `15-29`, behavior seeds `210000-210007`, feature seeds `410000-410003`. Only
the frozen pair is run. Do not reuse discovery states or noise-seed families.

Run G1 exactly once:

```bash
$CLIENT_PY -m src.run_g1 \
  --train-behavior results/g0_discovery_A_*.csv results/g0_discovery_B_*.csv \
  --test-behavior results/g0_confirmation_A_*.csv results/g0_confirmation_B_*.csv \
  --train-features results/features_disc_A.npz results/features_disc_B.npz \
  --test-features results/features_conf_A.npz results/features_conf_B.npz \
  --checkpoint-a <A> --checkpoint-b <B> \
  --out results/g1_confirmation.json
```

It re-checks that bidirectional crossover replicates, fits one shared ridge readout on
discovery (penalty chosen by state-grouped CV inside discovery only), and applies the
frozen AUROC gate on confirmation states.

```text
PASS_POLICY_SPECIFIC_SUCCESS_SIGNAL     relative AUROC >= 0.70 and CI95 lower > 0.60
KILL_SELF_KNOWLEDGE_INTERPRETATION      readout tracks success, but not whose success
INCONCLUSIVE_NO_ABSOLUTE_SUCCESS_SIGNAL readout tracks nothing; the test had no power
KILL_CROSSOVER_NOT_REPLICATED           confirmation lost the crossover
```

## Resources

Idle GPUs on `fvcrc10 fvcrc11 fvcrc12 fvcrc13 fvcrc15 fvcrc20 fvcrc21`. Use independent
checkpoint / task / seed shards, never cross-node distributed training. `$HOME` is shared
storage, so both venvs are visible from every node.

## Stop rule

After a clean negative, do not add perturbations, nonlinear probes, separate per-checkpoint
probes, representation alignment, layer sweeps, SAE, or new confidence metrics. The point
of this candidate is that the core contrast should be simple.
