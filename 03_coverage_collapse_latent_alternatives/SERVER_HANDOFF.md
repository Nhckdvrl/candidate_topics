# Server Agent Handoff — Topic 03

You are taking over **server-side execution and runtime debugging** for `candidate_topics/03_coverage_collapse_latent_alternatives`.

Your job is not to redesign the hypothesis while running it. The repository now contains a falsification-first experiment contract. Preserve its scientific invariants, fix only genuine runtime/engineering problems, and report the gate result exactly.

## Scientific question

We are testing whether SFT-induced reasoning coverage collapse is an **access/commitment failure** rather than complete loss of a useful alternative branch representation.

On the controlled `arithchain_2_10` binary Graph Branching task, SFT may make the model strongly commit to one first branch and reduce sampled `pass@k`. We ask whether a branch-viability readout learned at an earlier, higher-coverage checkpoint still transfers to the late checkpoint and can outperform the native first-fork readout on cases selected **without using correctness labels**.

The intended positive result is the conjunction:

```text
sampled coverage shrinks
AND first-fork commitment increases
AND an early branch-viability readout is genuinely target-sensitive
AND that exact frozen readout still transfers to the late checkpoint
AND target-blind control is near chance
AND on strongly committed native/probe disagreements selected without labels,
    the frozen latent readout wins reliably
```

Do not weaken this chain after seeing results.

## Two audit findings you must preserve

### 1. Use the paper-exact SFT learning rate

The seed paper appendix reports Qwen2.5-0.5B Graph Branching SFT with:

```text
learning rate = 2e-5
```

The pinned upstream `NNHieu/reasoning_forks` `run_sft.sh` currently hard-codes `1e-5` for Qwen2.5-0.5B. Therefore a legacy `1e-5` failure is **not** a valid scientific stop.

This repository adds `run_train_paper_exact.sh`, which directly invokes the pinned upstream `src/training/sft.py` with `2e-5`.

### 2. Do not resurrect the old `output-wrong rescue` statistic

In a binary fork with exactly one viable branch, selecting examples where the native branch is already known to be wrong makes the opposite branch deterministically correct. That selection leaks the answer structurally.

The corrected primary subset is label-free:

```text
abs(native A-B logprob margin) >= 2
AND native_choice != frozen_probe_choice
```

Only after that subset is fixed do we reveal graph ground truth and ask which readout won.

## Representation test: matched target flip

For each graph, the code reconstructs the terminal leaf of both first branches.

It then creates a matched counterfactual that changes **only the queried terminal** from one branch leaf to the other. All equations, premise, candidate letters and graph structure remain fixed. The correct first branch must flip.

This is essential. A branch-viability readout should move when the queried target changes; a static lexical/letter shortcut should not.

The latent readout is also deliberately frozen across checkpoints:

```text
reference checkpoint candidate-embedding basis
+ reference-discovery-selected layer
+ one reference-trained StandardScaler -> PCA<=32 -> LogisticRegression probe
```

No late-checkpoint probe may be retrained for the main gate.

## Environment and hardware policy

1. Prefer the server's existing/local Python or virtual environment first.
2. Install missing packages into that environment if compatible.
3. Create a new isolated environment only if there is a genuine dependency conflict that cannot be repaired cleanly.
4. Do not assume Slurm. The scripts are ordinary shell/Python launchers.
5. Do not do cross-node distributed training. This experiment does not need it.
6. Exploit node-local GPUs aggressively. A normal 4-GPU node is enough for the parallel sampling/state jobs.
7. If several independent nodes are available, you may use them for independent confirmations after G0, but do not introduce synchronization or weight-sharding across slow inter-node links.

## Start here

From the repository root:

```bash
cd 03_coverage_collapse_latent_alternatives

# Prepare the exact pinned upstream source + exact generated dataset.
bash ./prepare_upstream.sh

# Run the complete falsification-first pilot.
# If the paper-exact SFT trajectory is absent, this command trains it first.
GPUS=0,1,2,3 TRAIN_GPU=0 bash ./run_g0.sh
```

Pinned upstream commit:

```text
64bf9e3e86231bc6b52f2974ca285ad8aa8fc181
```

Default paper-exact training directory:

```text
external/reasoning_forks/runs/topic03_paper_exact/
qwen2.5_0.5b_sft_arithchain_2_10_forward_lr2e-5_bs32_ga1
```

Expected checkpoint mapping:

```text
e01 -> checkpoint-200
e02 -> checkpoint-400
e04 -> checkpoint-800
e08 -> checkpoint-1600
e16 -> checkpoint-3200
```

If training already exists and is complete, reuse it. Do not retrain merely for cleanliness.

## Before expensive execution: static sanity checks

Run:

```bash
python -m pytest -q tests
python -m py_compile src/*.py
for f in *.sh; do bash -n "$f"; done
```

The committed code was already statically checked before merge, but rerun this in the actual server environment to catch checkout/runtime differences.

## G0-A — behavior premise gate

Default workload:

```text
200 test problems
16 generations/problem
e01,e02,e04,e16
temperature=1.0
top_p=0.95
max_tokens=512
```

The reference checkpoint is chosen automatically among non-late checkpoints by best sampled `pass@8`.

Primary output:

```text
artifacts/behavior/<RUN_ID>/gate.json
```

Continue only if **all** are satisfied:

```text
CI95_low(pass@8_ref - pass@8_e16) > 0
pass@8_ref - pass@8_e16 >= 0.03

CI95_low(first_fork_entropy_ref - first_fork_entropy_e16) > 0
first_fork_entropy_ref - first_fork_entropy_e16 >= 0.05 nats

first-branch parse rate at ref >= 0.90
first-branch parse rate at e16 >= 0.90
```

If this gate returns `stop_or_redesign`, stop the scientific experiment there. Do not run hidden-state analysis to rescue a missing behavior premise.

The 200 behavior problems are then excluded from the latent experiment to avoid reusing checkpoint-selection data.

## G0-B — latent gate

Only if G0-A passes, the launcher extracts the remaining 800 problems under five conditions:

```text
reference/original
reference/target_flip
late/original
late/target_flip
reference/target_blind
```

Those 800 problems are split 60/40 discovery/confirmation.

The probe is trained only on reference discovery original+target-flip pairs. The selected layer, reference candidate-embedding basis and probe are frozen before late evaluation.

Primary outputs:

```text
artifacts/behavior/<RUN_ID>/latent_gate_metrics.csv
artifacts/behavior/<RUN_ID>/latent_gate_predictions.csv
artifacts/behavior/<RUN_ID>/latent_gate_metrics.json
```

The latent gate continues only if every criterion passes:

```text
reference paired hidden AUC CI95_low > 0.70
reference target-flip direction CI95_low > 0.75

late frozen-transfer paired hidden AUC CI95_low > 0.60
late target-flip direction CI95_low > 0.60

target-blind frozen-probe AUC satisfies |AUC - 0.50| < 0.10

label-free committed native/probe disagreement events >= 30
hidden-win-rate on those disagreements CI95_low > 0.55
```

The final G0 status is:

```text
continue_full_confirmation
```

or:

```text
stop_or_redesign
```

Do not modify thresholds after seeing the measurements.

## Runtime debugging authority

You may fix genuine engineering problems such as:

- CUDA / PyTorch / transformers / vLLM / Unsloth compatibility;
- missing Python packages;
- checkpoint paths;
- tokenizer loading;
- GPU visibility or memory-utilization settings;
- multiprocessing worker counts;
- device-map issues;
- shell portability;
- stale partial outputs from an interrupted run, provided you use a fresh `RUN_ID` when scientific sampling is rerun.

You may reduce batch-like runtime settings if required to fit memory, as long as the model, checkpoints, prompts, sampling parameters, dataset examples and decision rules remain unchanged.

You must **not** silently change:

- paper-exact `2e-5` SFT learning rate for the primary trajectory;
- the 200-problem G0-A scientific sample unless a documented runtime-only rerun is needed;
- the sampling temperature/top-p for the primary behavior gate;
- the target-flip construction;
- exclusion of behavior-gate problem IDs from latent analysis;
- discovery/confirmation separation;
- reference-only layer selection;
- frozen reference candidate basis;
- frozen reference-trained probe;
- label-free committed-disagreement definition;
- any stop/continue threshold.

If one of those scientific invariants genuinely must change because the implementation is invalid, stop and document why instead of quietly repairing the experiment into a different question.

## Important integrity checks during the run

Before trusting latent results, explicitly verify a few saved examples:

1. `original` and `target_flip` have identical graph equations and premise.
2. Only the query target changes.
3. `label_A_viable` flips exactly.
4. original/flip/reference/late files have identical ordered `problem_id` arrays.
5. the target-blind text removes query target identity without altering graph equations.
6. the late feature uses the **reference** candidate embedding difference, not late embeddings.
7. no late or target-blind probe fitting occurs.

If any of these fail, treat it as an implementation bug and repair before interpreting results.

## If G0 passes

Only then run the more expensive confirmation.

Behavior confirmation:

```bash
NUM_PROBLEMS=1000 \
NUM_SAMPLES=64 \
TAGS=e01,e02,e04,e08,e16 \
RUN_ID=full_$(date +%Y%m%d_%H%M%S) \
bash ./run_behavior_passk_forward.sh
```

Then run the hidden trajectory follow-up using the G0-locked logic:

```bash
bash ./run_sft_dynamics_example.sh
```

Do not scale to extra training seeds/backbones unless the core G0 result is clearly positive. If it is positive, the next paper-level confirmation is at least one additional training seed or one additional backbone from the seed-paper setup.

## What to return after execution

Give a compact, evidence-first report containing:

1. exact `candidate_topics` git commit used;
2. Python environment and any runtime fixes made;
3. paper-exact training directory and checkpoint availability;
4. G0-A table and `gate.json` status;
5. automatically selected reference checkpoint;
6. G0-B metrics, especially target-flip transfer, target-blind AUC, disagreement count and hidden-win CI;
7. final verdict, using one of:

```text
STOP_BEHAVIOR_PREMISE
STOP_OR_REDESIGN
CONTINUE_FULL_CONFIRMATION
```

8. paths to all result artifacts;
9. if you changed code to repair a genuine runtime bug, the exact patch/commit and why it does not alter the scientific contract.

Do not soften a failed gate and do not launch extra sweeps to search for a positive result after a preregistered stop condition fires.
