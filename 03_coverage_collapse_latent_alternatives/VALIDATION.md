# Topic 03 validation contract — audited 2026-08-21

## Scientific object

We reuse the official `arithchain_2_10` Graph Branching task from `NNHieu/reasoning_forks`. The generator creates two ten-hop chains from one premise and randomly remaps all node letters per problem. At the first reasoning step there are exactly two locally valid children, but exactly one is an ancestor of the queried target. Therefore branch viability is exact graph ground truth and requires no LLM judge.

The upstream implementation is pinned to audited commit:

```text
64bf9e3e86231bc6b52f2974ca285ad8aa8fc181
```

## Audit findings and fixes

### 1. The previous base-model kill criterion was invalid

The old G0 stopped if raw Qwen2.5-0.5B did not expose global branch viability. But target-reachability can itself be learned during SFT. The scientific question is what happens **after the capability exists and coverage later shrinks**.

**Fix:** choose an early reference checkpoint from the reproduced SFT behavior trajectory. Base is optional context, not a kill gate.

### 2. The old order spent compute before verifying the premise

The old workflow extracted 1,000 hidden states across five checkpoints and only then ran expensive `pass@k` sampling.

**Fix:** first reproduce the behavioral phenomenon on 200 problems x 16 samples at e01/e02/e04/e16. Hidden extraction is blocked unless that gate passes.

### 3. Mean viable probability is the wrong shrinkage variable

Coverage shrinkage can happen because per-problem decisions polarize: some problems become confidently correct and some confidently wrong. `pass@1` can therefore improve while `pass@k` falls.

**Fix:** the behavior gate requires both sampled coverage decline and first-fork entropy decline. It does not require the mean viable-branch probability to fall.

### 4. The previous latent-vs-output comparison was confounded

The candidate-conditioned feature is

```text
z_l = h_l * (embedding(A) - embedding(B))
```

Qwen2.5 ties input/output embeddings, so this feature shares geometry with the ordinary LM readout. `latent_auc - output_auc` is not evidence for an independent latent variable.

**Fix:** keep the low-capacity compatibility probe but change the decisive measurement. On held-out examples where the ordinary output margin chooses the wrong branch, ask whether the hidden probe still identifies the globally viable branch. This is the **hidden-rescue-on-output-wrong** metric.

### 5. Layer fishing and uncertainty were weak

The old code swept all layers but informally called 50% depth primary, and fold standard deviation was used as uncertainty.

**Fix:** stratified 60% discovery / 40% confirmation split. Select one layer only on the reference checkpoint discovery split, lock it, and report confirmation metrics with problem-level bootstrap CIs.

### 6. A shortcut-negative control was missing

**Fix:** remove only the queried target identity while keeping all graph equations. Without knowing which terminal node is queried, the two chains are symmetric. Target-blind viability AUROC >= 0.60 is a stop/redesign signal.

### 7. Sampling reruns could silently contaminate results

The upstream sampler writes timestamped CSVs. The old analyzer globbed all prior CSVs in a checkpoint directory.

**Fix:** each behavior run gets a fresh `RUN_ID`; the analyzer requires exactly one raw generation CSV per checkpoint and refuses contaminated directories.

### 8. Upstream was floating

**Fix:** `prepare_upstream.sh` pins the exact audited `reasoning_forks` commit and refuses to overwrite a dirty checkout.

### 9. Prompt-prefix robustness was hidden

The upstream generator randomly chooses among three opening sentences before the first numbered step. The old probe hard-coded one.

**Fix:** all three official variants are implemented. Variant 0 is the G0 primary; variants 1/2 are post-G0 robustness checks so we do not multiply compute before the core signal exists.

## G0-A — behavior premise gate

Default:

```text
problems     200
samples      16/problem
checkpoints  e01, e02, e04, e16
temperature  1.0
top_p        0.95
max_tokens   512
```

Run:

```bash
./prepare_upstream.sh
(cd external/reasoning_forks && ./run_sft.sh arithchain_2_10_forward qwen2.5_0.5b 16)  # once
GPUS=0,1,2,3 ./run_behavior_preflight.sh
```

The wrapper reuses the upstream prompt builder, Alpaca template, VLLM sampler and exact math evaluator. Reference checkpoint is chosen among non-late checkpoints by highest sampled `pass@8` (`pass_at_half`; tie: `pass@16`).

Primary paired problem-level bootstrap differences:

```text
pass@8(reference) - pass@8(e16)
first_branch_entropy(reference) - first_branch_entropy(e16)
```

Continue only if the lower 95% bootstrap bound of both differences is > 0 and the late first-branch parser coverage is >= 90%.

`run_state_preflight.sh` provides a 200-example teacher-forced debugging summary, but it is disabled by default and cannot establish coverage shrinkage.

## G0-B — branch-specific latent gate

Only after G0-A passes:

```bash
./run_latent_gate.sh
```

Extract all 1,000 official test problems for:

- behavior-selected early reference checkpoint;
- e16;
- target-blind reference control.

At each transformer block, store only the final prompt-position hidden vector and exact teacher-forced A/B candidate log-probability margin.

Probe:

```text
StandardScaler -> PCA<=32 -> LogisticRegression(C=1)
```

Layer selection is performed only on reference/discovery. The layer is then locked. Each checkpoint trains its own checkpoint-local probe on discovery and is evaluated on the same confirmation problem IDs.

Report:

- hidden viability AUROC + bootstrap CI;
- hidden viability accuracy + bootstrap CI;
- output candidate-margin AUROC / accuracy;
- count of output-wrong cases;
- hidden rescue accuracy among output-wrong cases;
- hidden rescue among strongly wrong cases (`true viable log-odds margin < -2`).

The central suppression signature is:

```text
normal late readout chooses the wrong branch
BUT
held-out hidden probe still identifies the viable branch
```

## G0-C — target-blind control

At the reference checkpoint, replace the target identity in the query with `the requested variable`, leaving all graph rules unchanged.

Expected: viability should become near chance. The automated G0 uses target-blind AUROC >= 0.60 as a shortcut warning/stop condition.

## Automated stop/continue thresholds

The latent gate returns `stop_or_redesign` if any of these holds:

- reference hidden AUROC 95% CI lower bound <= 0.55;
- fewer than 30 late output-wrong confirmation examples;
- late hidden-rescue 95% CI lower bound <= 0.50;
- target-blind hidden AUROC >= 0.60.

These are screening thresholds, not final publication criteria.

## Interpretation

- **coverage shrink + polarization + hidden rescue:** strongest suppression/access result; continue.
- **coverage shrink + polarization + latent signal collapses:** possible branch-specific erasure; lower priority and must be separated from generic representation collapse.
- **coverage shrink but no first-fork polarization:** stop this mechanism.
- **no sampled coverage shrinkage:** stop before probing.
- **no reference viability signal:** stop.
- **positive target-blind control:** stop/redesign.

## Full confirmation only after G0

```bash
# 1,000 problems x 64 samples x e01/e02/e04/e08/e16
./run_behavior_passk_forward.sh

# all five hidden-state checkpoints
./run_sft_dynamics_example.sh
```

Do not use full sweeps to rescue a failed G0.

## Collision boundary

As of 2026-08-21, the claim must remain narrow:

- Nguyen et al. already establish SFT-driven coverage shrinkage, decision-point commitment and prefix-based recovery.
- Zur et al. already establish hidden representation of alternative future outcomes.
- `When Are Teacher Tokens Reliable?` already uses forced-token continuation success as a behavior-level branch-viability diagnostic.
- generic post-training representation collapse is already active in 2026.

Therefore the surviving contribution is specifically **graph-ground-truth hidden viability on late wrong-commitment cases through a known coverage-shrinking SFT trajectory**, with shortcut controls. Generic CKA/effective-rank/anisotropy changes are out of scope.

## Verification before PR

Completed locally:

- `python -m py_compile` for all changed Python files;
- `bash -n` for all changed launchers;
- prompt variant / target-mask / pass@k unit tests;
- synthetic end-to-end behavior gate: known coverage shrink + polarization correctly yields `continue_to_latent`;
- synthetic end-to-end latent gate: injected viability remains recoverable on output-wrong cases while target-blind control stays at chance.

Actual Qwen training/sampling/forward passes require the GPU server and are deliberately guarded by the staged G0.
