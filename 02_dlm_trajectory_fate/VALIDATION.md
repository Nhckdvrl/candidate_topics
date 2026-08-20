# G0 validation implementation

This is the highest-priority pilot. It combines two established measurements without inventing a new DLM sampler:

1. **surface temporal oscillation** from *Time Is a Feature*;
2. **hidden-state correctness probing** from *Probing Functional Correctness in Diffusion Language Models*.

The one changed variable is the label: **eventual final correctness -> future fate of the current state**.

## What was verified against the public implementations

### Time Is a Feature / dLLM-MidTruth

Paper: <https://arxiv.org/abs/2508.09138>

Code: <https://github.com/aim-uofa/dLLM-MidTruth>

The crucial implementation detail is in `eval/generate.py`: at each denoising iteration the model first predicts a complete `x0` for all currently masked positions, then only transfers a subset of high-confidence tokens into persistent state `x`. Temporal voting parses the **full x0 prediction**, not the partially unmasked `x`.

Therefore this implementation defines current surface correctness from `x0` **before transfer**. Decoding `x` would measure a different process and can miss the temporal oscillation phenomenon.

The public MidTruth GSM8K evaluation example uses a 128-token generation, 64 denoising steps and temperature 0. The repository also uses 128 diffusion steps during temporal-consistency RL. A separate robustness launcher is included for the 64-step evaluation geometry.

### Probing Functional Correctness in DLMs

Paper: <https://aclanthology.org/2026.acl-srw.15/>

Code: <https://github.com/guan404ming/dlm-probing>

Public implementation details reproduced here:

- model: `GSAI-ML/LLaDA-8B-Instruct`;
- mask token ID: `126336`;
- GSM8K test: 1,319 examples;
- denoising: 128 steps;
- GSM8K generation length: 512;
- LLaDA block length: 32;
- LLaDA sampling temperature: 0.2;
- public seed-0 generation;
- original hidden checkpoints: `{0,1,4,16,32,64,127}`;
- hidden states mean-pooled over generation positions;
- probe implementation: `StandardScaler -> PCA(64) -> LogisticRegression(C=1, lbfgs, max_iter=1000)` with 5-fold stratified CV.

The public source counts the entire `hidden_states` tuple when reporting layers. G0 therefore uses tuple indices `24, 25, 28` directly; these are in the upper region where their GSM8K signal is strong.

## What G0 changes, and only what it needs to change

The generation loop is kept close to the public probing implementation. G0 adds:

1. decode `x0` at **all 128 steps** and record current correctness;
2. save hidden states only at 16 preregistered steps: `0,1,2,4,8,16,24,32,48,64,80,96,112,120,124,127`;
3. construct conditional trajectory-fate labels from the complete 128-step correctness trace;
4. compare hidden probes against entropy / selected-token probability / fraction-unmasked baselines;
5. group by problem ID when pooling multiple denoising steps for lead-time analysis.

This avoids a label artifact: if correctness were inspected only at the 16 hidden checkpoints, a short-lived `wrong->correct->wrong` event between checkpoints would be silently missed.

## Fate labels

For a saved step `t`:

### Current state is wrong

```text
recoverable = 1  iff any later x0 prediction is correct
recovery_lead = first later correct step - t
```

Otherwise it is `doomed` for the remainder of this sampled trajectory.

### Current state is correct

```text
will_overwrite = 1 iff any later x0 prediction is incorrect
overwrite_lead = first later incorrect step - t
```

Otherwise it is `stable-correct` for the remainder of the sampled trajectory.

These are trajectory labels, not claims about an intrinsic deterministic property of the prompt.

## Intermediate-answer parsing

Early DLM predictions can contain stray numbers before a genuine answer field exists. To avoid counting an accidental matching number as a "correct state", the **primary** trajectory label is strict:

- probing prompt: requires `#### <number>`;
- MidTruth prompt: requires a numeric `\\boxed{...}`.

For robustness, the extractor also stores `correct_all_fallback`, which reproduces the public probing code's fallback to the final number in the text. `train_probes.py --label-mode fallback` reruns the analysis with that looser parser.

## Files

```text
src/gsm8k_utils.py          # strict/fallback/boxed answer parsers
src/fate_labels.py          # recover/overwrite labels + lead time
src/generate_fates.py       # LLaDA denoising + full x0 trajectory + sparse hidden capture
src/train_probes.py         # conditional hidden probes and surface baselines
run_pilot_4gpu.sh           # primary probing-geometry G0
run_midtruth_geometry_4gpu.sh # robustness run closer to MidTruth GSM8K eval
tests/test_fate_labels.py
```

## Primary 4-GPU G0

```bash
cd 02_dlm_trajectory_fate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Engineering smoke run first
NUM_EXAMPLES=40 ./run_pilot_4gpu.sh

# Preregistered G0
NUM_EXAMPLES=1000 ./run_pilot_4gpu.sh
```

The launcher creates four independent LLaDA processes, one per GPU, so there is no cross-node or cross-GPU synchronization requirement. Each GPU handles approximately 250 GSM8K problems.

Primary G0 geometry:

```text
prompt style    public dlm-probing GSM8K prompt
steps           128
generation      512 tokens
block length    32
temperature     0.2
hidden indices  24,25,28
hidden regions  1 (cheap G0; --n-regions 4 reproduces region pooling more closely)
```

Outputs:

```text
artifacts/raw/shard_*.npz
artifacts/raw/shard_*_final_texts.jsonl
artifacts/probes/class_counts.csv
artifacts/probes/step_layer_auc.csv
artifacts/probes/lead_time_auc.csv
artifacts/probes/fate_labels.npz
```

## MidTruth-geometry robustness run

```bash
NUM_EXAMPLES=1000 ./run_midtruth_geometry_4gpu.sh
```

This uses:

```text
prompt style    MidTruth boxed-answer format + <reasoning> prefix
steps           64
generation      128 tokens
block length    32
temperature     0
```

A strong result should not depend on one accidental decoding configuration.

## Statistical safeguards

- Fixed-step probes contain at most one row per problem, so ordinary stratified CV cannot leak neighboring steps from the same problem.
- Lead-time analysis pools states across steps and therefore uses `StratifiedGroupKFold(groups=problem_id)`.
- A step is skipped when either class has fewer than 20 examples by default.
- Surface baselines are trained with the same CV policy.
- The primary claim requires hidden-state AUC to exceed surface baselines **before** the visible transition, not merely at the transition step.

## Collision check (August 20, 2026)

`Diffusion Language Models Know the Answer Before Decoding` / Prophet already establishes **early surface answer convergence** and uses a top-2 confidence gap for early commit. That makes “the answer can be known early” non-novel. It still does not ask the conditional hidden-state question here: among states with the same current surface correctness, can hidden activations predict **future recovery or future overwrite**?

The ACL 2026 probing paper predicts eventual final correctness from hidden states, but does not condition on current `x0` correctness or label recover/overwrite transitions. The exact gap therefore remains narrower than either seed paper.

## G0 decision rule

Continue if at least one conditional task shows a stable, pre-transition signal:

```text
AUC_hidden > AUC_surface_baseline
```

with non-trivial positive lead time and enough class support.

Especially valuable outcomes:

- wrong states are separable into recoverable vs doomed long before recovery;
- correct states are separable into stable vs future-overwritten before corruption;
- recovery and overwrite show a robust asymmetry.

Stop if:

- class counts are too small;
- signal appears only at/after the transition;
- entropy/confidence explains the effect;
- strict-parser results disappear and only loose last-number parsing gives a signal.

## Local verification performed before committing

- Python syntax/bytecode compilation: passed.
- strict/fallback/boxed GSM8K parser tests: passed.
- recoverability / overwrite / lead-time unit tests on hand-constructed trajectories: passed.
- synthetic shard -> conditional-probe CSV pipeline: passed.

The full LLaDA generation was not executed in the ChatGPT sandbox because the 8B weights and CUDA GPU are not available there.
