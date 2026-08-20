# Source-checked G0 implementation

## 1. What the seed code actually does

Paper: <https://arxiv.org/abs/2605.17026>

Code: <https://github.com/NNHieu/reasoning_forks>

### Graph structure

Inspection of `src/data_generation/gen_arithchain.py` shows that `arithchain_2_10` contains:

- one premise `p0`;
- chain `a1 -> ... -> a10`;
- chain `b1 -> ... -> b10`;
- fixed target node `a10` before random letter remapping;
- 20 random constants;
- per-problem random remapping of internal nodes to lowercase letters.

Thus the first decision point has two **locally valid** candidates but exactly one **globally target-reaching** candidate. This matches the paper's Graph Navigation description: the branch choice is locally ambiguous even though only one branch eventually succeeds.

The generator creates exactly:

```text
6400 SFT train
1600 RLVR train
1000 test
```

`src/graph_parser.py` reconstructs this dependency structure directly from each official test question and asserts exactly two first-fork children and exactly one globally viable child.

### SFT details

`run_sft.sh` uses Qwen2.5-0.5B with:

```text
batch size       32
grad accumulation 1
learning rate    1e-5
6400 examples
save_steps       200
16 epochs
```

Inspection of `src/training/sft.py` additionally confirms:

- `full_finetuning=True`;
- no 4-bit loading;
- BF16 when supported;
- max sequence length 2048;
- cosine LR scheduler;
- response-only loss masking through `train_on_responses_only`.

Therefore checkpoints `200/400/800/1600/3200` correspond to epochs `1/2/4/8/16`, exactly the checkpoints used by the upstream synthetic pass@k helper.

### Prompt / decision position

The upstream Alpaca template ends with `### Response:`. Forward answers then start with one of three opening sentences and a numbered first computation. G0 uses the canonical upstream opening

```text
To find the target value, we compute the following variables step by step:
1.
```

and extracts the final hidden state immediately after `1.`, before the branch variable is produced.

## 2. G0 measurement

For alphabetically ordered candidates `A` and `B`:

```text
y = 1 iff A is globally viable
```

For every transformer block save hidden state `h_l` and input-embedding difference

```text
de = e_A - e_B
```

The implemented latent feature is

```text
z_l = h_l elementwise_mul de
```

A linear classifier on `z_l` is a low-capacity diagonal bilinear compatibility probe. The primary layer is preregistered at 50% depth; all layers are retained as diagnostics.

Probe pipeline:

```text
StandardScaler -> PCA(up to 64) -> LogisticRegression(C=1, lbfgs)
5-fold stratified CV
```

The script saves per-problem out-of-fold probabilities, not only aggregate AUROC.

### Output-accessibility baseline

At the identical decision state, `src/extract_branch_states.py` also calculates teacher-forced continuation scores for the concrete branch letters, including multi-token continuations:

```text
margin = log p(" A" | state) - log p(" B" | state)
```

This distinguishes “latent viability remains readable” from “the next-token readout still prefers the viable branch.”

## 3. Run G0

```bash
cd 03_coverage_collapse_latent_alternatives
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./prepare_upstream.sh
./run_g0.sh
```

Outputs:

```text
artifacts/forks.jsonl
artifacts/states/base.npz
artifacts/branch_probe_metrics.csv
artifacts/branch_probe_metrics_oof.csv
```

### Kill criterion

Do not enter training dynamics unless:

- graph parsing succeeds on essentially all 1,000 official test problems;
- both labels have substantial support;
- primary-layer AUROC is clearly above chance under resampling;
- the signal is not a one-layer accident;
- the result is stable to low-capacity probe variations.

If base-model branch viability is not measurable, stop.

## 4. SFT dynamics

Generate the official forward SFT run in the cloned upstream repository, then:

```bash
./run_sft_dynamics_example.sh
```

This evaluates `e01/e02/e04/e08/e16`; `base` comes from G0. The primary comparison is:

```text
latent viability AUROC
vs
output candidate-margin AUROC
vs
behavioral pass@k / viable-branch access
```

at the same checkpoints.

## 5. Behavior-side reproduction

The upstream `prepare_sampling_synthetic.sh` creates both forward and reverse jobs; `spawn_sampling.sh` hard-codes GPU IDs `4 5 6 7`. Neither is necessary for this candidate topic.

`run_behavior_passk_forward.sh` is therefore a forward-only wrapper. It still uses the upstream:

- `src/inference/build_prompts.py`;
- exact `src/alpaca_template.jira`;
- `src/inference/run_sampling.py` / `VLLMSampler`;
- `src/math_eval/evaluate_pass_k.py`.

It preserves the official forward evaluation settings:

```text
epochs       1,2,4,8,16
temperature  1.0
top_p        0.95
top_k        -1
max_tokens   512
samples      64/problem
```

Run:

```bash
GPUS=0,1,2,3 NUM_SAMPLES=64 ./run_behavior_passk_forward.sh
```

After upstream pass@k evaluation, `src/analyze_sampled_branches.py` parses the first numbered variable in each sampled response and writes:

```text
artifacts/behavior/first_branch_samples.csv
artifacts/behavior/first_branch_per_problem.csv
artifacts/behavior/first_branch_summary.csv
```

Metrics include candidate parse rate, probability of selecting the globally viable first branch, and binary first-branch entropy.

## 6. Result interpretation

### Suppression

```text
pass@k / viable-branch accessibility decreases
latent global-viability AUROC remains high
```

This is the strongest result: post-training changes access before deleting structural information.

### Erasure

```text
coverage/accessibility and latent global-viability AUROC decline together
```

Potentially publishable only if the change is branch-specific and not reducible to generic rank/anisotropy collapse.

### Latent loss first

If global-viability decoding degrades before behavioral coverage, latent representation loss may be a precursor to coverage shrinkage.

### No stable relation

Stop.

## 7. Collision check — 2026-08-20

**When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning** (arXiv:2605.21606) already introduces a **branch-viability diagnostic**: record alternative next tokens, force each alternative, and test whether its continuation recovers the correct answer. Therefore “branch viability” alone is not novel.

This candidate survives only as the conjunction of:

1. exact graph-ground-truth global viability;
2. hidden-state encoding of the concrete viable branch before selection;
3. dynamics across a coverage-shrinking SFT trajectory.

**Beyond the Best Guess: Improving LLM Solution Coverage with Evolution Strategies** (arXiv:2608.12679, August 13 2026) provides new behavior-level evidence about post-training coverage but does not measure branch-specific hidden viability.

The seed paper itself already shows that prefix diversification can recover some coverage. Our claim therefore cannot merely be “the branch is recoverable.” It must establish what branch-specific information remains internally available while normal access changes.

## 8. Files

```text
src/graph_parser.py
src/prompt_utils.py
src/prepare_forks.py
src/extract_branch_states.py
src/train_pairwise_probe.py
src/analyze_sampled_branches.py
prepare_upstream.sh
run_g0.sh
run_sft_dynamics_example.sh
run_behavior_passk_forward.sh
tests/test_graph_parser.py
```

## 9. Local verification completed

Before committing:

- Python compilation passed;
- graph parser test passed;
- exact prompt-shape test passed;
- shell syntax checks passed;
- synthetic hidden-state -> OOF probe pipeline passed;
- synthetic sampled-generation -> branch-accessibility/entropy pipeline passed.

The actual Qwen checkpoint forwards, SFT training and vLLM sampling were **not** executed in the ChatGPT sandbox because GPU/model-weight access is unavailable there. The repository contains executable experiment code, but the scientific result still requires running it on a GPU machine.
