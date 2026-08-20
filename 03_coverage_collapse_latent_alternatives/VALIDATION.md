# G0 validation implementation

This topic is deliberately gated more aggressively than the other two. Before training a checkpoint trajectory, G0 asks whether the proposed variable is measurable at all:

> At the exact first fork, can a low-capacity candidate-conditioned probe identify which concrete branch is actually viable?

If not, stop. Do not fall back to a generic representation-rank/CKA paper.

## What was verified against reasoning_forks

Paper: <https://arxiv.org/abs/2605.17026>

Code: <https://github.com/NNHieu/reasoning_forks>

### Exact graph structure

The official `src/data_generation/gen_arithchain.py` constructs two 10-hop arithmetic chains sharing one premise node. The target lies on one chain; the other is a distractor. For every generated problem, internal graph nodes are randomly remapped to ordinary single lowercase letters.

Consequences for our validation:

- the first decision point has exactly two branch candidates;
- there is exactly one viable branch leading to the target;
- branch viability can be reconstructed **deterministically from the equations**, not labeled by an LLM;
- because node letters are randomly remapped per problem, a probe cannot succeed by memorizing one fixed branch token.

The official generator creates:

```text
6400 SFT train
1600 RLVR train
1000 test
```

### Exact SFT checkpoint spacing

The public `run_sft.sh` uses, for Qwen2.5-0.5B on ArithChain:

```text
model         unsloth/Qwen2.5-0.5B
batch size    32
grad accum    1
learning rate 1e-5
data size      6400
save_steps     data_size / batch_size = 200
```

Inspection of `src/training/sft.py` additionally confirms **full-parameter fine-tuning** (`full_finetuning=True`, no 4-bit loading), BF16 when supported, max sequence length 2048, cosine LR scheduling, and response-only loss masking via `train_on_responses_only`. These details matter if the checkpoint trajectory is regenerated rather than downloaded.

Thus one 16-epoch run naturally produces one saved checkpoint per epoch:

```text
epoch 1  -> checkpoint-200
epoch 2  -> checkpoint-400
epoch 4  -> checkpoint-800
...
epoch 16 -> checkpoint-3200
```

`run_sft_dynamics_example.sh` uses epochs **1,2,4,8,16**, exactly matching the checkpoint set in the upstream `prepare_sampling_synthetic.sh` behavior evaluation.

### Exact response template / decision point

The public Jinja template ends in:

```text
### Response:
```

The generator's forward answer then begins with one of three opening sentences followed by a numbered first variable computation. G0 uses the canonical opening:

```text
To find the target value, we compute the following variables step by step:
1.
```

and extracts the hidden state at that exact position, immediately before the first branch variable should be named.

This is a cleaner operationalization than probing a generic final prompt state.

## Relation to Road Not Taken

Paper: <https://arxiv.org/abs/2511.04527>

That work establishes that hidden activations can contain information about unchosen future outcomes and uses activation interventions to make alternatives more accessible. Our G0 does **not** reproduce their expensive alternate-continuation tree. The controlled graph gives us the alternative's ground-truth viability for free.

If G0 and the training-dynamics stage succeed, an activation intervention is a later causal extension—not part of the initial feasibility test.

## Why the claim is narrower than generic "representation collapse"

By 2026 there are already papers on output-diversity collapse, sequential-post-training representation collapse, and suppression of exploratory reasoning primitives. Therefore this project must track a specific fact:

```text
Does h at this fork still encode which named unchosen branch can reach the target?
```

Effective rank, anisotropy or CKA alone cannot answer that question.

## Collision check (August 20, 2026)

Two newer neighboring results make the scope even narrower.

### When Are Teacher Tokens Reliable? (May 2026)

This paper explicitly introduces a **branch-viability diagnostic**: it records alternative next tokens at a reasoning prefix, forcibly follows each alternative, and tests whether the continuation still reaches the correct answer. Therefore **"measure whether an alternative branch is viable" is not itself a novel contribution**.

What remains different here is the conjunction of three constraints:

1. viability is exact graph ground truth rather than estimated by sampled continuation success;
2. the measurement is whether the **hidden state encodes the viability of a concrete candidate branch** before choosing it;
3. that branch-specific representation is tracked across the same SFT trajectory in which behavioral coverage collapses.

If we cannot maintain all three, this topic should be considered collided.

Paper: <https://arxiv.org/abs/2605.21606>

### Beyond the Best Guess (August 2026)

This paper provides fresh behavior-level evidence that RL can narrow the output distribution and reduce `pass@k`, and compares RL with Evolution Strategies as an alternative post-training method. It strengthens the importance of coverage as a phenomenon, but it does not measure hidden branch-specific information.

Paper: <https://arxiv.org/abs/2608.12679>

So the current claim is **not** "post-training loses coverage" and **not** "some alternatives remain viable". It is only:

```text
When a known viable branch disappears from sampled behavior during post-training,
does the model still encode that branch's viability at the decision state?
```

## Files

```text
src/graph_parser.py              # parse equations, target, premise, exact viable first branch
src/prompt_utils.py              # reproduce official Alpaca wrapper + canonical decision prefix
src/prepare_forks.py             # official test.parquet -> 1000 exact fork labels
src/extract_branch_states.py     # hidden state, candidate embeddings, candidate output log-prob margin
src/train_pairwise_probe.py      # candidate-conditioned latent probe vs output-accessibility baseline
prepare_upstream.sh              # clone/generate official reasoning_forks data
run_g0.sh                        # base-model feasibility test
run_sft_dynamics_example.sh      # epoch 1/2/4/8/16 hidden-state extraction
run_behavior_passk_forward.sh    # forward-only upstream pass@k reproduction
tests/test_graph_parser.py
```

## G0 measurement

For candidates `A` and `B`, sort their random letter names alphabetically and define

```text
y = 1 if candidate A is the viable branch, else 0
```

At the decision point, save hidden state `h_l` at every transformer block and candidate embedding difference

```text
de = e_A - e_B
```

The deliberately low-capacity candidate-conditioned feature is

```text
z_l = h_l elementwise_mul de
```

A linear probe on `z_l` is a diagonal bilinear compatibility probe. It tests whether the state contains branch information that aligns with the **concrete candidate identity**, rather than only measuring generic hidden geometry.

### Output-accessibility baseline

The extractor also computes the teacher-forced continuation log-probability of `" A"` and `" B"` immediately after the decision prefix, including the rare case where a letter continuation tokenizes into multiple tokens:

```text
margin = log p(" A" | state) - log p(" B" | state)
```

This baseline is essential. During training dynamics we want to distinguish:

- latent branch information remains readable;
- ordinary output access to that branch becomes increasingly one-sided.

A probe that merely mirrors next-token margin is much less interesting.

## Prepare the exact official dataset

```bash
cd 03_coverage_collapse_latent_alternatives
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./prepare_upstream.sh
```

This clones `NNHieu/reasoning_forks`, runs their own `gen_arithchain.py`, and parses the generated 1,000-test parquet. No alternate benchmark is introduced.

## Base-model G0

```bash
./run_g0.sh
```

Outputs:

```text
artifacts/forks.jsonl
artifacts/states/base.npz
artifacts/branch_probe_metrics.csv
```

Primary layer is preregistered at 50% model depth; all layers are still reported diagnostically. Evaluation uses 5-fold stratified CV and `StandardScaler -> PCA(64) -> LogisticRegression`.

### Kill criterion

Do not enter training dynamics unless the base model provides a stable branch-specific signal. A useful practical threshold is not a magic single number, but G0 should at minimum satisfy all of:

- both labels have substantial support (random remapping should make this near balanced);
- primary-layer AUROC is clearly above chance under resampling;
- the result is not confined to one pathological layer;
- parser assertions hold for essentially all 1,000 official test graphs.

If AUROC is ~0.5 or highly probe-capacity-sensitive, stop.

## SFT dynamics

First produce the official run from inside the cloned upstream repository:

```bash
cd external/reasoning_forks
bash run_sft.sh arithchain_2_10_forward qwen2.5_0.5b 16
cd ../..
```

Then:

```bash
./run_sft_dynamics_example.sh
```

This extracts `base, e01, e02, e04, e08, e16` (the `base` file is produced by `run_g0.sh`) and rewrites `artifacts/branch_probe_metrics.csv` with all aligned checkpoints.

For the behavioral `pass@k` side, run:

```bash
GPUS=0,1,2,3 NUM_SAMPLES=64 ./run_behavior_passk_forward.sh
```

This wrapper preserves the upstream evaluation choices—epochs 1/2/4/8/16, temperature 1.0, top-p 0.95, max 512 generated tokens, 64 samples/problem, upstream prompt builder, upstream `VLLMSampler`, and upstream `evaluate_pass_k.py`—but schedules only the **forward** checkpoints. This is deliberate: the upstream `prepare_sampling_synthetic.sh` also creates reverse-model jobs, and the upstream `spawn_sampling.sh` hard-codes GPU IDs `4 5 6 7`; neither is necessary for our G0.

The representation plot of interest is then not generic hidden similarity. It is the trajectory of **branch-viability AUROC** next to output candidate margin / first-branch behavior / pass@k at the same five checkpoints.

## Result interpretation

Strongest suppression result:

```text
pass@k / branch access falls
but branch-specific viability AUROC remains high
```

Especially strong if a later activation intervention can restore the suppressed viable branch.

Erasure result:

```text
branch-specific viability AUROC falls in step with coverage
```

Also potentially publishable if robust and distinct from generic rank collapse.

Stop if only effective-rank/CKA-style collapse appears or if branch viability was never measurable at baseline.

## Local verification performed before committing

- Python syntax/bytecode compilation: passed.
- graph parser unit test on official-format two-chain question: passed.
- exact prompt-format unit test: passed.
- shell syntax checks for both checkpoint and forward-pass@k launchers: passed.
- synthetic pairwise-probe end-to-end pipeline: passed; the injected signal is recovered in the corresponding layer while the preregistered primary layer remains independent.

The actual Qwen checkpoint forward passes and upstream SFT sampling were not run in the ChatGPT sandbox because external model weights/GPU execution are unavailable there.
