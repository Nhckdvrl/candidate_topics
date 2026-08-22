# 11 — What Does Diffusion Confidence Actually Know?

**Status:** VALIDATION CODE READY — run locked G-0 before any extension

## Scientific question

> Does native diffusion-LM confidence track the **internal consistency of a reasoning trajectory** independently of whether the final answer is externally correct?

This is deliberately narrower than “DLLM confidence is not calibrated.” The seed paper, [The Confidence Paradox](https://aclanthology.org/2026.findings-acl.2142/), reports that LLaDA-8B is badly calibrated on GSM8K yet strongly discriminative, and interprets the signal as structural consistency. It also shows a large confidence drop for forced arithmetic contradictions. What remains unidentified is whether **trajectory-internal consistency itself** can be separated from final-answer correctness.

The goal here is to answer that with one minimal factorial experiment, not with hidden states, probes, learned judges, threshold search, or many hand-built controls.

---

## Why the old 2×2 was still not clean enough

The earlier README proposed making a wrong intermediate arithmetic result and then propagating it. That is better than merely changing the final token, but the “consistent-but-wrong” trajectory still contains an arithmetic statement that is false relative to its own operands. A critic can therefore say that the supposedly coherent condition already contains a local contradiction.

The implemented design removes this problem entirely.

We separate:

1. **trajectory-internal consistency** — whether the state announced by the trajectory matches the state actually used by all downstream arithmetic;
2. **external correctness** — whether the state used by the trajectory matches the state specified by the problem prompt.

Every arithmetic equation in every cell is valid.

---

## Locked 2×2 design

Choose two nearby integer anchors `X != Y` and one deterministic three-step arithmetic program `f`.

For one orientation, the downstream trajectory always computes `f(X)` and is **text-identical after the first line** in all four cells.

| Cell | Prompt says | Trajectory announces | Downstream uses | Internal consistency | Final correctness |
|---|---:|---:|---:|---|---|
| `CC` | X | X | X | consistent | correct |
| `IC` | X | Y | X | inconsistent | correct |
| `CW` | Y | X | X | consistent | wrong |
| `IW` | Y | Y | X | inconsistent | wrong |

Example:

```text
Prompt: A calculator starts from 23. Add 5, multiply by 3, subtract 4.

CC:
Initial state: 23
Step 1: 23 + 5 = 28
Step 2: 28 * 3 = 84
Step 3: 84 - 4 = 80
Final answer: 80

IC:
Initial state: 29
Step 1: 23 + 5 = 28
Step 2: 28 * 3 = 84
Step 3: 84 - 4 = 80
Final answer: 80
```

`CC` vs `IC` changes only the announced initial state while prompt, downstream reasoning, and final answer stay fixed.

For the external-correctness manipulation, the complete output is held fixed and only the prompt anchor changes:

- `CC` and `CW` use the exact same output;
- `IC` and `IW` use the exact same output.

This yields a genuine factorial rather than four loosely matched error types.

### Important interaction that the design also identifies

A trivial prompt-copy heuristic predicts a different pattern:

- `CC` and `IW` have prompt/announcement agreement;
- `IC` and `CW` do not.

Therefore a simple “does the announced number match the prompt?” effect appears as the **factorial interaction**, not as the internal-consistency main effect. The design can distinguish:

```text
trajectory consistency:       CC ≈ CW > IC ≈ IW
final-answer correctness:     CC ≈ IC > CW ≈ IW
prompt/announcement matching: CC ≈ IW > IC ≈ CW
```

This is the main reason for using the full 2×2 instead of a single correct/wrong pair.

---

## Mirrored anchors: cancelling number-token preference

Each anchor pair is evaluated twice:

1. downstream branch uses `X`, with `Y` as the alternative;
2. downstream branch uses `Y`, with `X` as the alternative.

Effects are averaged at the anchor-pair level before bootstrap resampling.

This prevents a model-level preference for one particular number token from masquerading as a consistency effect.

The scorer additionally rejects any orientation where tokenization breaks the minimal intervention. By default both the prompt change and announcement change must be **exactly one token position** after tokenization.

---

## Confidence protocol

The implementation follows the seed paper’s primary LLaDA score:

1. concatenate the chat-formatted prompt and the prescribed trajectory;
2. run **one teacher-forced forward pass** on the fully specified sequence;
3. at every output position, read the softmax probability assigned to the token already occupying that position;
4. average probabilities over output tokens.

No diffusion generation is needed for this intervention experiment. This makes G-0 cheap: the expensive 128-step denoising loop is not part of the validation.

### Primary paper-compatible score

`confidence_full`

Mean same-position probability over every output token.

### Identification guardrail

`confidence_tail`

Mean probability over the **unchanged downstream continuation only**, excluding the manipulated announcement line.

This is not an alternative metric to rescue a weak result. It is a stricter guardrail:

> if full-sequence confidence changes but confidence on the text-identical downstream tokens does not, we do **not** claim trajectory-level structural consistency.

A local penalty on the changed announcement token is insufficient.

`confidence_announcement` is saved only as a diagnostic.

---

## Locked effects

For each orientation:

```text
consistency_when_correct = CC - IC
consistency_when_wrong   = CW - IW

Delta_consistency = 0.5 * [(CC - IC) + (CW - IW)]

correctness_when_consistent   = CC - CW
correctness_when_inconsistent = IC - IW

Delta_correctness = 0.5 * [(CC - CW) + (IC - IW)]
```

The strongest direct contrast is:

```text
CW - IC
```

because it asks whether a **coherent-but-wrong** trajectory receives higher confidence than an **incoherent-but-correct** trajectory.

Algebraically, `CW - IC = Delta_consistency - Delta_correctness`, so it is one contrast, not two pieces of evidence.

We also report the factorial interaction:

```text
(CC - IC) - (CW - IW)
```

which captures prompt/announcement agreement effects.

All uncertainty intervals are pair-level bootstrap intervals over mirrored anchor pairs; the two orientations from one pair are never treated as independent observations.

---

## G-0 decision rule

The verdict is deliberately locked before seeing model scores.

### `GO_STRONG_STRUCTURAL_SIGNAL`

Required:

1. the 95% bootstrap CI for `Delta_consistency` on **unchanged continuation tokens** is entirely above zero;
2. the paper-default full-output consistency effect is also stably positive;
3. the 95% CI for `CW - IC` on continuation tokens is entirely above zero.

Interpretation:

> internal trajectory coherence predicts native diffusion confidence strongly enough to beat final correctness in the decisive coherent-wrong vs incoherent-correct contrast.

### `MIXED_INTERNAL_SIGNAL_ONLY`

A stable continuation-token consistency effect exists, but `CW - IC` does not reliably exceed zero.

Interpretation: confidence contains a real internal-consistency signal, but the stronger “structural consistency over correctness” story is not established.

Do not tune prompts or invent error categories to rescue this state.

### `INCONCLUSIVE_DO_NOT_TUNE`

Continuation-token consistency is positive on average but its CI crosses zero.

Stop. Increase sample size only if the preregistered G-0 run accidentally loses too many pairs to tokenizer filtering; otherwise do not method-shop.

### `KILL_NO_INTERNAL_CONSISTENCY_SIGNAL`

The continuation-token consistency CI is non-positive.

Archive the topic.

---

## Files

```text
11_dlm_confidence_internal_consistency/
├── README.md
├── requirements.txt
├── build_design.py       # deterministic mirrored 2×2 dataset
├── score_llada.py        # final-forward token probability scorer
├── analyze.py            # paired effects, bootstrap CIs, locked verdict
├── run_g0.sh             # one-command single-/multi-GPU run
├── configs/
│   └── g0.json           # frozen default settings
└── tests/
    ├── test_design.py
    └── test_analysis.py
```

Generated runs go under `runs/` and are ignored by git.

---

## Environment

Do **not** create another isolated environment if Topic 10 has already prepared a working LLaDA environment. Reuse the same Python environment and Hugging Face cache; Topic 11 introduces no separate training stack.

Minimum imports are only:

- PyTorch
- Transformers
- NumPy

The official LLaDA repository documents `transformers==4.38.2`, which is recorded in `requirements.txt` for reproducibility. If the existing shared environment already loads `GSAI-ML/LLaDA-8B-Instruct` correctly, do not reinstall it just to satisfy the text file.

Model weights are roughly 16 GB in BF16; the scoring script uses one GPU by default and has no inter-GPU communication.

---

## Run

First verify the symbolic construction without loading a model:

```bash
cd 11_dlm_confidence_internal_consistency
python -m unittest discover -s tests -v
```

Single GPU:

```bash
bash run_g0.sh
```

Four independent GPUs on one node:

```bash
NUM_GPUS=4 BATCH_SIZE=8 bash run_g0.sh
```

This launches one LLaDA replica per GPU and shards by **anchor pair**, so mirrored orientations are never split statistically. Workers communicate only through output files; there is no DDP/NCCL requirement and no sensitivity to slow cross-node links.

Useful overrides:

```bash
MODEL=GSAI-ML/LLaDA-8B-Instruct \
NUM_PAIRS=256 \
MIN_PAIRS=128 \
NUM_GPUS=4 \
BATCH_SIZE=8 \
bash run_g0.sh
```

Outputs:

```text
runs/g0/design.jsonl
runs/g0/scores.jsonl
runs/g0/summary.json
runs/g0/summary.md
```

The last file contains the locked verdict and the four-cell means/effects.

---

## What must not be changed after the first real run

Do not respond to an unfavorable result by:

- changing the confidence definition;
- changing the four-cell semantics;
- selecting only templates or arithmetic programs with a larger effect;
- adding an LLM judge;
- changing the anchor range based on scores;
- searching for a special layer or denoising step;
- replacing continuation-token confidence with announcement-token confidence;
- trying many prompt phrasings and reporting the best one.

The tokenizer audit may reject examples **before scoring** because a one-token intervention did not survive tokenization. That filtering is structural and score-blind.

---

## If G-0 passes

Only then is a confirmation run justified. The same frozen factorial can be rerun with more anchor pairs or another compatible LLaDA checkpoint (for example `GSAI-ML/LLaDA-1.5`) by changing only `MODEL`; no new construct is required.

A successful replication would support a compact scientific claim:

> Native diffusion confidence responds to whether the generated reasoning state is internally self-consistent, even when the downstream token sequence is held fixed and final-answer correctness is manipulated independently.

If G-0 fails, do not proceed to this stage.
