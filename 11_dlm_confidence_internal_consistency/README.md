# 11 — What Does Diffusion Confidence Actually Know?

**Status:** VALIDATION CODE HARDENED — run the locked G-0 before any extension

## Scientific question

> Does native diffusion-LM confidence track the **internal consistency of a reasoning trajectory** independently of whether its conclusion is externally correct?

This is deliberately narrower than “DLLM confidence is not calibrated.” The seed paper, [The Confidence Paradox](https://aclanthology.org/2026.findings-acl.2142/), reports that LLaDA-8B is badly calibrated on mathematical reasoning but has strong discriminative power. Its interpretation is that the final-forward diffusion confidence behaves more like a signal of structural consistency than an ordinary probability of correctness.

The paper also shows a strong arithmetic intervention: changing a correct arithmetic result to a wrong one causes a much larger confidence drop than a factual answer swap. What it does **not** identify is whether internal trajectory consistency itself can be separated from external correctness.

Topic 11 tests exactly that claim with one programmatic factorial experiment.

---

## Why the naive 2×2 is invalid

A construction such as:

- correct reasoning + correct answer;
- correct reasoning + wrong final token;
- wrong reasoning + correct answer;
- wrong reasoning + wrong answer;

is not a clean factorial. If the chain derives `42` and the final line says `37`, the supposedly “valid reasoning + wrong answer” condition is internally contradictory by construction.

Likewise, creating an arithmetic error and then “propagating it consistently” still leaves one locally false equation at the point where the error is introduced.

The implemented design avoids both problems: **every arithmetic equation in every cell is valid**.

---

## Locked 2×2 design

For an anchor pair `X != Y`, choose a deterministic three-step arithmetic program `f`. In one orientation, the downstream trajectory always computes from branch anchor `X`.

| Cell | Prompt anchor | Announced anchor | Downstream branch | Internal consistency | External correctness |
|---|---:|---:|---:|---|---|
| `CC` | X | X | X | consistent | correct |
| `IC` | X | Y | X | inconsistent | correct |
| `CW` | Y | X | X | consistent | wrong |
| `IW` | Y | Y | X | inconsistent | wrong |

Example:

```text
Prompt: A calculator starts from 23. Add 5, multiply by 3, subtract 4.

CC output:
Initial state: 23
Step 1: 23 + 5 = 28
Step 2: 28 * 3 = 84
Step 3: 84 - 4 = 80
Final answer: 80

IC output:
Initial state: 29
Step 1: 23 + 5 = 28
Step 2: 28 * 3 = 84
Step 3: 84 - 4 = 80
Final answer: 80
```

Within one orientation:

- the downstream continuation is **byte-identical in all four cells**;
- `CC` vs `IC` changes only the announced anchor;
- `CW` vs `IW` changes only the announced anchor;
- `CC` vs `CW` holds the complete output fixed and changes only the prompt anchor;
- `IC` vs `IW` does the same.

The same anchor pair is then mirrored with `Y` as the downstream branch. Effects are averaged across the two orientations before any bootstrap. This cancels stable preferences for one number token.

### Tokenizer-level minimality

Text-space matching is not enough. The scorer rejects an orientation unless:

- prompt manipulation changes at most one token (default: exactly one under `max_intervention_tokens=1`);
- announcement manipulation changes at most one token;
- total token length is identical across the four cells;
- downstream token IDs are identical across all four cells;
- all scored arithmetic-result token positions are identical.

Both orientations must pass or the entire anchor pair is dropped.

---

## Why we score late result tokens

A remaining shallow explanation is locality: the inconsistent announcement is adjacent to Step 1, so a drop on the first equation could just be local numeric compatibility.

The v2 scorer therefore records four nested views of the **same forward pass**:

1. `confidence_result_first` — Step-1 result only; a local-sensitivity diagnostic.
2. `confidence_result_late` — arithmetic result tokens from Step 2 onward; **primary identification metric**.
3. `confidence_tail` — every token in the unchanged downstream continuation; breadth diagnostic.
4. `confidence_full` — every output token; the seed-paper-compatible sequence score.

It also saves `confidence_result`, `confidence_final`, and `confidence_announcement` as diagnostics.

The primary claim is deliberately tied to `confidence_result_late`: if changing only the announced initial state changes confidence on later, text-identical arithmetic results several reasoning steps away, the signal is harder to explain as a one-token local mismatch.

No extra model calls are required for these metrics.

---

## Scoring protocol: match the seed paper first

The seed paper defines LLaDA sequence confidence using a **final teacher-forced forward pass on the fully specified sequence**:

1. format the user prompt with the LLaDA chat template;
2. append the prescribed response;
3. run one final forward pass;
4. at every scored output position, read the softmax probability assigned to the token already occupying that position;
5. average the selected token probabilities.

We do **not** average confidence over the 128 denoising steps, and G-0 does not need to generate a response.

### Positive-control protocol audit

Before the factorial is interpreted, shard 0 also runs 100 synthetic arithmetic pairs matching the seed paper's intervention idea:

```text
23 + 45 = 68   # correct
23 + 45 = 72   # one-token wrong result
```

Only one-token result substitutions are kept. The same final-forward scorer reads probability on the result token.

If the paired 95% CI for:

```text
confidence(correct result) - confidence(wrong result)
```

does not lie above zero, the run is labelled:

```text
INVALID_PROTOCOL_DO_NOT_INTERPRET
```

A failed positive control is a broken/changed scoring protocol, model revision, or environment until proven otherwise. It must **not** be used to kill the research question.

This is a prerequisite check, not a control added to rescue the hypothesis.

---

## Locked effects

For one orientation and any confidence metric `c`:

```text
consistency_when_correct = CC - IC
consistency_when_wrong   = CW - IW
Delta_consistency = 0.5 * [(CC - IC) + (CW - IW)]

correctness_when_consistent   = CC - CW
correctness_when_inconsistent = IC - IW
Delta_correctness = 0.5 * [(CC - CW) + (IC - IW)]

coherent_wrong_minus_incoherent_correct = CW - IC
interaction = (CC - IC) - (CW - IW)
```

The two mirrored orientations are averaged first. The statistical unit is the **anchor pair**, never the individual row or orientation.

Uncertainty is reported with:

- pair-level bootstrap 95% confidence intervals;
- a one-sided pair-level sign-flip randomization p-value as a diagnostic;
- fraction of pairs with positive effect.

The bootstrap CI remains the locked decision criterion; the permutation p-value is not an alternate rescue gate.

---

## G-0 decision rule

The decision hierarchy is frozen before model scores are seen.

### 0. `INVALID_PROTOCOL_DO_NOT_INTERPRET`

The arithmetic positive-control CI includes zero.

Stop. Fix the scorer/model/environment. Do not interpret the factorial.

### 1. `KILL_NO_INTERNAL_CONSISTENCY_SIGNAL`

Protocol audit passes, but the 95% CI for `Delta_consistency` on **late downstream result tokens** is entirely non-positive.

Archive the topic.

### 2. `INCONCLUSIVE_DO_NOT_TUNE`

The late-result consistency effect is positive on average but its CI crosses zero.

Do not prompt-shop, change metrics, or add error taxonomies. Increase sample size only if many preregistered pairs were lost to tokenizer filtering for a purely mechanical reason.

### 3. `MIXED_LOCAL_RESULT_SIGNAL_ONLY`

Late-result consistency is stable, but the seed-paper-compatible `confidence_full` consistency effect is not stable.

There is an interesting local/structured score effect, but not enough evidence for the intended broad confidence interpretation.

### 4. `MIXED_INTERNAL_SIGNAL_ONLY`

Late-result and full-output consistency effects are stable, but `CW - IC` on late result tokens does not stably exceed zero.

Interpretation: native confidence contains a real internal-consistency signal independent of correctness, but the stronger story that consistency dominates correctness is not established.

### 5. `GO_STRONG_STRUCTURAL_SIGNAL`

Required:

1. arithmetic protocol audit passes;
2. late-result `Delta_consistency` CI is entirely above zero;
3. full-output `Delta_consistency` CI is entirely above zero;
4. late-result `CW - IC` CI is entirely above zero.

The all-tail score is reported as breadth evidence but is deliberately **not** another gate; averaging punctuation and boilerplate should not decide whether the scientific signal exists.

---

## Efficiency

This experiment is intentionally cheap.

With the default `256` mirrored anchor pairs:

- 8 factorial rows per pair = 2,048 short sequences;
- 100 arithmetic protocol-control pairs = 200 very short sequences;
- each sequence requires exactly **one** model forward pass;
- no diffusion generation;
- no training;
- no hidden-state extraction;
- no LLM judge;
- no DDP/NCCL.

Multi-GPU mode launches independent model replicas and shards by **complete anchor pair**. There is no inter-GPU communication. That makes it suitable for the same infrastructure style as Topic 10 and insensitive to slow cross-node links.

The scorer batches only sequences of **exactly equal token length**. It therefore needs no padding and no attention-mask behavior, avoiding a subtle compatibility/confounding issue across LLaDA remote-code revisions.

---

## Environment: reuse Topic 10

Do **not** create a separate Topic-11 environment if Topic 10 already has a working LLaDA environment/cache.

Topic 11 now deliberately uses the same compatible range:

```text
torch>=2.4
transformers>=4.49,<5
pytest>=8
```

and only adds an explicit NumPy dependency (normally already present through Transformers).

The old Topic-11 pin `transformers==4.38.2` was removed because it would force a second environment. The scorer uses `AutoModel`/`AutoTokenizer` with `trust_remote_code=True` and avoids the padding path that caused the main compatibility concern.

Before loading 8B weights, `run_g0.sh` checks the installed versions and performs a tokenizer-only design audit.

---

## Run

### Cheap static tests

```bash
cd 11_dlm_confidence_internal_consistency
python -m unittest discover -s tests -v
```

### Single GPU

```bash
bash run_g0.sh
```

### Four GPUs

```bash
NUM_GPUS=4 BATCH_SIZE=8 bash run_g0.sh
```

If the machine's desired physical GPU ids are not `0,1,2,3`:

```bash
NUM_GPUS=4 GPU_IDS=2,3,6,7 bash run_g0.sh
```

Infrastructure-only overrides are allowed, e.g. batch size/GPU ids/run directory. Scientific design changes should be made in a new config rather than silently overriding the frozen G-0.

### Outputs

```text
runs/g0/design.jsonl
runs/g0/scores.jsonl
runs/g0/protocol_probe.jsonl
runs/g0/runtime.json
runs/g0/summary.json
runs/g0/summary.md
```

`runtime.json` records the requested model revision, resolved model commit when exposed by Transformers, library versions, dtype, and tokenization-audit counts.

---

## Files

```text
11_dlm_confidence_internal_consistency/
├── README.md
├── AUDIT.md
├── requirements.txt
├── build_design.py
├── score_llada.py
├── analyze.py
├── run_g0.sh
├── configs/
│   └── g0.json
└── tests/
    ├── test_design.py
    ├── test_analysis.py
    └── test_scoring_utils.py
```

Generated runs are ignored by git.

---

## What must not be tuned after seeing scores

Do not rescue a weak/null G-0 by changing:

- which cell comparison is primary;
- which result step is called primary;
- templates selected after the fact;
- anchor ranges because some numbers “look better”;
- confidence aggregation after looking at model outputs;
- arithmetic operation families;
- model prompt wording;
- bootstrap seed;
- error categories;
- an LLM judge or learned verifier.

A clean null should kill the topic, not start a measurement search.

---

## Why a positive result matters

The interesting result is not “LLaDA notices arithmetic errors”; the seed paper already established that.

The stronger result would be:

> With external correctness factorially separated, changing only whether a trajectory agrees with its own announced state systematically changes native diffusion confidence on **later, text-identical downstream result tokens**.

And in the strongest case:

> A coherent-but-externally-wrong trajectory receives higher late-result confidence than an externally-correct but internally-broken trajectory.

That would make the “structural consistency” interpretation substantially more identified than a correct-vs-wrong AUROC correlation, without introducing probes, hidden representations, training, or a complicated stack of controls.
