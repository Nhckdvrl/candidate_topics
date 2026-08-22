# 11 — What Does Diffusion Confidence Actually Know?

**Status:** VALIDATION v3 LOCKED — retroactive G-0 ready to run

## Scientific question

> Does native diffusion-LM confidence encode **global internal consistency of a reasoning trajectory**, independently of whether the trajectory is externally correct?

The motivating result is *The Confidence Paradox* (ACL Findings 2026): LLaDA confidence is poorly calibrated as a probability of correctness yet strongly discriminative on mathematical reasoning, and arithmetic contradictions cause large confidence drops. The open question is what that confidence is actually reading.

We want a first experiment that can establish a real phenomenon before inventing any method.

---

## Why v2 was still not decisive enough

The previous design put an "announced initial state" immediately before the arithmetic trajectory. A consistent condition therefore contained a literal number match:

```text
Initial state: 23
Step 1: 23 + 5 = ...
```

while the inconsistent condition did not.

Even if we scored later tokens, a bidirectional DLM could propagate this raw match/mismatch signal through the full sequence. A positive result would therefore still admit the shallow explanation "the same number appeared twice."

There was a second issue: the internal-consistency cue sat immediately next to the trajectory, while external correctness was encoded far away in the prompt. Comparing their magnitudes could partly measure textual position rather than semantic priority.

v3 removes both problems instead of adding post-hoc controls.

---

## v3: a retroactive factorial

Choose two anchors `X != Y` and a deterministic three-step arithmetic program. In one orientation the **trajectory always computes from X**.

The trajectory itself is identical in every cell:

```text
Step 1: 23 + 5 = 28
Step 2: 28 * 3 = 84
Step 3: 84 - 4 = 80
Final answer: 80
```

Two independent relations are manipulated:

1. **External correctness** — only the user prompt changes which starting value is ground truth.
2. **Internal consistency** — only a suffix *after the trajectory* changes which starting value the completed derivation claims to have used.

The factorial is:

| Cell | Prompt truth | Future consistency check | Fixed trajectory | Internally consistent | Externally correct |
|---|---|---|---|---|---|
| `CC` | X | X | computes X | yes | yes |
| `IC` | X | Y | computes X | no | yes |
| `CW` | Y | X | computes X | yes | no |
| `IW` | Y | Y | computes X | no | no |

The X/Y roles are then mirrored and averaged before inference.

### No literal anchor copying

Prompt and suffix do **not** contain the literal branch anchor.

For branch value 23, for example:

```text
prompt truth: 7 + 16
trajectory:   ... starts from 23 ...
future check: 11 + 12
```

Prompt and check use different arithmetic decompositions. The builder rejects accidental cases where the changed residual equals an anchor, a trajectory state, or an operation value.

After tokenization, both manipulations must be no more than one changed token (locked default: one token), with identical sequence lengths and identical trajectory token IDs. Both mirrored orientations must pass or the pair is discarded.

---

## The decisive readout: confidence before the contradiction

The internal-consistency intervention occurs **after** the entire trajectory.

The primary score is:

```text
confidence_result_middle
```

which averages the observed-token confidence on the Step-2 and Step-3 result tokens.

Those tokens:

- occur **before** the consistency-check suffix;
- are text- and token-identical in all four cells;
- are away from both the prompt boundary and the suffix boundary;
- are never directly edited.

Therefore:

> If changing a future consistency check changes confidence on earlier, unchanged reasoning results, the effect is genuinely retroactive under the DLM final-forward score.

This is much harder to explain as changed-token surprise, immediate numeric copying, or a local mismatch.

The same forward pass also reports:

- `confidence_result_first`
- `confidence_result_final`
- `confidence_result_all`
- `confidence_trajectory`
- `confidence_full` — paper-compatible all-output score
- `confidence_check` — manipulated suffix diagnostic

None can replace the locked primary after results are seen.

---

## Confidence scoring

`score_llada.py` follows the seed paper's final-forward protocol:

1. chat-format the user prompt;
2. append the prescribed complete output;
3. run one teacher-forced LLaDA forward pass;
4. at each scored position, read the softmax probability assigned to the token occupying that position;
5. average the predeclared positions.

No 128-step diffusion generation is required for this causal scoring experiment.

Examples are batched only with exactly equal token length, so the run does not depend on padding/attention-mask behavior.

---

## Two prerequisites prevent false scientific nulls

A broken model revision, tokenizer path, or scoring implementation must not be allowed to kill the topic.

Before the factorial is interpreted, the same model/scorer runs two 100-pair probes.

### 1. Seed-paper-like arithmetic probe

Compare a correct arithmetic result with a one-token wrong result.

Locked requirement:

- paired 95% CI is above zero; and
- mean confidence gap is at least **0.10**.

The seed paper reports a much larger arithmetic contradiction effect, so this floor is deliberately conservative.

### 2. Semantic-alias comprehension probe

The v3 design relies on expressions like `7 + 16` denoting 23. We therefore verify that the scorer responds to that semantic relation.

Example:

```text
Equality: 7 + 16 = 23   # correct
Equality: 7 + 20 = 23   # wrong
```

The target `23` is unchanged; only one earlier operand changes.

Locked requirement:

- paired 95% CI is above zero; and
- mean target-confidence gap is at least **0.02**.

If either prerequisite fails, the verdict is:

```text
INVALID_PROTOCOL_DO_NOT_INTERPRET
```

That is an engineering/protocol failure, **not evidence against the research hypothesis**.

---

## Locked effects

For each orientation:

```text
consistency_when_correct = CC - IC
consistency_when_wrong   = CW - IW
Delta_consistency = 0.5 * (consistency_when_correct + consistency_when_wrong)

correctness_when_consistent   = CC - CW
correctness_when_inconsistent = IC - IW
Delta_correctness = 0.5 * (correctness_when_consistent + correctness_when_inconsistent)

coherent_wrong_minus_incoherent_correct = CW - IC
```

The two mirrored orientations are averaged at the anchor-pair level first. Bootstrap and sign-flip inference operate on those pair-level effects.

### Primary scientific effect size

A statistically nonzero epsilon is not enough.

The locked meaningful floor for the primary retroactive consistency effect is:

```text
Delta_consistency >= 0.01
```

(one percentage point in mean token probability).

---

## Verdict logic

The design is intentionally asymmetric between "does the phenomenon exist?" and "is the strongest headline true?"

### `GO_RETROACTIVE_CONSISTENCY_SIGNAL`

The topic **stands** if:

- both protocol prerequisites pass;
- primary `Delta_consistency` has 95% CI entirely above zero;
- mean effect is at least 0.01;
- the effect points positive in both external-correctness strata.

This is already a substantive result: a future semantic contradiction changes confidence assigned to earlier unchanged reasoning.

### `GO_STRONG_COHERENCE_OVER_CORRECTNESS`

Everything above, plus:

```text
CW - IC > 0
```

with 95% CI above zero.

Then coherent-but-wrong reasoning is more trusted than incoherent-but-correct reasoning.

This is the strongest headline, **not a requirement for keeping the project alive**.

### `MIXED_INTERACTION_DEPENDENT`

A positive average exists but reverses sign in one correctness stratum. The simple "independent consistency signal" story is not established.

### `WEAK_REAL_BUT_TOO_SMALL`

The effect is stably positive but smaller than the preregistered meaningful floor.

### `INCONCLUSIVE_FROZEN_DESIGN`

The CI includes zero but still allows an effect >= 0.01. Do **not** redesign or metric-shop. If needed, increase only the number of pairs under the frozen v3 design.

### `KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL`

The upper 95% CI is below 0.01. The experiment has excluded a consistency effect large enough to matter.

This is a real scientific negative and is the appropriate point to archive the topic.

---

## Run

Topic 11 intentionally reuses the same LLaDA environment/cache as Topic 10. Do not create another environment unless the shared one is actually incompatible.

From the repository root:

```bash
cd 11_dlm_confidence_internal_consistency
python -m unittest discover -s tests -v
NUM_GPUS=4 BATCH_SIZE=8 bash run_g0.sh
```

For selected GPUs:

```bash
NUM_GPUS=2 GPU_IDS=1,3 bash run_g0.sh
```

Outputs:

```text
runs/g0/design.jsonl
runs/g0/runtime.json
runs/g0/protocol_probe.jsonl
runs/g0/scores.jsonl
runs/g0/summary.json
runs/g0/summary.md
```

`summary.md` is the first file to read.

Scientific knobs in `configs/g0.json` are read as locked values by `run_g0.sh`; only infrastructure knobs such as GPU IDs/count and batch size are overrideable. The run directory also records the locked config and repository commit.

---

## What not to do after seeing the result

Do not:

- change the primary metric from Step-2/3 results to whichever diagnostic looks best;
- change the 0.01 meaningful-effect floor;
- loosen protocol gates because the factorial looks exciting;
- add hand-selected subsets, prompt sweeps, or error taxonomies to rescue a null;
- interpret `INVALID_PROTOCOL_DO_NOT_INTERPRET` as a scientific failure.

Engineering fixes that restore the exact frozen measurement are allowed. Scientific redesign after inspecting G-0 is not.

---

## If G-0 is strong

Only then move to G-1:

1. replicate on a more natural programmatically checkable reasoning family;
2. test at least one additional masked/diffusion LM;
3. study how far backward the retroactive consistency signal propagates.

Those are follow-ups. They are not prerequisites for deciding whether Topic 11 is a real research problem.
