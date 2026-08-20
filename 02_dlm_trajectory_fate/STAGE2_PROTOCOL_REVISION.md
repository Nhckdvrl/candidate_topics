# Stage-2 Protocol Revision: Full-Support Confirmation

## Why this revision exists

The first Stage-2 execution stopped after a 200-example GSM1K surface preflight. The locked counts were:

| task | positive | negative | old gate |
|---|---:|---:|---|
| transient recovery, step 16 / lead >= 4 | 4 | 26 | fail (`positive < 6`) |
| transient overwrite, step 4 / lead >= 16 | 5 | 13 | fail (`positive < 6`, `negative < 20`) |

Crucially, **no GSM1K hidden-state confirmation had been run or inspected when this protocol problem was identified**. The scientific cells were still locked; only surface event counts were known.

The old 200-example gate had been added as a compute-saving heuristic, but it was never power-calibrated. Using the G0 locked-cell frequencies as the null expectation for a perfectly stable phenomenon:

```text
recovery:  41 positive / 1000, 109 negative / 1000
overwrite: 46 positive / 1000, 120 negative / 1000
```

an exact multinomial calculation gives only about:

```text
P(old 200-example gate passes | G0 recovery rates)  = 0.569
P(old 200-example gate passes | G0 overwrite rates) = 0.751
```

So even if GSM1K had the same event rates as G0, the old gate had an unacceptably high false-stop probability. It was therefore not a valid scientific kill criterion.

This revision removes that heuristic stopping rule. It does **not** change any hypothesis, step, layer, lead threshold, parser, sampler, probe family, or confirmation statistic.

## What remains frozen

The two G0 discoveries remain exactly:

| hypothesis | task | step | LLaDA hidden tuple index | minimum lead |
|---|---|---:|---:|---:|
| H1 | transient recovery | 16 | 25 | 4 |
| H2 | transient overwrite | 4 | 28 | 16 |

Primary geometry remains:

```text
model             GSAI-ML/LLaDA-8B-Instruct
dataset           ScaleAI/gsm1k test
examples          all 1,205
steps             64
generation        128 tokens
block length      32
temperature       0
prompt/parser     MidTruth strict numeric boxed answer
capture steps     [0,4,16,63]
hidden indices    [25,28]
probe             StandardScaler -> PCA(max 64) -> LogisticRegression
CV                5-fold stratified
bootstrap         2,000
```

The original confirmatory statistic is unchanged:

```text
M = min(
  AUC_hidden - 0.55,
  (AUC_hidden - AUC_surface) - 0.03,
  (AUC_hidden - AUC_step0) - 0.03
)
```

A supported task confirms only when the one-sided 97.5% bootstrap lower bound of `M` is > 0.

## Revised support rule

Support is now decided on **all 1,205 GSM1K examples**, not a 200-example prefix.

For each locked task independently:

```text
positive >= 25
negative >= 25
```

This matches the minimum class count already required by the confirmatory probe.

Possible support outcomes:

```text
GO_BOTH  -> fit both locked probes
GO_ONE   -> fit/report both, but only the supported task is eligible to confirm
STOP_LOW_LOCKED_SUPPORT -> both tasks are too sparse on full GSM1K; strongly demote/kill the broad claim
```

No new step or lead threshold is searched when support is low.

## Efficiency: one full generation, not two

The revised runner does not first run 1,205 surface-only trajectories and then repeat the same 1,205 examples with hidden extraction.

Instead it generates each example once and stores only:

```text
4 denoising steps x 2 locked layers x 1 global pooled vector
```

The surface-support script reads only correctness/observation arrays first. Hidden probes are fitted only if the full-dataset support rule passes. This preserves the decision order while avoiding a second 64-step forward pass over the whole dataset.

## Exact execution for the current repository state

The historical G1-A audit has already been completed, so the server should resume with:

```bash
cd candidate_topics
git checkout main
git pull --ff-only
cd 02_dlm_trajectory_fate

python -m pytest -q tests/test_stage2.py
python -m py_compile src/*.py
bash -n run_stage2_retry_4gpu.sh

GPUS="0 1 2 3" ./run_stage2_retry_4gpu.sh
```

The retry runner performs:

```text
1. LLaDA / GSM1K / all 1,205: one locked hidden+surface generation
2. full-dataset locked support count
3. if GO_ONE or GO_BOTH: locked hidden confirmation
4. if CONFIRM_ONE or CONFIRM_BOTH: Dream / GSM1K / all 1,205
5. Dream full-dataset support count
6. if supported: locked Dream confirmation
```

`run_stage2_4gpu.sh` is also updated so a fresh run from scratch uses the corrected protocol.

## Decision after G1-B

### If full GSM1K support is low for both tasks

The phenomenon is too sparse on distribution-matched new questions for the broad claim. Do not change the locked cells. Strongly demote the topic; in practice this is close to a kill unless a much narrower GSM8K-specific scientific claim is desired.

### If one or both tasks have enough support, but locked hidden confirmation fails

This is the decisive winner's-curse test. If the final-correctness positive control is valid and the locked hidden task fails the original confirmation statistic, **kill that hypothesis**. If both fail, kill Topic 02 rather than searching neighboring cells.

### If exactly one task confirms

Narrow the project to that direction. The other G0 direction is treated as non-replicated.

### If one or both confirm

Proceed to Dream only for cross-model generality. Dream failure cannot retroactively erase a confirmed LLaDA/GSM1K effect, but it limits the model-general claim.

## Historical transparency

`SECOND_STAGE_RESULTS.md` is retained as the record of the first execution. Its `STOP_LOW_LOCKED_SUPPORT` status describes the **old 200-example heuristic**, not a completed GSM1K hidden-state negative result. This protocol revision was made after seeing only the 200-example surface counts and before any GSM1K hidden-state confirmation.
