# Topic 05 — Temporal Forgetting: Lost Skill or Lost Entry Point?

**Status: READY FOR FAST VALIDATION**

> When a learner once solved a problem reliably but later fails, has the solution competence been erased, or has the learner mainly lost access to the old route?

## Why this is a natural question

A failure to recall does not uniquely imply that a memory or skill is gone. The old distinction is **availability/storage vs. accessibility/retrieval**. Reasoning-model training gives an unusually clean experiment because we can preserve both:

1. a later checkpoint that now fails a problem, and
2. an earlier checkpoint from the *same training lineage* that previously solved the same problem.

That lets us ask a direct behavioral question:

> **Can the later model re-enter and finish its own former solution route?**

No hidden-state probe is required for the primary test.

## Seed phenomenon

Li et al., **Temporal Sampling for Forgotten Reasoning in LLMs** (ACL 2026) shows Temporal Forgetting across SFT/RL, model scales, and reasoning benchmarks. On average, more than 20% of final errors had been solved at an earlier checkpoint. Their public assets include eight Qwen2.5-7B GRPO checkpoints (`step_32` ... `step_256`), 64-response releases for AIME24/AIME25/AMC, and MATH-500/Olympiad evaluation code.

- Paper: https://aclanthology.org/2026.acl-long.1305/
- Code: https://github.com/uw-nsl/Temporal_Forgetting
- Checkpoints: `UWNSL/Qwen2.5-7B-deepscaler_4k_step_{32,64,...,256}`

Their contribution is **Temporal Sampling**: use old checkpoints at inference time. Our question is different: **what kind of forgetting creates the old/new discrepancy?**

## Nearest collision

Li & Goyal, **Off-Trajectory Reasoning** (ICLR 2026), studies *Guidability*: whether a model can continue a correct partial trace supplied by another model, often on problems outside its demonstrated ability.

Our setting is deliberately stricter and different:

- the problem was demonstrably within this lineage's own earlier capability;
- the primary cue is this lineage's own former correct trajectory;
- forgotten items are compared against matched **never-correct** items;
- repeated sampling defines state membership; one greedy flip is not enough.

Therefore `correct hint helps` is **not** the claim.

## What was fixed after code audit

The initial implementation had several serious confounds. They are now removed:

1. **Incomplete checkpoint coverage** could mislabel an item as `never_correct` → primary classification now requires every expected checkpoint to have enough samples.
2. `bool("False") == True` could corrupt JSONL labels → boolean parsing is explicit.
3. `0%` baseline was emitted once per source → baseline is now emitted exactly once per problem.
4. Control prefixes were not length matched → `other_correct`, `final_wrong`, and matched-N prefixes now use the **old-self token budget** at complete reasoning-step boundaries.
5. Naive substring answer-leak filtering over-rejected legitimate derivations → only explicit answer constructions/boxed expressions trigger automatic rejection, with manual audit required.
6. Teacher-forced NLL ignored the model chat template → likelihood now uses `apply_chat_template(..., add_generation_prompt=True)` and scores assistant suffix tokens only.
7. `never_correct` controls were not explicitly paired → F/N matching is frozen before re-entry.
8. There was no cheap way to surface a different natural dynamic if forgetting was rare → `analyze_state_dynamics.py` reports robust `C/W/U` sequences, but is explicitly **exploratory and cannot rescue this topic**.

## Core validation structure

```text
G-1A  Official 64-response smoke test
      Does strong repeated-sampling forgetting exist at all?
        ↓
G-1B  MATH-500 × 8 checkpoints × 16 samples
      Freeze robust F / N / S sets
        ↓
G0-A  Re-entry experiment on final checkpoint
      old-self vs other-correct vs final-wrong + matched never-correct
        ↓
G0-B  Teacher-forced old-route likelihood
      Is the former route still compatible with the final policy?
        ↓
G1    Relearning savings (secondary, preregistered)
        ↓
G2    SFT / second-model replication only if G0 is informative
```

See [`VALIDATION.md`](./VALIDATION.md) for frozen thresholds and interpretation rules and [`RUNBOOK.md`](./RUNBOOK.md) for exact commands.

## Primary state definitions

With `n >= 16` samples per problem/checkpoint:

- **robust correct**: empirical pass rate `>= 0.75`;
- **robust wrong**: empirical pass rate `<= 0.125`;
- otherwise: uncertain.

Primary groups:

- `F forgotten`: final robust-wrong, at least one earlier robust-correct; select **latest** such old checkpoint;
- `N never_correct`: robust-wrong at every checkpoint;
- `S stable_correct`: final robust-correct and at least one earlier robust-correct.

Wilson intervals are reported for diagnostics, but group membership uses the frozen empirical thresholds above.

### G-1 kill gate

On MATH-500, proceed only if there are at least:

```text
F >= 50
N >= 50
S >= 50
```

If `F < 50`, do **not** loosen thresholds. Expand under the same definition to OlympiadBench or stop this topic.

## G0-A — Re-entry

For a forgotten item with frozen old solution

\[
r^{old}=(r_1,\ldots,r_n),
\]

feed the final model only an early prefix:

\[
x+r^{old}_{1:k}.
\]

Primary fractions are `10%, 25%, 50%` of the old-self reasoning-step sequence. The prefix is appended directly after the model's **assistant generation prompt**, not quoted inside the user message; G0 therefore tests continuation/re-entry rather than ordinary hint following. Every control prefix is truncated to approximately the same tokenizer-token budget.

Conditions:

| condition | purpose |
|---|---|
| `F baseline` | final model without cue |
| `F oldself` | former correct route from same lineage |
| `F other_correct` | MATH-500 canonical worked solution, same item |
| `F final_wrong` | final model's own wrong route |
| `N verified_correct` | correct route on matched problem the lineage never robustly solved |
| `S oldself` | positive-control route continuation |

The main result is **not** `oldself > baseline`; any correct hint can help.

Primary contrasts:

\[
\Delta_{route}(k)=R_{F,oldself}(k)-R_{F,other}(k)
\]

and

\[
\Delta_{history}(k)=R_{F,oldself}(k)-R_{N,correct}(k)
\]

with F/N pairs matched on subject, level, and prompt length proxy and prefix token budget inherited from F.

## G0-B — Former-route compatibility

Free generation may fail simply because the final policy does not select an old route. We separately teacher-force the former trace and measure per-token suffix NLL:

\[
-\frac{1}{|s|}\log p_{final}(s\mid x,r^{old}_{1:k}).
\]

Useful pattern:

```text
free generation fails
+ short old-self prefix selectively rescues
+ old-route suffix NLL remains favorable vs never-correct controls
```

This supports **selection/access failure** without making a stronger unobservable claim that a particular hidden representation is literally stored.

## Four informative outcomes

| result | interpretation |
|---|---|
| short `oldself` selectively rescues F and old-route NLL is favorable | **lost entry / route selection** is a major component |
| `oldself ≈ other_correct`, both rescue F | forgotten problems remain broadly guidable, but no evidence for old-route-specific retention |
| only long prefixes rescue, and old-route NLL has degraded | partial competence / route erosion |
| even long non-leaking prefixes fail and old-route NLL resembles never-correct controls | strong evidence that robust cases are closer to genuine route/skill loss |

A clean negative therefore still characterizes the seed phenomenon.

## If the original hypothesis is weak: what else is allowed to be inspected?

`analyze_state_dynamics.py` reports robust checkpoint-state sequences such as:

- `W→C`: late acquisition;
- `C→W`: forgetting;
- `C→W→C` or repeated flips: **reasoning-state volatility / oscillation**;
- long uncertain regions.

This is permitted because it is descriptive and predetermined. **It cannot be used to relabel a failed re-entry study as successful.** Any genuinely interesting alternative phenomenon must be registered as a new topic with a new independent gate.

## Code map

```text
code/common.py                    shared parsing / leakage / CI utilities
code/convert_official_release.py official 64-response adapter
code/prepare_math500_requests.py  MATH-500 request builder
code/run_vllm_generate.py         embarrassingly-parallel vLLM generation
code/score_math_samples.py        official PRIME scorer + optional 32B fallback
code/validate_dataset.py          integrity audit
code/build_forgotten_set.py       robust C/W/U state construction
code/analyze_state_dynamics.py    exploratory state-sequence audit
code/select_traces.py             freeze old/gold/wrong traces before G0
code/match_controls.py            forgotten ↔ never-correct matching
code/build_reentry_prompts.py     token-budget-matched re-entry requests
code/analyze_reentry.py           locked bootstrap contrasts
code/trace_likelihood.py          chat-template-correct teacher-forced NLL
scripts/run_4gpu_sharded.sh       one model replica / GPU launcher
```

## Fast start

```bash
cd 05_temporal_forgetting_reentry

# 1. Prepare MATH-500 once.
python code/prepare_math500_requests.py --output data/math500_requests.jsonl

# 2. Generate each checkpoint independently (ideally one node/checkpoint).
MODEL=UWNSL/Qwen2.5-7B-deepscaler_4k_step_256 \
INPUT=data/math500_requests.jsonl OUTDIR=results/raw_step256 N=16 \
bash scripts/run_4gpu_sharded.sh

# 3. Score with the seed repository's grader.
python code/score_math_samples.py \
  --input results/raw_step256/all.jsonl \
  --output results/scored_step256.jsonl \
  --temporal-repo /path/to/Temporal_Forgetting \
  --method hybrid \
  --checkpoint step_256 --checkpoint-order 7

# 4. Concatenate eight scored checkpoint files, then classify.
cat results/scored_step*.jsonl > results/checkpoint_samples.jsonl
python code/validate_dataset.py --input results/checkpoint_samples.jsonl
python code/build_forgotten_set.py \
  --input results/checkpoint_samples.jsonl \
  --output results/groups.jsonl \
  --min-samples 16
python code/analyze_state_dynamics.py \
  --groups results/groups.jsonl \
  --output-json results/state_dynamics.json
```

Continue with [`RUNBOOK.md`](./RUNBOOK.md).
