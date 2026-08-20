# DLM Trajectory Fate

## Status

**ARCHIVED / FALSIFIED AS A BROAD CLAIM.**

The project showed a promising exploratory signal, but both preregistered final-outcome-controlled effects failed on independent GSM1K confirmation while the final-correctness positive control remained strong. The topic is therefore stopped rather than extended with new layers, steps, parsers, probe families, or post-hoc thresholds.

See:

- [`ARCHIVE_SUMMARY.md`](./ARCHIVE_SUMMARY.md) — complete failure summary and lessons;
- [`G0_RESULTS.md`](./G0_RESULTS.md) — initial G0 results;
- [`SECOND_STAGE_RESULTS.md`](./SECOND_STAGE_RESULTS.md) — independent confirmation results;
- [`VALIDATION.md`](./VALIDATION.md) and [`STAGE2_PROTOCOL_REVISION.md`](./STAGE2_PROTOCOL_REVISION.md) — validation contracts and controls.

**Question:** before a visible denoising transition happens, does a DLM hidden state contain information about whether the current answer will recover or be overwritten?

Final conclusion: transient surface events exist, but the proposed new claim that a single current hidden state robustly predicts future transient fate after controlling current correctness and final outcome did not survive confirmation.

---

## Why this was a candidate topic

Two 2026 results left a narrow adjacent gap:

1. **Time Is a Feature** / `dLLM-MidTruth` shows that complete intermediate `x0` predictions can oscillate during denoising: an answer can become correct and later become wrong again.
2. **Probing Functional Correctness in Diffusion Language Models** / `dlm-probing` shows that DLM hidden states increasingly predict **final** functional correctness.

The proposed adjacent question was to replace the target `final correctness` with `future fate of the current state`.

However, a naive version is confounded: if we label a currently-wrong state as “recoverable” whenever it later becomes correct, the probe can succeed simply by reading the already-known **final-correctness** signal. The primary novelty test therefore used **final-outcome-controlled** labels.

## Primary scientific test

At a fixed denoising step, condition on current surface correctness **and final outcome**.

### Transient recovery

Among trajectories that are **wrong now and wrong at the end**:

- positive: they become observably correct at least once later (`wrong -> correct -> wrong`);
- negative: they never become observably correct later.

### Transient overwrite

Among trajectories that are **correct now and correct at the end**:

- positive: they become observably wrong at least once later (`correct -> wrong -> correct`);
- negative: they remain observably correct.

Because the final outcome is identical inside each comparison, success cannot be explained by merely reproducing the known final-correctness probe.

## Important implementation choices

- **Surface state = complete `x0` before token transfer.** This matches `dLLM-MidTruth` temporal voting.
- **No-answer-yet is not wrong.** Strict parsing stores an `observed` mask.
- **Deterministic denoising is the primary G0.** `temperature=0` removes future Gumbel randomness invisible to the current hidden state.
- **Same-step comparisons.** Positives and negatives are compared at the same absolute denoising step.
- **Three controls per probe.** Current hidden state is compared with observable uncertainty/progress features and the same layer at step 0.
- **Problem-level independence.** No `(problem, step)` leakage.

## Validation path that was run

```text
surface-event census
-> hidden-state exploratory G0
-> positive-control check
-> preregistered independent GSM1K confirmation
```

The exploratory stage motivated confirmation, but the confirmation did not reproduce the proposed novel effects strongly enough to satisfy the preregistered gates.

## Final decision

```text
EXPLORATORY SIGNAL: PRESENT
INDEPENDENT CONFIRMATION: FAILED
POSITIVE CONTROL: PASSED
FINAL: ARCHIVED
```

Do not revive the project by searching additional denoising steps, layers, lead thresholds, answer parsers, or probe families on the same hypothesis. A restart would require a genuinely different scientific question, not a post-hoc rescue of this one.

## Reference code

- Time Is a Feature / dLLM-MidTruth: https://github.com/aim-uofa/dLLM-MidTruth
- Probing Functional Correctness in Diffusion Language Models / dlm-probing: https://github.com/guan404ming/dlm-probing
