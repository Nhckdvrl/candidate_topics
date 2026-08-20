# Server-agent handoff prompt

You are validating Topic 05 in `Nhckdvrl/candidate_topics/05_temporal_forgetting_reentry`.

Your goal is **not to prove the hypothesis**. Your goal is to determine as quickly and rigorously as possible what Temporal Forgetting actually means: when the final model reliably fails a problem that the same lineage reliably solved earlier, is the old route mainly inaccessible/self-unselected, or has the route/competence substantially eroded?

Read `README.md`, `VALIDATION.md`, and `RUNBOOK.md` completely before running anything. Treat `VALIDATION.md` as the scientific contract. Do not loosen thresholds or add conditions because initial results are inconvenient.

Use the cluster aggressively. Each node may have independent GPUs and cross-node communication may be slow; that is fine because checkpoint sampling and most scoring are embarrassingly parallel. Prefer one Qwen2.5-7B replica per GPU. If enough nodes are available, assign one of the eight UWNSL checkpoints to each node and sample all checkpoints simultaneously.

## Required phases

1. Clone `uw-nsl/Temporal_Forgetting` and optionally run the official 64-response smoke adapter.
2. Build MATH-500 requests.
3. Sample all eight UWNSL Qwen2.5-7B checkpoints with **16 samples/problem/checkpoint**, temperature=0.6, top_p=0.95.
4. Check truncation; rerun truncated cases at larger max length if needed.
5. Score using the official PRIME/MATH scorer plus Qwen2.5-32B fallback (`hybrid`) before freezing states.
6. Audit scorer accuracy on the stratified manual sample required by `VALIDATION.md`.
7. Run `validate_dataset.py` and `build_forgotten_set.py` with fixed `.75/.125` thresholds and `min_samples=16`.
8. Record F/N/S counts. If F<50, **do not lower thresholds**. Optionally run the same definition on OlympiadBench; otherwise conclude the premise is too sparse.
9. Run `analyze_state_dynamics.py` regardless. If another natural phenomenon dominates (e.g. robust C→W→C / W→C→W oscillations), document it separately, but do not use it to claim Topic 05 succeeded.
10. If G-1 passes, run `select_traces.py`, `match_controls.py`, and `build_reentry_prompts.py`. Manually audit at least 100 prefixes for leakage, step-boundary quality, and token-budget matching before inference.
11. Run G0 re-entry on the final checkpoint with exactly the predeclared conditions/fractions and 8 samples/request. The partial trace must remain an **assistant-generation prefix**, not a user hint. Score with the same hybrid checker and run `analyze_reentry.py`.
12. Run G0-B `trace_likelihood.py` with 0/10/25/50% prefix fractions and summarize with `analyze_trace_likelihood.py`.
13. If G0 is informative, rerun the **same frozen requests** with 16 samples/request for higher-power confirmation. Do not add conditions.
14. Only after G0 is complete, optionally run preregistered G1 relearning-savings analysis.

## Interpretation

- `oldself >> baseline`, `oldself > other_correct`, `oldself > final_wrong`, forgotten-oldself > matched-never-correct, plus favorable old-route NLL = strong evidence that route selection/access is a major component.
- `oldself ≈ other_correct >> baseline`, especially F≈N under matched correct prefixes = generic guidability, not route-specific retention.
- only long prefixes rescue + degraded NLL = partial route/competence erosion.
- non-leaking prefixes fail + F old-route NLL resembles N while S controls work = genuine route/skill loss is a better description.

## Hard no-rescue rules

Do **not**:

- introduce hidden-state probing as a rescue;
- change `.75/.125` state thresholds;
- change the old checkpoint selection rule;
- tune prefix fractions;
- pick a different external correct trace after outcomes;
- cherry-pick problems;
- report discovery-only effects as conclusions.

If something surprising appears outside the main hypothesis, document counts/examples and explain why it is a distinct natural question. Do not silently turn it into the original claim.

## Final deliverable

Write a concise `RESULTS.md` in the topic folder containing:

- exact environment and model revisions;
- sampling/truncation statistics;
- scorer audit;
- F/N/S counts;
- robust-state dynamics histogram;
- matched-control counts;
- prefix audit failures;
- raw re-entry rates;
- all predeclared contrasts + bootstrap CIs, separately for discovery and confirmation;
- NLL curves;
- final conclusion: `PROCEED`, `KILL`, or `NEW PHENOMENON — REGISTER SEPARATELY`;
- if killed, precise reason and what was learned.
