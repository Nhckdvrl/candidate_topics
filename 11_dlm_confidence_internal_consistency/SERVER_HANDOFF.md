# Topic 11 — local/server agent handoff

You are taking over `candidate_topics/11_dlm_confidence_internal_consistency/`.

Read `README.md` and `AUDIT.md` before running anything. The scientific goal is to test whether LLaDA's native final-forward confidence contains a **retroactive global consistency signal**: can a semantic contradiction that appears only *after* a fixed reasoning trajectory change confidence assigned to earlier, unchanged reasoning-result tokens?

The v3 design is frozen. External correctness is manipulated only in the prompt; internal consistency only in a future suffix check. Prompt/check anchors are semantic arithmetic aliases rather than literal copies. The primary metric is `confidence_result_middle` (Step 2/3 results before the suffix). Do not replace it with a nicer-looking diagnostic after seeing results.

Run:

```bash
cd 11_dlm_confidence_internal_consistency
python -m unittest discover -s tests -v
NUM_GPUS=4 BATCH_SIZE=8 bash run_g0.sh
```

Reuse the LLaDA/Transformers environment and HF cache already used by Topic 10 if possible; there is no need for a separate environment. Lowering batch size or changing GPU IDs is infrastructure-only and safe.

Two prerequisites must pass before interpreting G-0:
1. seed-paper-like arithmetic result discrimination;
2. semantic-alias comprehension.

If either fails, treat it as an environment/scoring/protocol problem and debug it. Do not kill the research question.

Main output is `runs/g0/summary.md`.

Interpret the frozen verdict literally:
- `GO_RETROACTIVE_CONSISTENCY_SIGNAL`: the topic stands;
- `GO_STRONG_COHERENCE_OVER_CORRECTNESS`: stronger result;
- `INCONCLUSIVE_FROZEN_DESIGN`: keep design fixed; more pairs may be justified;
- `KILL_NO_MEANINGFUL_RETROACTIVE_SIGNAL`: archive if protocol prerequisites passed;
- `INVALID_PROTOCOL_DO_NOT_INTERPRET`: fix engineering/protocol, not science.

You may fix engineering or obvious logic bugs, but preserve the intended factorial identities, one-token tokenizer audit, primary metric, thresholds, and verdict rules. If you find a genuine conceptual flaw in the frozen design, stop and explain it rather than silently changing the hypothesis.
