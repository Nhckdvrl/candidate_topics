# Frozen one-shot G0 validation contract

The old prompt explicitly instructed the model to choose the command with the
better observed outcome. That measured arithmetic/instruction following, not
experience-driven behavioral adaptation. Version 2 instead states a standing
score-maximization goal, presents an episode, inserts interference, and asks for
the next action without restating the comparison rule.

Each base cell now contains `positive`, `negative`, and equal-outcome `baseline`
conditions. A complete 64-cell block crosses eight symbol pairs with marked
identity, observation order, and answer-option order. The baseline is scored as
a preference control, never as correct/incorrect.

## Frozen panel gate

- minimum 64 base cells and three predeclared model families;
- at most 2% unparsable outputs;
- baseline marked-action deviation at most 10 points. A within-symbol-pair
  preference over 25 points is reported as a warning, but does not invalidate a
  complete design because marked identity is crossed and its aggregate baseline
  rate is exactly what tests residual confounding;
- `SURVIVE` only when pooled paired delta is at least 0.20, its bootstrap lower
  bound is at least 0.10, every model delta is at least 0.10, and every identity/
  order stratum delta is at least 0.10;
- `KILL` when the pooled 95% upper bound is below 0.10 **or** at most one model
  reaches a 0.10 gap (the README's original “only one model” kill line);
- otherwise `INCONCLUSIVE`; failed prerequisites produce `INVALID`.

Generate, run local models, and score:

```bash
.venv/bin/python 18_negative_behavioral_adaptation/generate_g0.py \
  --output design.jsonl --n-base 64

.venv_clean2/bin/python 18_negative_behavioral_adaptation/run_local_models.py \
  --design design.jsonl --output predictions.jsonl \
  --model family_a:model_a=/local/path/a \
  --model family_b:model_b=/local/path/b \
  --model family_c:model_c=/local/path/c

.venv/bin/python 18_negative_behavioral_adaptation/score_g0.py \
  --design design.jsonl --predictions predictions.jsonl
```

The runner flushes each completed batch. After an interruption, repeat the same
command with `--resume`; already persisted `(model_id, item_id)` rows are skipped.

Model IDs/paths, model revisions, decoding settings, and the label set must be
frozen before inspecting positive/negative outcomes. A failed baseline means
`INVALID`, not permission to replace labels after looking at the effect.
The local runner disables optional Qwen-style thinking mode for every item so
the token-only action, rather than a truncated reasoning preamble, is scored.

The completed frozen run and its stop decision are documented in
[G0_RESULTS.md](./G0_RESULTS.md).
