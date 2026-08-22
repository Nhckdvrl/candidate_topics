# G0-v3 published 4×4 reproduction

This is a separate protocol from G0-v2. It uses the public UPO 4×4 Sudoku test CSV and its published `SUDOKU_SYSTEM_PROMPT` / `formatting_sudoku` implementation. In particular, the public code passes `(formatted_prompt, formatted_answer)` as the chat message content; v3 preserves that tuple rendering and records it explicitly.

The first gate is baseline reproduction only. It reports blank-cell accuracy, exact puzzle accuracy, valid 1–4 predictions, and native scheduler agreement. No spatial symmetry manifest is frozen until this baseline is reproduced and audited.

The public test CSV is not silently regenerated. Supply the downloaded upstream file and verify its SHA256 against `LOCKED_CONFIG_V3.json`.

Example:

```bash
python src/run_published_4x4.py \
  --config LOCKED_CONFIG_V3.json \
  --dataset /path/to/UPO/dataset/4x4_test_sudoku.csv \
  --limit 4 --device cuda:0 --overwrite \
  --out results/v3_published_4x4_smoke.jsonl
```
