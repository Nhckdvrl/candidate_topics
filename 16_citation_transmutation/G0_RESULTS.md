# Topic 16 G0 results

## Verdict

`STOP_MEASUREMENT_NOT_RELIABLE`

The frozen gold measurement gate is not passed. Same-core precision, NONE
precision, and certainty agreement exceed their thresholds, but the prespecified
known Greenberg-style transmutation case is not recovered (2/4 exact; only 1/3
UP edges recovered). Therefore no formal primary G0 effect is reported and no
model, prompt, threshold, or subset tuning is attempted.

## Frozen instrument

- Model: `Qwen/Qwen3-32B` snapshot `9216db5781bf21249d130ec9da846c4624c16137`
- Served name: `Qwen3-32B-Instruct`; temperature 0; seed `20260823`
- vLLM 0.22.1, Python 3.12.13, PyTorch 2.11.0+cu130, Transformers 5.10.2
- Local workstation; 2 x NVIDIA RTX PRO 6000 Blackwell Max-Q; driver 580.82.07
- Tensor parallel 2; dtype auto (BF16 weights); max context 32768; vLLM generation config
- Qwen thinking disabled through `chat_template_kwargs` solely to make the required JSON output observable; this is serving plumbing and does not change the scientific labels or estimand.

## Gold audit

- Gold edges: 68
- Same-core precision: 51 TP, 3 FP, **0.9444**
- Direct-support precision: 33 TP, 1 FP, **0.9706**
- `NONE` precision: 36 TP, 0 FP, **1.0000**
- Primary-eligibility precision: 14 TP, 0 FP, **1.0000**
- Certainty exact agreement including abstentions: **0.9231** (52 determinate gold cases); conditional agreement 1.0000; model coverage 0.9231
- Known Greenberg case: 4 edges; exact 2/4; 1/3 gold UP edges recovered; **recovered=false**

Because the known-case requirement fails, `raw_candidates.jsonl` was not promoted
to a formal `raw_edges.jsonl`, no human NONE audit was used to manufacture a
primary subset, and no `g0_results.json` effect estimate is claimed.

## Reproducibility

`raw_gold_edges.jsonl`, `measured_gold_edges.jsonl`,
`gold_audit_report.json`, model metadata, and runner logs are committed. The
annotation runner records raw model responses in each measured row.
