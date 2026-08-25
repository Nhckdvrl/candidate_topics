# Topic 25 — G0 Results

## Final decision

`SEED_RELATION_NOT_REPRODUCED`

Per the frozen protocol, Topic 25 stops at the mandatory reproduction receipt. G0 was not run.

## Environment

- candidate_topics commit: `edcaf7fe574a9befeddb2afbb1dc65e45e7aca8b`
- upstream Weakest-Link commit: `9b01abaad354208a6a8fb26c58eb5c330036fb94`
- execution host: `fvcrc20`
- model: `Qwen/Qwen3-8B`
- bank: processed MuSiQue 18-doc bank, seed 42, `bank_n=1246`
- vLLM endpoint: local OpenAI-compatible server on `fvcrc21`, TP=4, max model length 32768
- sampling: temperature `0.0`, top_p `1.0`, seed `42`

## Receipt

All expected receipt files were complete and used the exact expected IDs, prompt IDs, thinking flags, and boolean `correct` fields.

| Cell | EM |
|---|---:|
| Qwen3-8B gold-only | 0.41493 |
| Qwen3-8B-Think gold-only | 0.45746 |
| Qwen3-8B noisy Spread pooled | 0.31182 |
| Qwen3-8B-Think noisy Spread pooled | 0.35179 |

By bucket, noisy Spread EM was:

- no-think: beginning `0.32472`, midsection `0.30803`, tail `0.30273`
- think: beginning `0.37592`, midsection `0.34286`, tail `0.33660`

Receipt relations:

- thinking gold-only >= no-think gold-only: `true`
- thinking noisy pooled >= thinking gold-only: `false`
- thinking noisy pooled > no-think noisy pooled: `true`

The failed relation is the frozen seed requirement that noisy thinking performance reach at least the thinking gold-only baseline. Therefore the receipt verdict is `SEED_RELATION_NOT_REPRODUCED`.

## G0

`NOT RUN` by design. No eligible-panel selection, model calls, G0 scoring, bootstrap, or G0 verdict was produced.

## Integrity

- frozen settings changed: **NO**
- outcome-based filtering: **NO**
- favorable-subset rerun: **NO**
- G0 rescue/model/prompt/sampling changes: **NO**

## Engineering anomalies

The first receipt attempt had invalid 404 results because the vLLM server exposed only `Qwen3-8B` while the upstream client requested `Qwen/Qwen3-8B`; those invalid files were excluded from scoring. A later exact rerun was interrupted by intermittent hostname resolution for `fvcrc21`; the server was subsequently addressed by its confirmed internal IP and all receipt files were regenerated and rechecked. The final receipt above has `complete_support=true`; the scientific stop is therefore based on the frozen relation failure, not incomplete calls.

