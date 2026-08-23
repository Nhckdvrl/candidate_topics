# G0 Results — Topic 21

## Verdict

**STOP_UPSTREAM_SEED_NOT_REPRODUCED**. The mandatory official prerequisite completed, but its frozen semantic gates failed. Per protocol, paired G0 was not run.

## Frozen run

- Candidate commit: `91ce7a32c3ecd25a8524a968bd8352050a136706`.
- Official repository commit: `0f8b327097f2a34bbc8d1c603480982e65053384`.
- Host/GPU: `fvcrc10`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, four A100 80 GB GPUs.
- Torch: `2.6.0+cu124`; transformers: `4.51.3`; vLLM: `0.8.5`.
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, HF revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- Exact command:

  ```bash
  python -m long_context_understanding.experiments.fsyn_output_prediction \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --num-functions 80 --num-contexts 800 --position-step 8 --seed 42
  ```

- The official run completed 8,800 examples and wrote `results/fsyn_output_prediction/Qwen/Qwen2.5-Coder-7B-Instruct/80/42/summary.json`.

## Official prerequisite metrics

All 11 frozen positions were present: 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80.

- Position accuracies: `0: 0.000`, `8: 0.000`, `16: 0.000`, `24: 0.000`, `32: 0.000`, `40: 0.000`, `48: 0.000`, `56: 0.000`, `64: 0.000`, `72: 0.000`, `80: 0.00125`.
- Edge mean: `0.000625` — required `>= 0.30`: **FAIL**.
- Middle position: `40`, accuracy `0.000`.
- Edge-to-middle drop: `0.000625` — required `>= 0.20`: **FAIL**.
- At least three positions: **PASS**.
- Contract verdict: `UPSTREAM_SEED_NOT_REPRODUCED`.

The official contract artifact is `artifacts/g0_upstream_contract.json`. Because the prerequisite failed, no Topic 21 paired G0 artifact exists and no mechanism verdict is assigned. No model, prompt, context, parser, seed, or threshold rescue was attempted.
