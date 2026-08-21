# Server Runbook

This is the handoff sequence for a GPU-side agent. Do not change the measurement after seeing results.

## A. Environment

Recommended: fresh Python 3.10–3.12 CUDA environment on one NVIDIA node.

```bash
git pull
cd 07_memory_interference_architecture
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

FLA's current package layout requires the full CUDA extra. If the machine already has a compatible PyTorch/CUDA stack and dependency resolution tries to replace it incorrectly, follow upstream `flash-linear-attention/INSTALL.md` rather than installing random older FLA versions.

## B. Engineering validation only

```bash
python scripts/download_data.py
python -m pytest -q
python -m memory_interference.preflight --config configs/pilot.yaml
```

Expected:

- data Git blob is verified;
- all unit tests pass;
- all four checkpoint configs/tokenizers load;
- shared tokenizer fingerprint check passes;
- sampled prompt lengths are safely below context limits.

If any item fails, stop. This is an INVALID setup, not a hypothesis result.

## C. One-model inference smoke test

```bash
./run_smoke.sh
```

Inspect:

```bash
cat outputs/smoke/summary.csv
cat outputs/smoke/token_audit.json
```

This test is only to verify weight loading and score extraction. Do not draw scientific conclusions from it.

## D. Frozen discovery pilot

```bash
./run_pilot.sh
```

The script executes:

1. dataset checksum;
2. unit tests;
3. preflight;
4. sequential inference for the four matched models;
5. analysis;
6. paired bootstrap decision.

Primary outputs:

```bash
cat outputs/architecture_pi_ri_pilot/summary.csv
cat outputs/architecture_pi_ri_pilot/pairwise_bootstrap.json
cat outputs/architecture_pi_ri_pilot/token_audit.json
cat outputs/architecture_pi_ri_pilot/decision.json
```

## E. Decision handling

### `PARADIGM_FAIL`

Stop. The Transformer baseline did not reproduce PI > RI under this measurement. Do not compare linear models mechanistically.

### `KILL`

Stop and archive the architecture-separation hypothesis. Do not add models/prompts/metrics to rescue it.

### `INCONCLUSIVE_DO_NOT_TUNE`

Preserve outputs. Do not tune against the result. Escalate only by deciding, before another run, whether the topic is still worth an independently registered measurement.

### `GO_TO_LOCKED_CONFIRMATION` or `STRONG_GO`

Run exactly:

```bash
./run_confirmation.sh
```

No config edits are authorized between discovery and confirmation.

## F. Confirmation

Inspect:

```bash
cat outputs/architecture_pi_ri_confirmation/decision.json
```

A positive paper-direction signal requires the same primary architecture gap (`>= 0.10`, bootstrap lower bound > 0) on the independent seed.

If confirmation fails, do not pool discovery and confirmation to manufacture significance.

## G. GPU usage

The runner intentionally loads models sequentially, so one GPU with enough memory is sufficient. On a multi-GPU node, the cheapest clean workflow is often to run separate model filters concurrently, one model per GPU, **only if each process writes to a distinct run directory**. The default script writes all models into one JSONL and therefore should not be launched concurrently against the same output directory.

For the safest first execution, run sequentially exactly as written.

## H. What to report back

Return these files, not just a verbal summary:

```text
resolved_config.json
summary.csv
pairwise_bootstrap.json
token_audit.json
intrusions.json
decision.json
```

If a run crashes, also report the exact model, update level, exception, CUDA/PyTorch/FLA versions, and whether preflight passed. Do not edit the scientific config as a debugging shortcut.
