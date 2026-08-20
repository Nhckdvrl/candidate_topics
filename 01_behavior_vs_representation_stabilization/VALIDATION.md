# G0 validation implementation

This directory contains a runnable **screening experiment**, not a full crosscoder paper implementation. The purpose of G0 is to decide whether behavioral stabilization and representational stabilization are measurably decoupled before spending compute on sparse feature alignment.

## What was verified against the seed work

### Behavioral side — Kishino et al., Findings ACL 2026

Paper: <https://aclanthology.org/2026.findings-acl.1163/>

The paper represents every checkpoint by its vector of sequence log-likelihoods on a common text set, double-centers the checkpoint × text matrix, and uses local squared Euclidean distance as a second-order approximation of KL. In nats,

```text
2 KL(p_i, p_j) ~= ||q_i - q_j||^2 / N
```

For the paper's byte-normalized scale, the pilot reports

```text
||q_i - q_j||^2 / (2 N mean_bytes ln 2)
```

in bits/byte.

The paper uses 10,000 Pile texts and dense Pythia checkpoints. G0 intentionally reduces this to 1,000 fixed texts and 13 checkpoints. This is a **screening approximation**, not an exact numerical reproduction of their table/figure.

Important code-availability note: as checked in August 2026, `shimo-lab/modelmap/kl-scale` exposes its README/figures but not the full experiment source. Therefore `src/metrics.py` and `src/extract_checkpoint.py` implement the published equations directly rather than pretending that an unavailable script can be reused.

### Representation side — Evolution of Concepts / Crosscoding Through Time

- Evolution of Concepts: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html>
- current SAE/crosscoder framework: <https://github.com/OpenMOSS/Llamascopium>
- Crosscoding Through Time: <https://github.com/bayazitdeniz/crosscoding-through-time>

The full papers use cross-checkpoint sparse features. G0 does **not** replace that with a claim about CKA. Linear CKA is only used as a cheap falsification screen. If no robust geometry-level separation exists, we stop; if it does, the next stage should use crosscoders/RelIE.

The Crosscoding Through Time repository already provides Pythia activation extraction, crosscoder training/evaluation, RelIE attribution and ablation scripts. It reports a 1×A100-80GB research environment, so that machinery is appropriate only after G0 succeeds.

## Files

```text
src/metrics.py               # double centering, KL proxy, linear CKA
src/extract_checkpoint.py    # one Pythia checkpoint -> LL + fixed hidden-state samples
src/analyze.py               # adjacent movement curves and CSV
run_pilot.sh                 # 13-checkpoint G0
tests/test_metrics.py        # pure numerical unit tests
```

## Exact G0 protocol

### Model / checkpoints

Default model:

```text
EleutherAI/pythia-410m
```

Revisions:

```text
step1000 step2000 step4000 step8000 step16000 step32000
step48000 step64000 step80000 step96000 step112000 step128000 step143000
```

### Texts

Default source is `NeelNanda/pile-10k`, sampled once with seed 42. Every checkpoint verifies that the same dataset row IDs and the same token positions were used.

After tokenizer truncation, the script recomputes the number of UTF-8 bytes actually seen by the model. This avoids normalizing a truncated likelihood by bytes that were never fed to the model.

### Hidden states

For each text, 8 deterministic quantile positions are sampled from the non-padding sequence. Four transformer-block depths are selected dynamically at approximately 25%, 50%, 75% and 100% depth. Hidden vectors are saved as float16; log-likelihoods remain float64.

### Primary measurements

For every adjacent pair of checkpoints:

1. behavior movement: local KL proxy in bits/byte;
2. representation movement per selected layer: `1 - linear_CKA`;
3. mean representation movement across the four selected depths.

No stabilization step is hard-coded. G0 first asks whether the curves visibly separate; change-point fitting should only be added after that premise survives.

## Run

```bash
cd 01_behavior_vs_representation_stabilization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# quick 100-example engineering smoke run
NUM_EXAMPLES=100 POSITIONS_PER_TEXT=4 ./run_pilot.sh

# preregistered G0
NUM_EXAMPLES=1000 POSITIONS_PER_TEXT=8 ./run_pilot.sh
```

Outputs:

```text
artifacts/checkpoints/step*.npz
artifacts/analysis/adjacent_metrics.csv
artifacts/analysis/behavior_vs_step.png
artifacts/analysis/representation_vs_step.png
artifacts/analysis/behavior_vs_representation.png
```

## G0 decision rule

Continue only if one of these is stable under bootstrap resampling of texts / positions:

- behavioral movement reaches a clear late-training floor while one or more layers keep moving;
- representation movement reaches a floor clearly earlier than behavior;
- there is a systematic layer-wise stabilization order.

Stop if KL and CKA simply decay together, or if the apparent gap disappears after changing text/position samples.

## If G0 succeeds

Do **not** write a CKA paper. Replace geometry-level movement with feature-level movement:

1. choose checkpoint triplets around the discovered transition;
2. cache activations using the public Crosscoding Through Time pipeline;
3. train crosscoders on those triplets;
4. quantify feature emergence / maintenance / discontinuation after `t_behavior`;
5. use RelIE/ablation to separate behaviorally silent feature turnover from causally relevant turnover;
6. replicate across Pythia scales and, ideally, multiple training seeds.

## Local verification performed before committing

- Python syntax/bytecode compilation: passed.
- `double_center` row/column-zero-mean test: passed.
- KL identity test: passed.
- linear CKA identity + orthogonal-invariance test: passed.
- synthetic end-to-end `analyze.py` run producing all three plots/CSV: passed.

The full Pythia GPU extraction was not executed in the ChatGPT sandbox because model weights/internet/GPU are not available there.
