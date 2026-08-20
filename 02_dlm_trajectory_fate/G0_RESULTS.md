# G0 Results — DLM Trajectory Fate

## Environment

- Host: `fvcrc20`
- GPUs: 4 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB each
- Driver: 580.82.07; CUDA runtime reported by PyTorch: 13.0
- Python: 3.12.3 (`/dev/shm/candidate_topics_venv`)
- PyTorch: 2.13.0+cu130
- transformers: 4.52.2; datasets: 2.21.0
- Model: `GSAI-ML/LLaDA-8B-Instruct`, revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`
- Repository commit: `490e0476cb65632c2af1b364141af900834fc312`
- Primary geometry: MidTruth GSM8K, 64 steps, 128 tokens, block length 32, temperature 0, strict parser, hidden tuple indices 24/25/28.

## Commands actually run

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b server/02-dlm-trajectory-fate-g0
for f in *.sh; do bash -n "$f"; done
python -m py_compile src/*.py
python -m pytest -q tests                 # 9 passed
PATH=/dev/shm/candidate_topics_venv/bin:$PATH HF_HOME=/dev/shm/xiang-hf \
  HUGGINGFACE_HUB_CACHE=/dev/shm/xiang-hf/hub NUM_EXAMPLES=20 GPUS="0 1 2 3" \
  ./run_surface_preflight_4gpu.sh
PATH=/dev/shm/candidate_topics_venv/bin:$PATH HF_HOME=/dev/shm/xiang-hf \
  HUGGINGFACE_HUB_CACHE=/dev/shm/xiang-hf/hub NUM_EXAMPLES=200 GPUS="0 1 2 3" \
  ./run_surface_preflight_4gpu.sh
PATH=/dev/shm/candidate_topics_venv/bin:$PATH HF_HOME=/dev/shm/xiang-hf \
  HUGGINGFACE_HUB_CACHE=/dev/shm/xiang-hf/hub NUM_EXAMPLES=1000 GPUS="0 1 2 3" \
  ./run_pilot_4gpu.sh
```

## Engineering smoke (20 examples)

All four shards completed and produced complete 64-state trajectories, NPZ files, and final-text JSONL files. Strict final-answer observed rate was 0.750 and final accuracy among observed answers was 0.667 (0.500 over all examples). The strict parser preserved unobserved states; no-answer-yet was not mapped to incorrect.

## G-1 surface census (200 examples)

- Strict final-answer observed rate: **0.905**
- Final accuracy over all examples: **0.635**
- Final accuracy among observed answers: **0.702**
- Maximum final-controlled novel-task minimum class support: **29** (gate: 10)
- Largest transient recovery support: step 16, 29 per class minimum
- Largest transient overwrite support: step 4, 29 per class minimum
- Status: **GO_HIDDEN_G0**

## Positive control: final correctness

The later hidden-state pipeline reproduced the expected final-correctness signal. Best fixed-step row was step 63, layer 25: AUC **0.8723**, step-0 AUC **0.7938**, delta **+0.0785**. Thus the MidTruth geometry passed the positive-control requirement; the novelty result is interpretable.

## Novel task results

All rows below use strict labels, identical absolute-step comparisons, surface controls, step-0 hidden controls, and 500 paired bootstrap resamples.

### transient_recovery

- Strong pre-transition row: step **16**, layer **25**, minimum lead **4** steps
- Class support: 41 positive / 109 negative (n=150 after lead filtering)
- Hidden AUC: **0.6762**, bootstrap 95% CI **[0.5844, 0.7581]**
- Surface AUC: **0.4594**; hidden minus surface: **+0.2168**
- Step-0 hidden AUC: **0.5250**; hidden minus step 0: **+0.1513**
- This is a meaningful pre-transition signal under the final-wrong control.

### transient_overwrite

- Strong pre-transition row: step **4**, layer **28**, minimum lead **16** steps
- Class support: 46 positive / 120 negative (n=166 after lead filtering)
- Hidden AUC: **0.7046**, bootstrap 95% CI **[0.6131, 0.7900]**
- Surface AUC: **0.4190**; hidden minus surface: **+0.2856**
- Step-0 hidden AUC: **0.6239**; hidden minus step 0: **+0.0807**
- This is a meaningful pre-transition signal under the final-correct control.

## Decision

**CONTINUE**

The G-1 support gate passed, the final-correctness positive control replicated, and both final-controlled novelty tasks contain strong pre-transition rows exceeding surface and step-0 baselines with bootstrap lower bounds above 0.55. This supports proceeding to the next preregistered validation stage; it does not justify changing geometry, parser, layer search, or labels after the fact.

## Artifacts

- `artifacts/preflight_midtruth/surface_summary.json`
- `artifacts/preflight_midtruth/surface_class_counts.csv`
- `artifacts/g0_midtruth/probes/task_class_counts.csv`
- `artifacts/g0_midtruth/probes/step_layer_auc.csv`
- `artifacts/g0_midtruth/probes/pretransition_auc.csv`
- `artifacts/g0_midtruth/probes/decision.json`

Raw hidden-state NPZ files remain local and are intentionally not committed.
