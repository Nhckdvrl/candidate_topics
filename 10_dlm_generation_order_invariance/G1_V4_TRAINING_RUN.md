# G1/v4 formal training run

This run starts only after the model/cache, tokenizer length, official trainer import, collator mask, one-step FSDP forward/backward, save, and reload audits passed.

Fixed choices:

- model: cached `Dream-org/Dream-v0-Instruct-7B`;
- data: the committed 50-row train parquet and untouched 100-row test parquet;
- prompt+gold coverage: `max_length=512` (455 observed tokens plus margin);
- official collator: `perbatch_cutoff=True`, `random_with_input_pad`;
- four GPUs, micro-batch 1 per GPU, global train batch 4; formal execution uses
  FSDP `cpu_offload=true` and `offload_params=true` after the GPU-only Adam state
  initialization exceeded device memory (engineering-only change);
- official Dream SFT learning rate `2e-6`, no LoRA, gradient checkpointing enabled;
- no evaluation-driven changes to prompt, data, loss, decoding length, or exact metric.

The 10-epoch run is segmented only to retain checkpoints at epochs 2, 5, and 10 without saving every intermediate epoch. The segment boundaries are operational, not selected from test accuracy. Epoch 0 is the untouched cached model and is recorded separately.

No 9×9 symmetry manifest or symmetry trace may be generated before the ordinary exact-grid competence curve is complete.
