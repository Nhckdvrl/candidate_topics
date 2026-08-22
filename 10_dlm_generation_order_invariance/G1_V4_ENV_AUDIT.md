# G1/v4 environment audit

## Completed

- Model: `Dream-org/Dream-v0-Instruct-7B`
- Shared HF cache: `/home/xiang/.cache/huggingface/hub`
- Snapshot: `05334cb9faaf763692dcf9d8737c642be2b2a6ae`
- Cache size: approximately 15 GB; all four safetensors shards are complete and no `.incomplete` files remain.
- Node GPUs: four RTX PRO 6000 Blackwell Max-Q devices with approximately 96 GB free each at audit time.
- Official Dream inference smoke: model loads under `transformers==4.46.2`; a simple arithmetic prompt produces a response.
- The shared cache contains Instruct-7B but no Base-7B snapshot. Seed-paper Base/Instruct provenance is unresolved and is locked as such in `LOCKED_CONFIG_V4.json`.
- Dream tokenizer audit: every frozen train/test prompt is 283 tokens, every gold response is 172 tokens including EOS, and prompt plus gold is 455 tokens.

## Exact smoke observation

The locked 9×9 seed-aligned prompt was passed through the official `AutoModel`/`AutoTokenizer` and `diffusion_generate` path. At checkpoint 0, the generated 256-token response was all EOS tokens. This is recorded as a zero-shot competence observation, not yet as a scientific result: the fine-tuned checkpoint is the intended object, and the prompt/data provenance is explicitly reconstructed rather than recovered from the seed authors. The follow-up engineering audit uses the data-derived 172-token response length.

## Official SFT dependency chain

The Dream repository's `src.trainer.fsdp_sft_trainer` requires an older veRL layout. `verl==0.3.0.post1` provides both the required `verl.trainer.fsdp_sft_trainer` module and the expected `FSDPUlyssesShardingManager` export; `verl==0.9.0` does not. The node's newer vLLM is an optional import conflict with Dream's pinned transformers and is disabled only for the ordinary FSDP audit; see `tools/verl_compat/`.

The available pip source distribution for `flash-attn` attempts to resolve a separate Torch 2.13/CUDA 13 stack. The node already has Torch 2.11.0+cu130, and that resolution would be an uncontrolled multi-hundred-megabyte-plus environment replacement. It was cancelled before installation. No custom trainer or scientific protocol substitution was made.

With the explicit `tools/flash_attn_compat` import shim, the official trainer imports successfully. The official `random_with_input_pad` collator on two frozen train rows produces 454-token batches: 283 prompt-mask zeros and 171 non-pad response-mask ones. The 172nd gold token is EOS, which shares the pad ID and is intentionally removed by the upstream non-pad filter.

The first distributed step attempt then showed that Dream's released trainer explicitly requested `attn_implementation="flash_attention_2"` while loading the model. Since Dream's own model implementation supports SDPA and the official README documents SDPA, the isolated audit copy applies the one-line `tools/verl_compat/dream_trainer_sdpa.patch` to select `attn_implementation="sdpa"`. This changes only the attention backend for compatibility; it does not change the trainer loss, data, masking, optimizer, or scientific protocol.

With that isolated patch, a four-GPU one-step run completed forward/backward, validation, and save: `train/loss=1.408`, `val/loss=0.349`. The saved `global_step_1` checkpoint was reloaded successfully with Dream's tokenizer and SDPA model loader after supplying the upstream remote-code files that the trainer stores at the run root. The smoke output is outside the repository under `/tmp/candidate_topics/dream_sft_smoke` and is not a scientific result.

The first two-epoch formal start exposed a validation-only `0/0` edge case in upstream `q_sample`: a random validation batch can contain no masked response token. The isolated trainer copy now applies `tools/verl_compat/dream_zero_mask_guard.patch`, returning a finite zero diagnostic for that batch while leaving all nonempty training/validation losses unchanged. This guard does not affect exact-grid evaluation or the scientific protocol.

The formal `global_step_24` checkpoint has the same packaging quirk: the trainer
writes the remote-code Python files at the run root rather than inside each
checkpoint directory. Before inference/reload, the four upstream remote-code files
are copied into the checkpoint directory. This changes no weights or evaluation
behavior and is recorded as checkpoint packaging handling.

Resume inspection found a boundary bug in the released trainer: its saved epoch is
zero-based, but an exact-boundary resume would repeat the just-completed epoch.
`tools/verl_compat/dream_resume_epoch_boundary.patch` advances the in-memory epoch
only when `global_step % steps_per_epoch == 0`; mid-epoch resume behavior is kept.
This is required before the planned 2-to-5 segment and will be verified by a short
resume smoke.

## Import shim audit

`tools/flash_attn_compat` is a deliberately explicit, pure-PyTorch import shim for `flash_attn.bert_padding`. It is only prepended to `PYTHONPATH` for the environment audit. With the official trainer configuration's sequence-parallel size fixed to 1, the Ulysses branch containing these helpers is not executed; the normal path uses the trainer's ordinary PyTorch loss. The shim is not a FlashAttention implementation and is not valid for sequence-parallel training.

## Formal-run memory audit

The first guarded epoch-0-to-2 formal attempt failed at the first `AdamW.step()`
while lazily allocating optimizer state: GPU 2 had 107 MiB free and the allocation
requested 224 MiB. This is an engineering capacity failure, not a competence result;
it produced no scientific checkpoint or test prediction. The next attempt must use a
memory-safe FSDP/optimizer configuration while keeping the model, data, loss,
learning rate, and evaluation protocol fixed.

An isolated retry with the official trainer's `model.fsdp_config.cpu_offload=true`
and `offload_params=true` reached `AdamW.step()` without the GPU allocation failure;
it was stopped after about 90 seconds at the first step because this path is
substantially slower, not because of an error. CPU offload is therefore the current
memory-safe formal-run configuration.

## Next safe action

Run the official trainer import and a one-step dataloader/config smoke with the explicit shim, then inspect the actual branch and loss mask. Do not launch 10-epoch training until those checks pass.
