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

## Import shim audit

`tools/flash_attn_compat` is a deliberately explicit, pure-PyTorch import shim for `flash_attn.bert_padding`. It is only prepended to `PYTHONPATH` for the environment audit. With the official trainer configuration's sequence-parallel size fixed to 1, the Ulysses branch containing these helpers is not executed; the normal path uses the trainer's ordinary PyTorch loss. The shim is not a FlashAttention implementation and is not valid for sequence-parallel training.

## Next safe action

Run the official trainer import and a one-step dataloader/config smoke with the explicit shim, then inspect the actual branch and loss mask. Do not launch 10-epoch training until those checks pass.
