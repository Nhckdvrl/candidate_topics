# G1/v4 environment audit

## Completed

- Model: `Dream-org/Dream-v0-Instruct-7B`
- Shared HF cache: `/home/xiang/.cache/huggingface/hub`
- Snapshot: `05334cb9faaf763692dcf9d8737c642be2b2a6ae`
- Cache size: approximately 15 GB; all four safetensors shards are complete and no `.incomplete` files remain.
- Node GPUs: four RTX PRO 6000 Blackwell Max-Q devices with approximately 96 GB free each at audit time.
- Official Dream inference smoke: model loads under `transformers==4.46.2`; a simple arithmetic prompt produces a response.

## Exact smoke observation

The locked 9×9 seed-aligned prompt was passed through the official `AutoModel`/`AutoTokenizer` and `diffusion_generate` path. At checkpoint 0, the generated 256-token response was all EOS tokens. This is recorded as a zero-shot competence observation, not yet as a scientific result: the fine-tuned checkpoint is the intended object, and the prompt/data provenance is explicitly reconstructed rather than recovered from the seed authors.

## Official SFT dependency chain

The Dream repository's `src.trainer.fsdp_sft_trainer` imports successfully only after using `verl==0.5.0`; the current `verl==0.9.0` package does not contain the required `verl.trainer.fsdp_sft_trainer` module. The remaining hard import is `flash_attn.bert_padding`.

The available pip source distribution for `flash-attn` attempts to resolve a separate Torch 2.13/CUDA 13 stack. The node already has Torch 2.11.0+cu130, and that resolution would be an uncontrolled multi-hundred-megabyte-plus environment replacement. It was cancelled before installation. No custom trainer or scientific protocol substitution was made.

## Next safe action

Find or build a flash-attn wheel compatible with the existing Torch/CUDA ABI in an isolated temporary environment, then re-run only the official trainer import and a one-step dataloader/config smoke. Do not launch 10-epoch training until that import smoke passes.
