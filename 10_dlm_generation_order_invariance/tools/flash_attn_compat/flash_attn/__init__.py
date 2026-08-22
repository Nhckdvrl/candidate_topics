"""Import-only compatibility namespace for Dream's non-Ulysses FSDP audit.

This is not a FlashAttention implementation. It exists only because verl 0.5
imports flash_attn.bert_padding even when sequence parallelism is disabled.
"""
