# Dream SFT environment compatibility notes

The Dream training source imports an older veRL layout. For the checked
environment audit, `verl==0.3.0.post1` is used because it both provides
`verl.trainer.fsdp_sft_trainer` and re-exports
`FSDPUlyssesShardingManager` from `verl.workers.sharding_manager`.

That veRL package sees the node's newer vLLM installation and eagerly imports
its optional vLLM sharding manager, which is incompatible with Dream's pinned
`transformers==4.46.2`. The audit therefore disables only that optional import;
ordinary FSDP SFT does not use vLLM. The exact one-line patch is recorded in
`verl_sharding_manager_vllm_optional.patch`.
