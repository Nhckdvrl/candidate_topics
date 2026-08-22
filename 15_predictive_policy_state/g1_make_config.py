#!/usr/bin/env python3
"""Derive the matched G1 training config from the released Light-WAM config.

The matched pair must differ in EXACTLY ONE thing: `model.loss.lambda_video`.
Everything that could otherwise carry a training-condition difference into the deployed
action path is closed here, per README "The decisive next experiment if G0 is positive":

- backbone LoRA disabled, so the WAM adapters are the only trainable representation module
  shared by the future loss and the action loss;
- pretrained video backbone frozen (already the released default);
- proprio encoder frozen at run time via LIGHTWAM_FREEZE_PROPRIO=1, because upstream
  `build_inputs` feeds the proprio token to BOTH branches;
- global gradient clipping made inactive, because upstream clips the norm over ALL trainable
  parameters and would otherwise let future-head gradients rescale action-expert gradients;
- one shared initialization checkpoint, loaded by both arms through upstream `resume`.

`lambda_video` itself is NOT written here; `g1_train.py` sets it per arm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--released-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--latent-cache-dir", type=str, required=True)
    p.add_argument("--text-cache-dir", type=str, required=True)
    p.add_argument("--dataset-stats", type=str, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.released_config.expanduser().resolve())

    # --- the single scientific manipulation lives in g1_train.py; everything below is matched ---

    # 1. WAM adapters must be the only trainable shared representation module.
    cfg.model.wam_adapter.use_backbone_lora = False
    cfg.model.wam_adapter.lora_layer_indices = []
    assert bool(cfg.model.wam_adapter.freeze_backbone) is True
    assert bool(cfg.model.wam_adapter.use_wam_adapter) is True
    assert bool(cfg.model.wam_adapter.remove_original_action_expert) is True
    assert list(cfg.model.state_fusion_action_expert_config.feature_sources) == ["adapted"]

    # 2. Global grad clipping must never activate (verified from logged grad norms).
    cfg.max_grad_norm = 1.0e9

    # 3. Matched optimization / data budget.
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.learning_rate = args.learning_rate
    cfg.lr_scheduler_type = "cosine"
    cfg.warmup_steps = args.warmup_steps
    cfg.warmup_ratio = 0.0
    cfg.max_steps = args.max_steps
    cfg.num_epochs = 10_000  # never binds; max_steps is the budget
    cfg.gradient_accumulation_steps = 1
    cfg.weight_decay = 0.01
    cfg.mixed_precision = "bf16"
    cfg.save_every = args.save_every
    cfg.eval_every = 0
    cfg.log_every = 25

    # 4. Identical, deterministic data.
    cfg.data.train.dataset_dirs = [args.dataset_dir]
    cfg.data.train.latent_cache_dir = args.latent_cache_dir
    cfg.data.train.text_embedding_cache_dir = args.text_cache_dir
    cfg.data.train.pretrained_norm_stats = args.dataset_stats
    cfg.data.train.use_latent_cache = True
    cfg.data.train.val_set_proportion = 0.0
    cfg.data.train.is_training_set = True

    # 5. Remove per-run noise sources that are irrelevant to the contrast.
    cfg.model.load_text_encoder = False
    cfg.wandb.enabled = False
    cfg.train_visualization.enabled = False
    cfg.parameter_report.enabled = True
    cfg.timing_breakdown.enabled = False
    if "benchmark" in cfg:
        cfg.benchmark.enabled = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output)
    print(f"saved matched config: {args.output}")
    print(f"  use_backbone_lora = {cfg.model.wam_adapter.use_backbone_lora}")
    print(f"  max_grad_norm     = {cfg.max_grad_norm}")
    print(f"  max_steps         = {cfg.max_steps}")
    print(f"  seed              = {cfg.seed}")


if __name__ == "__main__":
    main()
