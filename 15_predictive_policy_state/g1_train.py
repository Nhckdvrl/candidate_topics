#!/usr/bin/env python3
"""Launch ONE arm of the matched G1 pair.

Usage (under `accelerate launch`):

    g1_train.py --lightwam-root ... --config matched_config.yaml \
                --init-checkpoint init.pt --output-dir ... --lambda-video {1.0|0.0}

`--lambda-video` is the ONLY thing that differs between the two arms. Everything else comes
from the shared matched config and the shared initialization checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lightwam-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lambda-video", type=float, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.lightwam_root.expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))
    # Matched-mechanism requirement 3: close the shared proprio route in BOTH arms.
    os.environ["LIGHTWAM_FREEZE_PROPRIO"] = "1"
    os.environ.setdefault("FASTWAM_STRICT_DATASET_ERRORS", "1")
    os.chdir(root)

    from omegaconf import OmegaConf

    from lightwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    cfg = OmegaConf.load(args.config.expanduser().resolve())

    # ---- the single manipulated variable ----
    cfg.model.loss.lambda_video = float(args.lambda_video)
    # -----------------------------------------

    cfg.output_dir = str(args.output_dir.expanduser().resolve())
    cfg.resume = str(args.init_checkpoint.expanduser().resolve())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    lora_action_only = os.environ.get("LIGHTWAM_LORA_ACTION_ONLY", "0") == "1"
    if bool(cfg.model.wam_adapter.use_backbone_lora) and not lora_action_only:
        raise ValueError(
            "matched run requires use_backbone_lora=false, unless LIGHTWAM_LORA_ACTION_ONLY=1 "
            "restricts LoRA to action-loss gradient (capacity-restored arm)"
        )
    if lora_action_only and not bool(cfg.model.wam_adapter.use_backbone_lora):
        raise ValueError("LIGHTWAM_LORA_ACTION_ONLY=1 is meaningless without backbone LoRA")
    if float(cfg.max_grad_norm) < 1.0e6:
        raise ValueError("matched run requires an inactive global clipping threshold")

    # Seed everything BEFORE model instantiation as well; the shared init checkpoint is the
    # authoritative source of initial weights, this only makes any residual RNG use matched.
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from lightwam.runtime import run_training

    manifest = {
        "lambda_video": float(cfg.model.loss.lambda_video),
        "lambda_action": float(cfg.model.loss.lambda_action),
        "use_backbone_lora": bool(cfg.model.wam_adapter.use_backbone_lora),
        "lora_action_gradient_only": lora_action_only,
        "freeze_backbone": bool(cfg.model.wam_adapter.freeze_backbone),
        "freeze_proprio_encoder": os.environ["LIGHTWAM_FREEZE_PROPRIO"] == "1",
        "max_grad_norm": float(cfg.max_grad_norm),
        "seed": seed,
        "max_steps": int(cfg.max_steps),
        "batch_size": int(cfg.batch_size),
        "learning_rate": float(cfg.learning_rate),
        "init_checkpoint": str(cfg.resume),
    }
    if int(os.environ.get("RANK", "0")) == 0:
        (Path(cfg.output_dir) / "g1_arm_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("[g1] arm manifest:", json.dumps(manifest), flush=True)

    run_training(cfg)


if __name__ == "__main__":
    main()
