#!/usr/bin/env python3
"""Build ONE common initialization checkpoint for the matched G1 pair.

Matched-mechanism requirement 1 (README "Identical initialization").

Upstream applies `cfg.seed` inside `Wan22Trainer.__init__`, i.e. AFTER
`instantiate(cfg.model, ...)` in `runtime.run_training`. Two separately launched runs are
therefore NOT guaranteed to start from identical randomly initialized WAM adapters and
state-fusion action expert merely because the CLI seed matches.

This script seeds Python / NumPy / torch BEFORE model instantiation, then writes one
checkpoint that both the future-on and future-off runs load through upstream
`resume=<path>.pt`, which loads weights only (no optimizer, no step).
"""
from __future__ import annotations

import argparse
import hashlib
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
    p.add_argument("--config", type=Path, required=True, help="training config the matched pair will use")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.lightwam_root.expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))
    os.chdir(root)

    # Seed BEFORE any model construction. This is the whole point of the script.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from hydra.utils import instantiate

    from lightwam.utils.config_compat import load_compatible_omegaconf
    from lightwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    cfg = load_compatible_omegaconf(str(args.config.expanduser().resolve()))
    cfg.model.load_text_encoder = False

    if bool(cfg.model.wam_adapter.use_backbone_lora) and os.environ.get("LIGHTWAM_LORA_ACTION_ONLY") != "1":
        raise ValueError(
            "The matched mechanism run requires `model.wam_adapter.use_backbone_lora=false` so that "
            "the WAM adapters are the only trainable representation module shared by the future and "
            "action objectives. Fix the config before building the shared initialization."
        )

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)

    fingerprint = {
        "seed": args.seed,
        "wam_adapters": module_hash(model.video_expert.wam_adapters),
        "state_fusion_action_expert": module_hash(model.state_fusion_action_expert),
        "proprio_encoder": module_hash(model.proprio_encoder),
        "video_head": module_hash(model.video_expert.head),
        "use_backbone_lora": bool(model.video_expert.use_backbone_lora),
        "adapter_layer_indices": [int(x) for x in model.video_expert.adapter_layer_indices],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(args.output), optimizer=None, step=0)
    (args.output.parent / "init_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )
    print(json.dumps(fingerprint, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
