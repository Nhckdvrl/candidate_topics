#!/usr/bin/env python3
"""G0 for Topic 15: does training-time world modeling act through a predictive policy state?

This script is deliberately narrow. On a released Light-WAM checkpoint it asks only:

1) Does the exact state read by the deployed action expert contain more held-out
   future-latent information after the native WAM adapter than immediately before it?
2) If the action expert is fed the corresponding pre-adapter state instead, does
   offline action error increase on the same held-out episodes?

No SAE/PCA/CCA/subspace search is used, and the causal intervention does not use
probe directions. The intervention is architecture-native: adapted -> backbone
at the action readout, using tensors cached by Light-WAM itself in the same pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from g0_core import (
    future_change_target,
    linear_ridge_probe,
    paired_bootstrap_mean_ci,
    relative_change,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lightwam-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="Saved Light-WAM training config.yaml. If omitted, search checkpoint parents.",
    )
    p.add_argument(
        "--dataset-stats",
        type=Path,
        default=None,
        help="dataset_stats.json. If omitted, search checkpoint parents.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        default=None,
        help="Override data.train.dataset_dirs. Repeat for multiple directories.",
    )
    p.add_argument("--latent-cache-dir", type=Path, default=None)
    p.add_argument("--text-cache-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("./g0_results"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument(
        "--probe-ridge",
        type=float,
        default=1e-2,
        help="Single frozen ridge value; this script never searches over it.",
    )
    p.add_argument("--target-chunk-size", type=int, default=4096)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument(
        "--min-relative-effect",
        type=float,
        default=0.05,
        help="Pilot continuation floor, not a publication claim threshold.",
    )
    p.add_argument(
        "--save-tensors",
        action="store_true",
        help="Save pooled features/targets for debugging. Off by default because targets are large.",
    )
    return p.parse_args()


def _resolve_near_checkpoint(
    checkpoint: Path,
    explicit: Path | None,
    filename: str,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = []
    checkpoint = checkpoint.expanduser().resolve()
    for parent in [checkpoint.parent, *list(checkpoint.parents)[:5]]:
        candidates.append(parent / filename)
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not locate {filename} near checkpoint {checkpoint}. Pass it explicitly."
    )


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _prepare_lightwam_import(root: Path):
    root = root.expanduser().resolve()
    if not (root / "src" / "lightwam").exists():
        raise FileNotFoundError(f"Not a Light-WAM checkout: {root}")
    for path in (root, root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root


def _load_cfg(args: argparse.Namespace, lightwam_root: Path):
    from lightwam.utils.config_compat import load_compatible_omegaconf

    training_config = _resolve_near_checkpoint(
        args.checkpoint, args.training_config, "config.yaml"
    )
    stats_path = _resolve_near_checkpoint(
        args.checkpoint, args.dataset_stats, "dataset_stats.json"
    )
    cfg = load_compatible_omegaconf(str(training_config))

    cfg.data.train.use_latent_cache = True
    cfg.data.train.pretrained_norm_stats = str(stats_path)
    cfg.data.train.val_set_proportion = 0.0
    cfg.data.train.is_training_set = True
    if cfg.model.get("load_text_encoder") is not None:
        cfg.model.load_text_encoder = False

    if args.dataset_dir:
        cfg.data.train.dataset_dirs = [str(p.expanduser().resolve()) for p in args.dataset_dir]
    if args.latent_cache_dir is not None:
        cfg.data.train.latent_cache_dir = str(args.latent_cache_dir.expanduser().resolve())
    if args.text_cache_dir is not None:
        cfg.data.train.text_embedding_cache_dir = str(args.text_cache_dir.expanduser().resolve())

    for key in ("dataset_dirs", "latent_cache_dir", "text_embedding_cache_dir"):
        if cfg.data.train.get(key) in (None, "", []):
            raise ValueError(f"data.train.{key} is unresolved; pass the corresponding CLI override")

    os.chdir(lightwam_root)
    return cfg, training_config, stats_path


def _instantiate_model_and_dataset(cfg, args: argparse.Namespace):
    from hydra.utils import instantiate

    model = instantiate(
        cfg.model,
        model_dtype=_dtype(args.dtype),
        device=args.device,
    )
    model.load_checkpoint(str(args.checkpoint.expanduser().resolve()))
    model.eval()

    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.to("cpu")
    if getattr(model, "vae", None) is not None:
        model.vae.to("cpu")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    dataset = instantiate(cfg.data.train)
    return model, dataset


def _architecture_audit(model) -> dict[str, Any]:
    problems = []
    if not hasattr(model, "uses_state_fusion_action_expert") or not model.uses_state_fusion_action_expert():
        problems.append("checkpoint is not using Light-WAM state-fusion action mode")
    expert = getattr(model, "state_fusion_action_expert", None)
    video_expert = getattr(model, "video_expert", None)
    if expert is None:
        problems.append("state_fusion_action_expert is missing")
    if video_expert is None:
        problems.append("video_expert is missing")

    layer_sources = []
    adapter_layers = []
    if expert is not None:
        layer_sources = [list(x) for x in getattr(expert, "layer_feature_sources", ())]
        if not layer_sources:
            problems.append("action expert exposes no layer_feature_sources")
        elif any(tuple(x) != ("adapted",) for x in layer_sources):
            problems.append(
                "G0 requires the released adapted-only action readout; current feature sources are "
                + repr(layer_sources)
            )
        for method in ("_pool_source_tokens", "forward"):
            if not hasattr(expert, method):
                problems.append(f"state_fusion_action_expert missing {method}")
    if video_expert is not None:
        adapter_layers = [int(x) for x in getattr(video_expert, "adapter_layer_indices", ())]
        if not adapter_layers:
            problems.append("video expert has no WAM adapter layers")
        if not hasattr(video_expert, "get_wam_action_fusion_layer_states"):
            problems.append("video expert does not expose native fusion layer states")
    if not hasattr(model, "_build_multilayer_action_fusion_inputs"):
        problems.append("model missing _build_multilayer_action_fusion_inputs")
    if not hasattr(model, "_build_action_observation_video_pre"):
        problems.append("model missing _build_action_observation_video_pre")

    if problems:
        raise RuntimeError("Architecture audit failed:\n- " + "\n- ".join(problems))

    return {
        "adapter_layers": adapter_layers,
        "layer_feature_sources": layer_sources,
        "token_pooling_type": str(getattr(expert, "token_pooling_type", "unknown")),
        "video_hidden_dim": int(getattr(expert, "video_hidden_dim", -1)),
        "use_backbone_lora": bool(getattr(video_expert, "use_backbone_lora", False)),
        "lora_layers": [int(x) for x in getattr(video_expert, "lora_layer_indices", ())],
    }


def _clean_episode_sample_indices(
    dataset,
    *,
    num_samples: int,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if num_samples < 16:
        raise ValueError("num_samples should be at least 16 for a meaningful held-out pilot")
    if not (0.1 <= test_fraction <= 0.5):
        raise ValueError("test_fraction must be in [0.1, 0.5]")
    if not hasattr(dataset, "get_num_episodes") or not hasattr(dataset, "get_episode_sample_range"):
        raise RuntimeError("Light-WAM dataset lacks episode-level indexing required for leakage-free split")

    rng = random.Random(seed)
    episode_ids = list(range(int(dataset.get_num_episodes())))
    rng.shuffle(episode_ids)

    n_test = max(4, int(round(num_samples * test_fraction)))
    n_train = num_samples - n_test
    chosen: list[tuple[int, int]] = []
    num_frames = int(dataset.num_frames)
    stride = int(dataset.lerobot_dataset.global_sample_stride)

    for episode_id in episode_ids:
        start, end = dataset.get_episode_sample_range(episode_id)
        latest_start = int(end) - 1 - (num_frames - 1) * stride
        if latest_start < int(start):
            continue
        sample_idx = rng.randint(int(start), int(latest_start))
        chosen.append((int(episode_id), int(sample_idx)))
        if len(chosen) >= num_samples:
            break

    if len(chosen) < num_samples:
        raise RuntimeError(
            f"Only found {len(chosen)} episodes with a full {num_frames}-frame window; need {num_samples}. "
            "Reduce --num-samples rather than reusing overlapping windows."
        )

    train_pairs = chosen[:n_train]
    test_pairs = chosen[n_train:n_train + n_test]
    train_eps, train_idx = zip(*train_pairs)
    test_eps, test_idx = zip(*test_pairs)
    assert set(train_eps).isdisjoint(set(test_eps))
    return list(train_idx), list(test_idx), list(train_eps), list(test_eps)


def _is_clean_sample(sample: dict[str, Any]) -> bool:
    for key in ("image_is_pad", "action_is_pad", "proprio_is_pad"):
        value = sample.get(key)
        if value is None:
            return False
        value = torch.as_tensor(value, dtype=torch.bool)
        if bool(value.any().item()):
            return False
    return True


def _pool_action_readout(expert, layer_states, tensor_key: str) -> torch.Tensor:
    pooled = []
    for pos, state in enumerate(layer_states):
        if tuple(expert.layer_feature_sources[pos]) != ("adapted",):
            raise RuntimeError("Unexpected action feature source during extraction")
        tokens = state[tensor_key]
        pooled.append(expert._pool_source_tokens(pos, "adapted", tokens))
    return torch.cat(pooled, dim=-1)


def _run_native_batch(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    inputs = model.build_inputs(batch)
    latents = inputs["input_latents"]
    observation_latents = inputs["first_frame_latents"]
    if observation_latents is None:
        observation_latents = latents[:, :, 0:1]
    if observation_latents.shape[2] != 1:
        raise ValueError(f"Expected one observation latent frame, got {tuple(observation_latents.shape)}")

    timestep = torch.zeros(
        observation_latents.shape[0],
        dtype=observation_latents.dtype,
        device=observation_latents.device,
    )
    video_pre = model._build_action_observation_video_pre(
        observation_latents=observation_latents,
        timestep_video=timestep,
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
    )
    _ = model.video_expert.forward_backbone(video_pre)
    layer_states = model._build_multilayer_action_fusion_inputs()
    expert = model.state_fusion_action_expert
    horizon = int(inputs["action"].shape[1])

    pred_adapted = expert(layer_states, action_horizon=horizon)
    backbone_readout_states = []
    for state in layer_states:
        replaced = dict(state)
        replaced["adapted"] = state["backbone"]
        backbone_readout_states.append(replaced)
    pred_backbone = expert(backbone_readout_states, action_horizon=horizon)

    target_action = inputs["action"]
    action_is_pad = inputs["action_is_pad"]
    loss_adapted = model._compute_action_loss_per_sample(
        pred_action=pred_adapted,
        target_action=target_action,
        action_is_pad=action_is_pad,
    )
    loss_backbone = model._compute_action_loss_per_sample(
        pred_action=pred_backbone,
        target_action=target_action,
        action_is_pad=action_is_pad,
    )
    action_shift = torch.sqrt(
        ((pred_backbone.float() - pred_adapted.float()) ** 2).mean(dim=(1, 2))
    )

    feature_adapted = _pool_action_readout(expert, layer_states, "adapted")
    feature_backbone = _pool_action_readout(expert, layer_states, "backbone")
    target_future = future_change_target(latents)

    return {
        "feature_adapted": feature_adapted.detach().float().cpu(),
        "feature_backbone": feature_backbone.detach().float().cpu(),
        "target_future": target_future,
        "loss_adapted": loss_adapted.detach().float().cpu(),
        "loss_backbone": loss_backbone.detach().float().cpu(),
        "action_shift": action_shift.detach().float().cpu(),
    }


def _collect(model, dataset, indices: list[int], batch_size: int) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {}
    batch_samples = []
    accepted = 0

    def flush():
        nonlocal batch_samples, accepted
        if not batch_samples:
            return
        batch = default_collate(batch_samples)
        with torch.inference_mode():
            result = _run_native_batch(model, batch)
        for key, value in result.items():
            outputs.setdefault(key, []).append(value)
        accepted += len(batch_samples)
        print(f"[extract] {accepted}/{len(indices)}", flush=True)
        batch_samples = []

    for idx in indices:
        sample = dataset[int(idx)]
        if not _is_clean_sample(sample):
            raise RuntimeError(f"Unexpected padding in supposedly clean sample idx={idx}")
        batch_samples.append(sample)
        if len(batch_samples) >= batch_size:
            flush()
    flush()

    merged = {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}
    for key, value in merged.items():
        if value.shape[0] != len(indices):
            raise RuntimeError(f"Collected {value.shape[0]} rows for {key}, expected {len(indices)}")
    return merged


def _mean_ci(values: torch.Tensor, seed: int, n_boot: int) -> dict[str, float]:
    mean, lo, hi = paired_bootstrap_mean_ci(values, seed=seed, n_boot=n_boot)
    return {"mean": mean, "ci95_low": lo, "ci95_high": hi}


def _analyse(
    train: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    probe_adapted = linear_ridge_probe(
        train["feature_adapted"],
        train["target_future"],
        test["feature_adapted"],
        test["target_future"],
        ridge=args.probe_ridge,
        target_chunk_size=args.target_chunk_size,
    )
    probe_backbone = linear_ridge_probe(
        train["feature_backbone"],
        train["target_future"],
        test["feature_backbone"],
        test["target_future"],
        ridge=args.probe_ridge,
        target_chunk_size=args.target_chunk_size,
    )

    future_mse_gain = probe_backbone.per_sample_mse - probe_adapted.per_sample_mse
    future_ci = _mean_ci(future_mse_gain, args.seed + 101, args.bootstrap)
    future_rel_gain = (
        (probe_backbone.mse - probe_adapted.mse) / probe_backbone.mse
        if probe_backbone.mse > 0 else float("nan")
    )

    action_delta = test["loss_backbone"] - test["loss_adapted"]
    action_ci = _mean_ci(action_delta, args.seed + 202, args.bootstrap)
    action_rel_increase = relative_change(
        float(test["loss_backbone"].mean().item()),
        float(test["loss_adapted"].mean().item()),
    )

    future_present = (
        math.isfinite(probe_adapted.r2)
        and probe_adapted.r2 > 0.0
        and future_rel_gain >= args.min_relative_effect
        and future_ci["ci95_low"] > 0.0
    )
    action_used = (
        action_rel_increase >= args.min_relative_effect
        and action_ci["ci95_low"] > 0.0
    )

    if future_present and action_used:
        verdict = "PROMISING_NATIVE_MEDIATOR"
    elif future_present and not action_used:
        verdict = "PREDICTIVE_BUT_NOT_ACTION_USED"
    elif (not future_present) and action_used:
        verdict = "ACTION_RELEVANT_BUT_NOT_PREDICTIVE"
    else:
        verdict = "NO_CLEAN_NATIVE_SIGNAL"

    metrics = {
        "future_probe": {
            "adapted": {
                "r2": probe_adapted.r2,
                "mse": probe_adapted.mse,
                "mean_target_baseline_mse": probe_adapted.baseline_mse,
            },
            "backbone": {
                "r2": probe_backbone.r2,
                "mse": probe_backbone.mse,
                "mean_target_baseline_mse": probe_backbone.baseline_mse,
            },
            "adapted_relative_mse_gain_vs_backbone": future_rel_gain,
            "paired_mse_gain": future_ci,
            "passes_clean_signal_bar": bool(future_present),
        },
        "causal_action_readout": {
            "adapted_action_loss_mean": float(test["loss_adapted"].mean().item()),
            "backbone_intervention_action_loss_mean": float(test["loss_backbone"].mean().item()),
            "relative_loss_increase": action_rel_increase,
            "paired_loss_increase": action_ci,
            "action_rms_shift_mean": float(test["action_shift"].mean().item()),
            "passes_clean_signal_bar": bool(action_used),
        },
    }
    return metrics, verdict


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    if args.min_relative_effect < 0:
        raise ValueError("min-relative-effect must be non-negative")

    args.lightwam_root = args.lightwam_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for name in ("training_config", "dataset_stats", "latent_cache_dir", "text_cache_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.dataset_dir:
        args.dataset_dir = [p.expanduser().resolve() for p in args.dataset_dir]

    os.environ["FASTWAM_STRICT_DATASET_ERRORS"] = "1"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    lightwam_root = _prepare_lightwam_import(args.lightwam_root)
    cfg, training_config, stats_path = _load_cfg(args, lightwam_root)
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] instantiating released Light-WAM checkpoint", flush=True)
    model, dataset = _instantiate_model_and_dataset(cfg, args)
    audit = _architecture_audit(model)
    print("[audit]", json.dumps(audit, indent=2), flush=True)

    train_idx, test_idx, train_eps, test_eps = _clean_episode_sample_indices(
        dataset,
        num_samples=args.num_samples,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(
        f"[split] train={len(train_idx)} test={len(test_idx)} "
        f"episode_disjoint={set(train_eps).isdisjoint(set(test_eps))}",
        flush=True,
    )

    train = _collect(model, dataset, train_idx, args.batch_size)
    test = _collect(model, dataset, test_idx, args.batch_size)
    metrics, verdict = _analyse(train, test, args)

    result = {
        "topic": "Does Training-Time World Modeling Act Through a Predictive Policy State?",
        "platform": "Light-WAM",
        "lightwam_revision": _git_revision(lightwam_root),
        "checkpoint": str(checkpoint),
        "training_config": str(training_config),
        "dataset_stats": str(stats_path),
        "architecture": audit,
        "design": {
            "num_samples": args.num_samples,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "one_window_per_episode": True,
            "episode_disjoint_probe_split": True,
            "probe": "single fixed linear ridge on exact action-pooled native states",
            "probe_ridge": args.probe_ridge,
            "future_target": "all cached future VAE latents minus first latent frame; flattened, no PCA",
            "intervention": "replace each action-readout adapted tensor with its same-pass same-layer backbone tensor",
            "min_relative_effect": args.min_relative_effect,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "sample_indices": {"train": train_idx, "test": test_idx},
        "episode_ids": {"train": train_eps, "test": test_eps},
        "dimensions": {
            "pooled_feature": int(train["feature_adapted"].shape[1]),
            "future_target": int(train["target_future"].shape[1]),
        },
        "metrics": metrics,
        "verdict": verdict,
        "interpretation_limit": (
            "This G0 tests the native Light-WAM adapted-state route. Because the released model also "
            "uses trainable backbone LoRA and action supervision, a positive result is evidence for a "
            "causally action-used predictive state, not yet proof that future loss uniquely mediates the "
            "full WAM performance gain or that the state is an action-conditioned world model."
        ),
    }

    result_path = output_dir / "g0_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_tensors:
        torch.save(
            {
                "train": train,
                "test": test,
                "train_indices": train_idx,
                "test_indices": test_idx,
            },
            output_dir / "g0_tensors.pt",
        )

    print("\n=== G0 RESULT ===")
    print(json.dumps(metrics, indent=2))
    print(f"VERDICT: {verdict}")
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
