#!/usr/bin/env python3
"""G0 screen for Topic 15.

Question at this stage:

Does the released Light-WAM contain a simple native adapter route such that
(1) enabling the trained WAM adapters makes the deployed policy state more
predictive of the real future in the SAME latent space used by future training,
and (2) bypassing those adapters worsens action prediction?

This is deliberately a SCREEN, not a full mediation proof. It does not claim
that the particular linearly decodable future bits are themselves the causal
code used by the action expert. A positive result says that a native pathway is
both more future-predictive and action-relevant, and is therefore worth a
matched-training mediation experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager
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
    p.add_argument("--training-config", type=Path, default=None)
    p.add_argument("--dataset-stats", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, action="append", default=None)
    p.add_argument("--latent-cache-dir", type=Path, default=None)
    p.add_argument("--text-cache-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("./g0_results"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--probe-ridge", type=float, default=1e-2)
    p.add_argument("--target-chunk-size", type=int, default=4096)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument(
        "--min-relative-effect",
        type=float,
        default=0.05,
        help="Continuation floor only; not a publication threshold.",
    )
    p.add_argument("--save-tensors", action="store_true")
    return p.parse_args()


def _resolve_near_checkpoint(checkpoint: Path, explicit: Path | None, filename: str) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    checkpoint = checkpoint.expanduser().resolve()
    seen = set()
    for parent in [checkpoint.parent, *list(checkpoint.parents)[:5]]:
        path = (parent / filename).resolve()
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


def _prepare_lightwam_import(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not (root / "src" / "lightwam").exists():
        raise FileNotFoundError(f"Not a Light-WAM checkout: {root}")
    for path in (root, root / "src"):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def _load_cfg(args: argparse.Namespace, lightwam_root: Path):
    from lightwam.utils.config_compat import load_compatible_omegaconf

    training_config = _resolve_near_checkpoint(args.checkpoint, args.training_config, "config.yaml")
    stats_path = _resolve_near_checkpoint(args.checkpoint, args.dataset_stats, "dataset_stats.json")
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

    model = instantiate(cfg.model, model_dtype=_dtype(args.dtype), device=args.device)
    model.load_checkpoint(str(args.checkpoint.expanduser().resolve()))
    model.eval()

    # Latent-cache G0 does not need the VAE or text encoder resident on GPU.
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
    if video_expert is not None:
        adapter_layers = [int(x) for x in getattr(video_expert, "adapter_layer_indices", ())]
        if not adapter_layers:
            problems.append("video expert has no WAM adapter layers")
        adapters = getattr(video_expert, "wam_adapters", None)
        if adapters is None or len(adapters) == 0:
            problems.append("video expert exposes no wam_adapters")
        else:
            for name, adapter in adapters.items():
                if not hasattr(adapter, "scale"):
                    problems.append(f"WAM adapter {name} has no mutable scale for clean bypass")
        if not hasattr(video_expert, "get_wam_action_fusion_layer_states"):
            problems.append("video expert does not expose native fusion layer states")

    for method in (
        "_build_multilayer_action_fusion_inputs",
        "_build_action_observation_video_pre",
        "_build_video_training_supervision_latents",
        "_maybe_downsample_video_latents_for_backbone",
    ):
        if not hasattr(model, method):
            problems.append(f"model missing {method}")

    if problems:
        raise RuntimeError("Architecture audit failed:\n- " + "\n- ".join(problems))

    return {
        "adapter_layers": adapter_layers,
        "layer_feature_sources": layer_sources,
        "token_pooling_type": str(getattr(expert, "token_pooling_type", "unknown")),
        "video_hidden_dim": int(getattr(expert, "video_hidden_dim", -1)),
        "use_backbone_lora": bool(getattr(video_expert, "use_backbone_lora", False)),
        "lora_layers": [int(x) for x in getattr(video_expert, "lora_layer_indices", ())],
        "video_latent_spatial_downsample_factor": int(
            getattr(model, "video_latent_spatial_downsample_factor", 1)
        ),
        "use_first_frame_residual_video_target": bool(
            getattr(model, "use_first_frame_residual_video_target", False)
        ),
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
        if bool(torch.as_tensor(value, dtype=torch.bool).any().item()):
            return False
    return True


def _fixed_state_summary(layer_states) -> torch.Tensor:
    """Parameter-free summary used ONLY for representation measurement.

    The action expert's learned-query pooler was trained only on adapted states.
    Using that learned pooler for a normal-vs-bypass representation comparison
    would bias the measurement toward the normal distribution. We therefore use
    the same fixed token mean on both sides, layer by layer, then concatenate.
    """
    pooled = []
    for state in layer_states:
        tokens = state["adapted"]
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B,S,D] adapted tokens, got {tuple(tokens.shape)}")
        pooled.append(tokens.mean(dim=1))
    return torch.cat(pooled, dim=-1)


@contextmanager
def _wam_adapter_scale(video_expert, scale: float):
    adapters = getattr(video_expert, "wam_adapters", None)
    if adapters is None or len(adapters) == 0:
        raise RuntimeError("No WAM adapters available for bypass")
    old = {}
    try:
        for name, adapter in adapters.items():
            old[name] = float(adapter.scale)
            adapter.scale = float(scale)
        yield
    finally:
        for name, adapter in adapters.items():
            adapter.scale = old[name]


def _build_action_pre(model, observation_latents, inputs):
    timestep = torch.zeros(
        observation_latents.shape[0],
        dtype=observation_latents.dtype,
        device=observation_latents.device,
    )
    return model._build_action_observation_video_pre(
        observation_latents=observation_latents,
        timestep_video=timestep,
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
    )


def _action_forward_and_states(model, video_pre, horizon: int):
    _ = model.video_expert.forward_backbone(video_pre)
    layer_states = model._build_multilayer_action_fusion_inputs()
    pred = model.state_fusion_action_expert(layer_states, action_horizon=horizon)
    return pred, layer_states


def _future_target_in_training_space(model, input_latents: torch.Tensor) -> torch.Tensor:
    """Build a clean future-change target in the spatial latent space used by video training."""
    future_latents = model._build_video_training_supervision_latents(input_latents)
    factor = int(getattr(model, "video_latent_spatial_downsample_factor", 1))
    if factor > 1:
        future_latents, _ = model._maybe_downsample_video_latents_for_backbone(future_latents)
    return future_change_target(future_latents)


def _run_native_batch(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    inputs = model.build_inputs(batch)
    latents = inputs["input_latents"]
    observation_latents = inputs["first_frame_latents"]
    if observation_latents is None:
        observation_latents = latents[:, :, 0:1]
    if observation_latents.shape[2] != 1:
        raise ValueError(f"Expected one observation latent frame, got {tuple(observation_latents.shape)}")

    horizon = int(inputs["action"].shape[1])

    # Normal deployed state.
    normal_pre = _build_action_pre(model, observation_latents, inputs)
    pred_normal, normal_states = _action_forward_and_states(model, normal_pre, horizon)
    feature_normal = _fixed_state_summary(normal_states)

    # Clean module intervention: rerun the SAME observation with every WAM adapter
    # set to identity (scale=0). This removes both local residuals and their
    # downstream propagation through later layers. Backbone LoRA is intentionally
    # left unchanged; this G0 isolates the explicit WAM-adapter route only.
    bypass_pre = _build_action_pre(model, observation_latents, inputs)
    with _wam_adapter_scale(model.video_expert, 0.0):
        pred_bypass, bypass_states = _action_forward_and_states(model, bypass_pre, horizon)

    # At scale zero every adapter output should exactly equal its local input.
    max_bypass_residual = 0.0
    for state in bypass_states:
        residual = (state["adapted"].float() - state["backbone"].float()).abs().max().item()
        max_bypass_residual = max(max_bypass_residual, float(residual))
    if max_bypass_residual > 1e-6:
        raise RuntimeError(
            f"Adapter bypass failed: max |adapted-backbone|={max_bypass_residual:.3e}"
        )
    feature_bypass = _fixed_state_summary(bypass_states)

    target_action = inputs["action"]
    action_is_pad = inputs["action_is_pad"]
    loss_normal = model._compute_action_loss_per_sample(
        pred_action=pred_normal,
        target_action=target_action,
        action_is_pad=action_is_pad,
    )
    loss_bypass = model._compute_action_loss_per_sample(
        pred_action=pred_bypass,
        target_action=target_action,
        action_is_pad=action_is_pad,
    )
    action_shift = torch.sqrt(
        ((pred_bypass.float() - pred_normal.float()) ** 2).mean(dim=(1, 2))
    )

    target_future = _future_target_in_training_space(model, latents)

    return {
        "feature_normal": feature_normal.detach().float().cpu(),
        "feature_bypass": feature_bypass.detach().float().cpu(),
        "target_future": target_future,
        "loss_normal": loss_normal.detach().float().cpu(),
        "loss_bypass": loss_bypass.detach().float().cpu(),
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
    probe_normal = linear_ridge_probe(
        train["feature_normal"],
        train["target_future"],
        test["feature_normal"],
        test["target_future"],
        ridge=args.probe_ridge,
        target_chunk_size=args.target_chunk_size,
    )
    probe_bypass = linear_ridge_probe(
        train["feature_bypass"],
        train["target_future"],
        test["feature_bypass"],
        test["target_future"],
        ridge=args.probe_ridge,
        target_chunk_size=args.target_chunk_size,
    )

    future_mse_gain = probe_bypass.per_sample_mse - probe_normal.per_sample_mse
    future_ci = _mean_ci(future_mse_gain, args.seed + 101, args.bootstrap)
    future_rel_gain = (
        (probe_bypass.mse - probe_normal.mse) / probe_bypass.mse
        if probe_bypass.mse > 0 else float("nan")
    )

    action_delta = test["loss_bypass"] - test["loss_normal"]
    action_ci = _mean_ci(action_delta, args.seed + 202, args.bootstrap)
    action_rel_increase = relative_change(
        float(test["loss_bypass"].mean().item()),
        float(test["loss_normal"].mean().item()),
    )

    future_gain = (
        math.isfinite(probe_normal.r2)
        and probe_normal.r2 > 0.0
        and future_rel_gain >= args.min_relative_effect
        and future_ci["ci95_low"] > 0.0
    )
    adapter_action_route = (
        action_rel_increase >= args.min_relative_effect
        and action_ci["ci95_low"] > 0.0
    )

    if future_gain and adapter_action_route:
        verdict = "PROCEED_TO_MATCHED_TRAINING"
    elif future_gain and not adapter_action_route:
        verdict = "FUTURE_GAIN_WITHOUT_CLEAN_ADAPTER_ACTION_EFFECT"
    elif (not future_gain) and adapter_action_route:
        verdict = "ADAPTER_ACTION_EFFECT_WITHOUT_FUTURE_GAIN"
    else:
        verdict = "NO_CLEAN_ADAPTER_ROUTE"

    metrics = {
        "future_probe_fixed_pooling": {
            "normal_adapters": {
                "r2": probe_normal.r2,
                "mse": probe_normal.mse,
                "mean_target_baseline_mse": probe_normal.baseline_mse,
            },
            "adapter_bypass": {
                "r2": probe_bypass.r2,
                "mse": probe_bypass.mse,
                "mean_target_baseline_mse": probe_bypass.baseline_mse,
            },
            "relative_mse_gain_from_adapters": future_rel_gain,
            "paired_mse_gain": future_ci,
            "passes_screen": bool(future_gain),
        },
        "causal_adapter_bypass": {
            "normal_action_loss_mean": float(test["loss_normal"].mean().item()),
            "adapter_bypass_action_loss_mean": float(test["loss_bypass"].mean().item()),
            "relative_loss_increase": action_rel_increase,
            "paired_loss_increase": action_ci,
            "action_rms_shift_mean": float(test["action_shift"].mean().item()),
            "passes_screen": bool(adapter_action_route),
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        "stage": "released-checkpoint native-route screen",
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
            "representation_summary": "fixed per-layer token mean; no trained action pooler used for the probe",
            "probe": "single fixed linear ridge",
            "probe_ridge": args.probe_ridge,
            "future_target": (
                "clean future VAE-latent change in the same spatial latent space used by the checkpoint's "
                "future/video training objective; flattened, no PCA"
            ),
            "intervention": (
                "second action-backbone pass on the identical observation with every explicit WAM adapter "
                "scale set to zero; backbone LoRA remains unchanged"
            ),
            "min_relative_effect": args.min_relative_effect,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "sample_indices": {"train": train_idx, "test": test_idx},
        "episode_ids": {"train": train_eps, "test": test_eps},
        "dimensions": {
            "fixed_pooled_feature": int(train["feature_normal"].shape[1]),
            "future_target": int(train["target_future"].shape[1]),
        },
        "metrics": metrics,
        "verdict": verdict,
        "interpretation_limit": (
            "This screen cannot identify the particular decodable future component as the causal code for action. "
            "A positive result only shows that enabling the native WAM-adapter pathway simultaneously increases "
            "future decodability and improves action readout. The released model also uses action supervision and "
            "backbone LoRA, so the full training-time mediation claim requires a matched-training experiment. "
            "A negative result kills this simple Light-WAM adapter route as the project's intended clean mechanism; "
            "it does not mathematically prove that no predictive state exists elsewhere in the model."
        ),
    }

    result_path = args.output_dir / "g0_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_tensors:
        torch.save(
            {
                "train": train,
                "test": test,
                "train_indices": train_idx,
                "test_indices": test_idx,
            },
            args.output_dir / "g0_tensors.pt",
        )

    print("\n=== G0 RESULT ===")
    print(json.dumps(metrics, indent=2))
    print(f"VERDICT: {verdict}")
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
