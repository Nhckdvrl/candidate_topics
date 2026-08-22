#!/usr/bin/env python3
"""G1 evaluation: matched future-on vs future-off, with the same adapter intervention.

Both arms are evaluated on the IDENTICAL episode-disjoint sample set, with the identical
fixed token-mean summary, the identical fixed linear ridge probe and the identical full
adapter bypass used by G0. The quantities produced are the four the README requires:

1. policy gain            = action_loss(future-off) - action_loss(future-on)          [normal]
2. predictive-state gain  = future-probe MSE gain from adapters, per arm
3. adapter-bypass cost    = C_arm = action_loss(bypass) - action_loss(normal)
4. the mediation interaction  dC = C_on - C_off
   plus whether the future-on policy advantage shrinks under bypass.

Nothing here is fitted per arm beyond the one fixed ridge probe that G0 already used.
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

import g0_lightwam as g0
from g0_core import linear_ridge_probe, paired_bootstrap_mean_ci, relative_change


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lightwam-root", type=Path, required=True)
    p.add_argument("--matched-config", type=Path, required=True)
    p.add_argument("--dataset-stats", type=Path, required=True)
    p.add_argument("--future-on-checkpoint", type=Path, required=True)
    p.add_argument("--future-off-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--probe-ridge", type=float, default=1e-2)
    p.add_argument("--target-chunk-size", type=int, default=4096)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-relative-effect", type=float, default=0.05)
    return p.parse_args()


def _mean_ci(values: torch.Tensor, seed: int, n_boot: int) -> dict[str, float]:
    mean, lo, hi = paired_bootstrap_mean_ci(values, seed=seed, n_boot=n_boot)
    return {"mean": mean, "ci95_low": lo, "ci95_high": hi}


def _load_arm(cfg, args: argparse.Namespace, checkpoint: Path):
    from hydra.utils import instantiate

    model = instantiate(cfg.model, model_dtype=g0._dtype(args.dtype), device=args.device)
    model.load_checkpoint(str(checkpoint.expanduser().resolve()))
    model.eval()
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.to("cpu")
    if getattr(model, "vae", None) is not None:
        model.vae.to("cpu")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def _probe_gain(train, test, args) -> dict[str, Any]:
    normal = linear_ridge_probe(
        train["feature_normal"], train["target_future"],
        test["feature_normal"], test["target_future"],
        ridge=args.probe_ridge, target_chunk_size=args.target_chunk_size,
    )
    bypass = linear_ridge_probe(
        train["feature_bypass"], train["target_future"],
        test["feature_bypass"], test["target_future"],
        ridge=args.probe_ridge, target_chunk_size=args.target_chunk_size,
    )
    per_sample_gain = bypass.per_sample_mse - normal.per_sample_mse
    return {
        "normal_r2": normal.r2,
        "normal_mse": normal.mse,
        "bypass_r2": bypass.r2,
        "bypass_mse": bypass.mse,
        "mean_target_baseline_mse": normal.baseline_mse,
        "relative_mse_gain_from_adapters": (
            (bypass.mse - normal.mse) / bypass.mse if bypass.mse > 0 else float("nan")
        ),
        "paired_mse_gain": _mean_ci(per_sample_gain, args.seed + 101, args.bootstrap),
        "_per_sample_gain": per_sample_gain,
    }


def main() -> None:
    args = parse_args()
    args.lightwam_root = args.lightwam_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import os
    import random

    os.environ["FASTWAM_STRICT_DATASET_ERRORS"] = "1"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = g0._prepare_lightwam_import(args.lightwam_root)
    from omegaconf import OmegaConf

    from lightwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    cfg = OmegaConf.load(str(args.matched_config.expanduser().resolve()))
    cfg.model.load_text_encoder = False
    cfg.data.train.use_latent_cache = True
    cfg.data.train.pretrained_norm_stats = str(args.dataset_stats.expanduser().resolve())
    cfg.data.train.val_set_proportion = 0.0
    cfg.data.train.is_training_set = True
    if bool(cfg.model.wam_adapter.use_backbone_lora):
        raise ValueError("matched evaluation expects the LoRA-free matched config")
    os.chdir(root)

    from hydra.utils import instantiate

    dataset = instantiate(cfg.data.train)
    train_idx, test_idx, train_eps, test_eps = g0._clean_episode_sample_indices(
        dataset,
        num_samples=args.num_samples,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    assert set(train_eps).isdisjoint(set(test_eps))
    print(f"[split] train={len(train_idx)} test={len(test_idx)} episode_disjoint=True", flush=True)

    arms: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    for arm, ckpt in (("future_on", args.future_on_checkpoint), ("future_off", args.future_off_checkpoint)):
        print(f"[arm] loading {arm}: {ckpt}", flush=True)
        model = _load_arm(cfg, args, ckpt)
        audits[arm] = g0._architecture_audit(model)
        train = g0._collect(model, dataset, train_idx, args.batch_size)
        test = g0._collect(model, dataset, test_idx, args.batch_size)
        arms[arm] = {"train": train, "test": test}
        del model
        torch.cuda.empty_cache()

    on, off = arms["future_on"], arms["future_off"]

    # --- 1. policy gain from future supervision (normal deployed path) ---
    policy_delta = off["test"]["loss_normal"] - on["test"]["loss_normal"]
    policy_gain = {
        "future_on_action_loss": float(on["test"]["loss_normal"].mean()),
        "future_off_action_loss": float(off["test"]["loss_normal"].mean()),
        "relative_gain_from_future_supervision": relative_change(
            float(on["test"]["loss_normal"].mean()), float(off["test"]["loss_normal"].mean())
        ),
        "paired_loss_reduction": _mean_ci(policy_delta, args.seed + 301, args.bootstrap),
    }

    # --- 2. predictive-state gain, per arm ---
    probe_on = _probe_gain(on["train"], on["test"], args)
    probe_off = _probe_gain(off["train"], off["test"], args)
    predictive_state_interaction = _mean_ci(
        probe_on["_per_sample_gain"] - probe_off["_per_sample_gain"], args.seed + 401, args.bootstrap
    )

    # --- 3. adapter-bypass cost, per arm, and 4. the interaction ---
    cost_on = on["test"]["loss_bypass"] - on["test"]["loss_normal"]
    cost_off = off["test"]["loss_bypass"] - off["test"]["loss_normal"]
    bypass_interaction = _mean_ci(cost_on - cost_off, args.seed + 501, args.bootstrap)

    # does the future-on advantage shrink once the shared adapters are removed?
    advantage_normal = off["test"]["loss_normal"] - on["test"]["loss_normal"]
    advantage_bypass = off["test"]["loss_bypass"] - on["test"]["loss_bypass"]
    advantage_shrinkage = _mean_ci(advantage_normal - advantage_bypass, args.seed + 601, args.bootstrap)

    for entry in (probe_on, probe_off):
        entry.pop("_per_sample_gain")

    gate_policy = (
        policy_gain["relative_gain_from_future_supervision"] <= -args.min_relative_effect
        and policy_gain["paired_loss_reduction"]["ci95_low"] > 0.0
    )
    gate_predictive = (
        probe_on["relative_mse_gain_from_adapters"] >= args.min_relative_effect
        and predictive_state_interaction["ci95_low"] > 0.0
    )
    gate_interaction = bypass_interaction["ci95_low"] > 0.0

    if gate_policy and gate_predictive and gate_interaction:
        verdict = "MECHANISM_SUPPORTED"
    elif not gate_policy:
        verdict = "NO_POLICY_GAIN_FROM_FUTURE_SUPERVISION"
    elif not gate_predictive:
        verdict = "POLICY_GAIN_WITHOUT_PREDICTIVE_ADAPTER_STATE"
    else:
        verdict = "NO_ADAPTER_DEPENDENCE_INTERACTION"

    result = {
        "topic": "Does Training-Time World Modeling Act Through a Predictive Policy State?",
        "stage": "G1 matched future-on / future-off mechanism test",
        "lightwam_revision": g0._git_revision(root),
        "checkpoints": {
            "future_on": str(args.future_on_checkpoint),
            "future_off": str(args.future_off_checkpoint),
        },
        "architecture": audits,
        "design": {
            "num_samples": args.num_samples,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "one_window_per_episode": True,
            "episode_disjoint_probe_split": True,
            "same_samples_for_both_arms": True,
            "min_relative_effect": args.min_relative_effect,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "episode_ids": {"train": train_eps, "test": test_eps},
        "metrics": {
            "policy_gain": policy_gain,
            "predictive_state": {
                "future_on": probe_on,
                "future_off": probe_off,
                "interaction_on_minus_off": predictive_state_interaction,
            },
            "adapter_bypass_cost": {
                "future_on": {
                    "normal_action_loss": float(on["test"]["loss_normal"].mean()),
                    "bypass_action_loss": float(on["test"]["loss_bypass"].mean()),
                    "relative_loss_increase": relative_change(
                        float(on["test"]["loss_bypass"].mean()), float(on["test"]["loss_normal"].mean())
                    ),
                    "paired_cost": _mean_ci(cost_on, args.seed + 701, args.bootstrap),
                },
                "future_off": {
                    "normal_action_loss": float(off["test"]["loss_normal"].mean()),
                    "bypass_action_loss": float(off["test"]["loss_bypass"].mean()),
                    "relative_loss_increase": relative_change(
                        float(off["test"]["loss_bypass"].mean()), float(off["test"]["loss_normal"].mean())
                    ),
                    "paired_cost": _mean_ci(cost_off, args.seed + 801, args.bootstrap),
                },
                "interaction_on_minus_off": bypass_interaction,
            },
            "future_on_advantage": {
                "under_normal_adapters": float(advantage_normal.mean()),
                "under_adapter_bypass": float(advantage_bypass.mean()),
                "paired_shrinkage": advantage_shrinkage,
            },
        },
        "gates": {
            "policy_gain": bool(gate_policy),
            "predictive_state_gain": bool(gate_predictive),
            "adapter_dependence_interaction": bool(gate_interaction),
        },
        "verdict": verdict,
    }

    path = args.output_dir / "g1_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(arms, args.output_dir / "g1_tensors.pt")
    print("\n=== G1 RESULT ===")
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["gates"], indent=2))
    print(f"VERDICT: {verdict}")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
