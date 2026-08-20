from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from metrics import (
    double_center,
    kl_proxy_bits_per_byte,
    linear_cka,
    lower_clip_per_checkpoint,
    mean_by_example,
    pair_outlier_mask,
)

DEFAULT_PAIR_STARTS = [2000, 5000, 10000, 20000, 50000, 100000, 142000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G0 analysis for behavior/representation stabilization.")
    p.add_argument("--phase", choices=["behavior", "representation"], required=True)
    p.add_argument("--input-dir", default="artifacts/checkpoints")
    p.add_argument("--output-dir", default="artifacts/analysis")
    p.add_argument("--pair-starts", type=int, nargs="+", default=DEFAULT_PAIR_STARTS)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--lower-clip", type=float, default=0.02)
    p.add_argument("--trim-fraction", type=float, default=0.03)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--cka-bootstrap", type=int, default=20)
    p.add_argument("--cka-proj-dim", type=int, default=128)
    p.add_argument("--layer-index", type=int, default=None)
    return p.parse_args()


def steps_and_pairs(starts: list[int], horizon: int) -> tuple[list[int], list[tuple[int, int]]]:
    step_values = sorted(set(starts + [s + horizon for s in starts]))
    index = {s: i for i, s in enumerate(step_values)}
    pairs = [(index[s], index[s + horizon]) for s in starts]
    return step_values, pairs


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def verify_behavior_blobs(blobs, steps):
    base_ids = blobs[0]["example_id"]
    base_bytes = blobs[0]["byte_lengths"]
    for step, blob in zip(steps[1:], blobs[1:]):
        if not np.array_equal(base_ids, blob["example_id"]):
            raise ValueError(f"example_id mismatch at step {step}")
        if not np.array_equal(base_bytes, blob["byte_lengths"]):
            raise ValueError(f"byte_lengths mismatch at step {step}")
    return base_ids, base_bytes


def bootstrap_behavior(ll, bytes_, pairs, n_boot, rng):
    n = ll.shape[1]
    values = np.empty((n_boot, len(pairs)), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        q = double_center(ll[:, idx])
        mean_bytes = float(bytes_[idx].mean())
        for p, (i, j) in enumerate(pairs):
            values[b, p] = kl_proxy_bits_per_byte(q[i], q[j], mean_bytes)
    return np.quantile(values, [0.025, 0.975], axis=0)


def analyze_behavior(args, in_dir: Path, out_dir: Path, steps: list[int], pairs: list[tuple[int, int]]):
    blobs = [load_npz(in_dir / f"step{s}_behavior.npz") for s in steps]
    _, byte_lengths = verify_behavior_blobs(blobs, steps)
    ll = np.stack([b["log_likelihood"].astype(np.float64) for b in blobs], axis=0)

    q_raw = double_center(ll)
    clipped = lower_clip_per_checkpoint(ll, args.lower_clip)
    keep, outlier_scores = pair_outlier_mask(clipped, pairs, args.trim_fraction)
    q_robust = double_center(clipped[:, keep])

    rng = np.random.default_rng(0)
    raw_ci = bootstrap_behavior(ll, byte_lengths, pairs, args.bootstrap, rng)
    robust_ci = bootstrap_behavior(clipped[:, keep], byte_lengths[keep], pairs, args.bootstrap, rng)

    rows = []
    for p, ((i, j), start) in enumerate(zip(pairs, args.pair_starts)):
        raw = kl_proxy_bits_per_byte(q_raw[i], q_raw[j], float(byte_lengths.mean()))
        robust = kl_proxy_bits_per_byte(q_robust[i], q_robust[j], float(byte_lengths[keep].mean()))
        rows.append(
            {
                "step_from": start,
                "step_to": start + args.horizon,
                "horizon": args.horizon,
                "kl_raw_bits_per_byte": raw,
                "kl_raw_ci_low": raw_ci[0, p],
                "kl_raw_ci_high": raw_ci[1, p],
                "kl_robust_bits_per_byte": robust,
                "kl_robust_ci_low": robust_ci[0, p],
                "kl_robust_ci_high": robust_ci[1, p],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "behavior_metrics.csv", index=False)

    pd.DataFrame(
        {
            "example_id": blobs[0]["example_id"],
            "outlier_score_max_abs_pair_ll_change": outlier_scores,
            "kept_for_robust": keep,
        }
    ).to_csv(out_dir / "behavior_outliers.csv", index=False)

    x = df["step_from"].to_numpy()
    plt.figure(figsize=(7, 4))
    for prefix, label in [("kl_raw", "raw"), ("kl_robust", "robust")]:
        y = df[f"{prefix}_bits_per_byte"].to_numpy()
        lo = df[f"{prefix}_ci_low"].to_numpy()
        hi = df[f"{prefix}_ci_high"].to_numpy()
        plt.plot(x, y, marker="o", label=label)
        plt.fill_between(x, lo, hi, alpha=0.15)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"training step t (fixed horizon Δ={args.horizon})")
    plt.ylabel("local KL proxy (bits/byte)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_constant_horizon.png", dpi=180)
    plt.close()

    summary = {
        "n_examples_raw": int(ll.shape[1]),
        "n_examples_robust": int(keep.sum()),
        "lower_clip_quantile": args.lower_clip,
        "trim_fraction": args.trim_fraction,
        "fixed_horizon": args.horizon,
        "first_to_last_robust_ratio": float(df.iloc[-1]["kl_robust_bits_per_byte"] / df.iloc[0]["kl_robust_bits_per_byte"]),
        "note": "G0-A is a premise check. Do not inspect representation until the fixed-horizon KL trajectory shows clear late-training stabilization and raw/robust conclusions agree.",
    }
    (out_dir / "behavior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))


def find_repr_file(in_dir: Path, step: int, layer_idx: int | None) -> tuple[Path, int]:
    if layer_idx is not None:
        path = in_dir / f"step{step}_repr_l{layer_idx}.npz"
        return path, layer_idx
    matches = sorted(in_dir.glob(f"step{step}_repr_l*.npz"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one representation file for step {step}, found {matches}")
    stem = matches[0].stem
    inferred = int(stem.rsplit("_repr_l", 1)[1])
    return matches[0], inferred


def per_observation_drifts(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xn = np.linalg.norm(x, axis=1)
    yn = np.linalg.norm(y, axis=1)
    cosine = np.sum(x * y, axis=1) / np.maximum(xn * yn, 1e-12)
    cosine_drift = 1.0 - np.clip(cosine, -1.0, 1.0)

    pooled = np.concatenate([x, y], axis=0)
    mu = pooled.mean(axis=0, keepdims=True)
    sd = np.maximum(pooled.std(axis=0, keepdims=True), 1e-6)
    standardized_sq = np.square((x - mu) / sd - (y - mu) / sd).mean(axis=1)
    return cosine_drift, standardized_sq


def example_average_hidden(h: np.ndarray, example_ids: np.ndarray, n_examples: int) -> np.ndarray:
    out = np.zeros((n_examples, h.shape[1]), dtype=np.float64)
    counts = np.bincount(example_ids, minlength=n_examples).astype(np.float64)
    np.add.at(out, example_ids, h.astype(np.float64))
    out /= counts[:, None]
    return out


def bootstrap_mean_ci(per_example: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    n = len(per_example)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = per_example[idx].mean()
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def bootstrap_cka_ci(x_proj, y_proj, n_boot, rng):
    if n_boot <= 0:
        return float("nan"), float("nan")
    n = len(x_proj)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = linear_cka(x_proj[idx], y_proj[idx])
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def analyze_representation(args, in_dir: Path, out_dir: Path, steps: list[int], pairs: list[tuple[int, int]]):
    blobs = []
    layer_idx = args.layer_index
    for step in steps:
        path, inferred = find_repr_file(in_dir, step, layer_idx)
        if layer_idx is None:
            layer_idx = inferred
        elif inferred != layer_idx:
            raise ValueError("Layer mismatch across checkpoints")
        blobs.append(load_npz(path))

    base_ids = blobs[0]["example_id"]
    base_obs_ids = blobs[0]["obs_example_id"]
    base_pos = blobs[0]["obs_position"]
    for step, blob in zip(steps[1:], blobs[1:]):
        if not np.array_equal(base_ids, blob["example_id"]):
            raise ValueError(f"example_id mismatch at step {step}")
        if not np.array_equal(base_obs_ids, blob["obs_example_id"]):
            raise ValueError(f"observation example IDs mismatch at step {step}")
        if not np.array_equal(base_pos, blob["obs_position"]):
            raise ValueError(f"sampled token positions mismatch at step {step}")

    n_examples = len(base_ids)
    hidden_dim = blobs[0]["hidden"].shape[1]
    proj_dim = min(args.cka_proj_dim, hidden_dim)
    rng_proj = np.random.default_rng(12345)
    projection = rng_proj.normal(size=(hidden_dim, proj_dim)) / np.sqrt(proj_dim)
    rng = np.random.default_rng(0)

    rows = []
    for (i, j), start in zip(pairs, args.pair_starts):
        x = blobs[i]["hidden"].astype(np.float32)
        y = blobs[j]["hidden"].astype(np.float32)
        cos_obs, std_obs = per_observation_drifts(x, y)
        cos_ex = mean_by_example(cos_obs, base_obs_ids, n_examples)
        std_ex = mean_by_example(std_obs, base_obs_ids, n_examples)
        cos_ci = bootstrap_mean_ci(cos_ex, args.bootstrap, rng)
        std_ci = bootstrap_mean_ci(std_ex, args.bootstrap, rng)

        x_ex = example_average_hidden(x, base_obs_ids, n_examples)
        y_ex = example_average_hidden(y, base_obs_ids, n_examples)
        x_proj = x_ex @ projection
        y_proj = y_ex @ projection
        cka = linear_cka(x_proj, y_proj)
        cka_ci = bootstrap_cka_ci(x_proj, y_proj, args.cka_bootstrap, rng)
        rows.append(
            {
                "step_from": start,
                "step_to": start + args.horizon,
                "layer_idx": layer_idx,
                "cosine_drift": float(cos_ex.mean()),
                "cosine_ci_low": cos_ci[0],
                "cosine_ci_high": cos_ci[1],
                "standardized_drift": float(std_ex.mean()),
                "standardized_ci_low": std_ci[0],
                "standardized_ci_high": std_ci[1],
                "projected_linear_cka": cka,
                "cka_ci_low": cka_ci[0],
                "cka_ci_high": cka_ci[1],
                "cka_movement": 1.0 - cka,
                "cka_projection_dim": proj_dim,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "representation_metrics.csv", index=False)

    xstep = df["step_from"].to_numpy()
    plt.figure(figsize=(7, 4))
    for metric, label in [
        ("cosine_drift", "matched cosine drift"),
        ("standardized_drift", "pooled-standardized drift"),
        ("cka_movement", "1 - projected CKA"),
    ]:
        plt.plot(xstep, df[metric], marker="o", label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"training step t (fixed horizon Δ={args.horizon})")
    plt.ylabel("representation movement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "representation_constant_horizon.png", dpi=180)
    plt.close()

    behavior_path = out_dir / "behavior_metrics.csv"
    summary = {
        "layer_idx": layer_idx,
        "fixed_horizon": args.horizon,
        "cosine_first_to_last_ratio": float(df.iloc[-1]["cosine_drift"] / df.iloc[0]["cosine_drift"]),
        "standardized_first_to_last_ratio": float(df.iloc[-1]["standardized_drift"] / df.iloc[0]["standardized_drift"]),
        "cka_movement_first_to_last_ratio": float(df.iloc[-1]["cka_movement"] / max(df.iloc[0]["cka_movement"], 1e-12)),
        "note": "CKA is a rotation/scale-tolerant control, not a falsifier by itself. G0-B is promising only when behavior is clearly stable while matched/standardized representation movement remains systematically elevated.",
    }

    if behavior_path.exists():
        behavior = pd.read_csv(behavior_path)
        merged = behavior.merge(df, on=["step_from", "step_to"], how="inner")
        merged.to_csv(out_dir / "behavior_vs_representation.csv", index=False)
        plt.figure(figsize=(5, 5))
        plt.scatter(merged["kl_robust_bits_per_byte"], merged["standardized_drift"])
        for _, row in merged.iterrows():
            plt.annotate(str(int(row["step_from"])), (row["kl_robust_bits_per_byte"], row["standardized_drift"]), fontsize=7)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("robust behavior movement (bits/byte)")
        plt.ylabel("pooled-standardized representation drift")
        plt.tight_layout()
        plt.savefig(out_dir / "behavior_vs_representation.png", dpi=180)
        plt.close()
        summary["behavior_robust_first_to_last_ratio"] = float(
            merged.iloc[-1]["kl_robust_bits_per_byte"] / merged.iloc[0]["kl_robust_bits_per_byte"]
        )

    (out_dir / "representation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, pairs = steps_and_pairs(args.pair_starts, args.horizon)
    if args.phase == "behavior":
        analyze_behavior(args, in_dir, out_dir, steps, pairs)
    else:
        analyze_representation(args, in_dir, out_dir, steps, pairs)


if __name__ == "__main__":
    main()
