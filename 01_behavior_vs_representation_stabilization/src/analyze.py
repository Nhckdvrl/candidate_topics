from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from metrics import double_center, kl_proxy_bits_per_byte, linear_cka


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare behavioral and representational movement across Pythia checkpoints.")
    p.add_argument("--input-dir", default="artifacts/checkpoints")
    p.add_argument("--output-dir", default="artifacts/analysis")
    p.add_argument("--steps", type=int, nargs="+", default=[1000,2000,4000,8000,16000,32000,48000,64000,80000,96000,112000,128000,143000])
    p.add_argument("--cka-max-observations", type=int, default=4000, help="Deterministic subsample for faster CKA; 0 means all.")
    return p.parse_args()


def load_step(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blobs = [load_step(in_dir / f"step{s}.npz") for s in args.steps]
    base_ids = blobs[0]["dataset_row_ids"]
    base_pos = blobs[0]["sampled_positions"]
    base_layers = blobs[0]["block_layers"]
    for s, b in zip(args.steps[1:], blobs[1:]):
        if not np.array_equal(base_ids, b["dataset_row_ids"]):
            raise ValueError(f"Dataset rows differ at step {s}; all checkpoints must use identical examples.")
        if not np.array_equal(base_pos, b["sampled_positions"]):
            raise ValueError(f"Sampled token positions differ at step {s}; tokenizer/input mismatch.")
        if not np.array_equal(base_layers, b["block_layers"]):
            raise ValueError(f"Layer selection differs at step {s}.")

    ll = np.stack([b["log_likelihood"].astype(np.float64) for b in blobs], axis=0)
    q = double_center(ll)
    mean_bytes = float(blobs[0]["byte_lengths"].mean())

    rng = np.random.default_rng(0)
    n_obs = blobs[0]["hidden"].shape[0]
    if args.cka_max_observations and n_obs > args.cka_max_observations:
        obs_idx = np.sort(rng.choice(n_obs, size=args.cka_max_observations, replace=False))
    else:
        obs_idx = np.arange(n_obs)

    rows = []
    for i in range(len(args.steps) - 1):
        s0, s1 = args.steps[i], args.steps[i + 1]
        behavior = kl_proxy_bits_per_byte(q[i], q[i + 1], mean_bytes)
        row = {"step_from": s0, "step_to": s1, "behavior_kl_proxy_bits_per_byte": behavior}
        h0 = blobs[i]["hidden"]
        h1 = blobs[i + 1]["hidden"]
        repr_moves = []
        for li, block_idx in enumerate(base_layers.tolist()):
            cka = linear_cka(h0[obs_idx, li].astype(np.float32), h1[obs_idx, li].astype(np.float32))
            move = 1.0 - cka
            row[f"cka_layer_{block_idx}"] = cka
            row[f"repr_movement_layer_{block_idx}"] = move
            repr_moves.append(move)
        row["repr_movement_mean"] = float(np.nanmean(repr_moves))
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "adjacent_metrics.csv"
    df.to_csv(csv_path, index=False)

    x = df["step_to"].to_numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(x, df["behavior_kl_proxy_bits_per_byte"], marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("training step (interval endpoint)")
    plt.ylabel("behavior movement: KL proxy (bits/byte)")
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_vs_step.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for block_idx in base_layers.tolist():
        plt.plot(x, df[f"repr_movement_layer_{block_idx}"], marker="o", label=f"layer {block_idx}")
    plt.xscale("log")
    plt.xlabel("training step (interval endpoint)")
    plt.ylabel("representation movement: 1 - linear CKA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "representation_vs_step.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(df["behavior_kl_proxy_bits_per_byte"], df["repr_movement_mean"])
    for _, r in df.iterrows():
        plt.annotate(str(int(r["step_to"])), (r["behavior_kl_proxy_bits_per_byte"], r["repr_movement_mean"]), fontsize=7)
    plt.xscale("log")
    plt.xlabel("behavior movement (bits/byte)")
    plt.ylabel("mean representation movement (1-CKA)")
    plt.tight_layout()
    plt.savefig(out_dir / "behavior_vs_representation.png", dpi=180)
    plt.close()

    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
