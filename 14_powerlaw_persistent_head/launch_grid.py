#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from core import LOCKED_FULL_SEEDS, LOCKED_PAPER_ANCHOR_SEEDS


def visible_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        if not vals or vals == ["-1"]:
            return []
        return vals
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def default_seeds(profile: str) -> str:
    if profile == "full":
        return ",".join(map(str, LOCKED_FULL_SEEDS))
    if profile == "paper_anchor":
        return ",".join(map(str, LOCKED_PAPER_ANCHOR_SEEDS))
    return "0"


def run_wave(commands: list[tuple[list[str], dict[str, str]]]) -> None:
    procs = [subprocess.Popen(cmd, env=env) for cmd, env in commands]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"arm failure codes={codes}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="pilot")
    p.add_argument("--seeds", default=None)
    p.add_argument("--output", default="outputs")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    seed_text = args.seeds or default_seeds(args.profile)
    seeds = [int(x) for x in seed_text.split(",") if x.strip()]
    devices = visible_devices()
    arms = ["uniform", "static"] if args.profile == "paper_anchor" else ["uniform", "static", "slow", "fast"]

    for seed in seeds:
        warm = [sys.executable, "train.py", "--mode", "warmup", "--profile", args.profile, "--seed", str(seed), "--output", args.output]
        if args.resume:
            warm.append("--resume")
        subprocess.run(warm, check=True)

        if not devices:
            for arm in arms:
                cmd = [sys.executable, "train.py", "--mode", "arm", "--condition", arm, "--profile", args.profile, "--seed", str(seed), "--output", args.output, "--device", "cpu"]
                if args.resume:
                    cmd.append("--resume")
                subprocess.run(cmd, check=True)
            continue

        # One process per visible GPU. With < number-of-arms GPUs, run waves rather
        # than oversubscribing a device. Child CUDA_VISIBLE_DEVICES values are the
        # actual tokens inherited from the parent, so external masks such as 2,3
        # remain respected.
        for start in range(0, len(arms), len(devices)):
            wave = []
            for arm, device_token in zip(arms[start : start + len(devices)], devices):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = device_token
                cmd = [sys.executable, "train.py", "--mode", "arm", "--condition", arm, "--profile", args.profile, "--seed", str(seed), "--output", args.output]
                if args.resume:
                    cmd.append("--resume")
                wave.append((cmd, env))
            run_wave(wave)

    subprocess.run(
        [sys.executable, "analyze.py", "--root", str(Path(args.output) / args.profile), "--profile", args.profile, "--seeds", seed_text],
        check=True,
    )


if __name__ == "__main__":
    main()
