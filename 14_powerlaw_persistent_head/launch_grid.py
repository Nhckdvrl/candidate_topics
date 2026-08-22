#!/usr/bin/env python3
"""Launch locked conditions on one or more GPUs without distributed training.

With four GPUs, the four conditions run concurrently. With fewer GPUs, conditions
are executed in waves. This keeps the experiment usable in a shared environment
without adding Ray/torchrun/NCCL dependencies.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ALL_CONDITIONS = ["uniform", "static", "balanced_slow", "balanced_fast"]


def parse_conditions(raw: str) -> list[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(out) - set(ALL_CONDITIONS))
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}. Allowed: {ALL_CONDITIONS}")
    if not out:
        raise SystemExit("At least one condition is required")
    return out


def run_wave(jobs: list[tuple[str, str]], *, args: argparse.Namespace, seed: int, exp: Path) -> None:
    procs = []
    for cond, gpu in jobs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable,
            str(exp),
            "--condition", cond,
            "--profile", args.profile,
            "--seed", str(seed),
            "--output", args.output,
            "--precision", args.precision,
            "--lr-schedule", args.lr_schedule,
            "--warmup-steps", str(args.warmup_steps),
        ]
        if args.compile:
            cmd.append("--compile")
        cmd.extend(args.extra)
        log_dir = Path(args.output) / args.profile / f"seed{seed}" / cond
        log_dir.mkdir(parents=True, exist_ok=True)
        log_f = (log_dir / "stdout.log").open("w")
        print(f"GPU {gpu}: {' '.join(cmd)}", flush=True)
        procs.append((cond, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))

    failed = []
    for cond, proc, log_f in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failed.append((cond, code))
    if failed:
        raise SystemExit(f"seed {seed} failed: {failed}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="pilot", choices=["smoke", "pilot", "confirm", "full"])
    p.add_argument("--seeds", default="0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    p.add_argument("--output", default="outputs")
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    conditions = parse_conditions(args.conditions)
    if not seeds:
        raise SystemExit("At least one seed is required")
    if not gpus:
        raise SystemExit("At least one GPU id is required")

    here = Path(__file__).resolve().parent
    exp = here / "experiment.py"
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        for start in range(0, len(conditions), len(gpus)):
            wave_conditions = conditions[start : start + len(gpus)]
            jobs = list(zip(wave_conditions, gpus[: len(wave_conditions)]))
            run_wave(jobs, args=args, seed=seed, exp=exp)


if __name__ == "__main__":
    main()
