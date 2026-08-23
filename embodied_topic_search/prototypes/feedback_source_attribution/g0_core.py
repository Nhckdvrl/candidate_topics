from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITIONS = ("fresh", "vla_replay", "actuator_replay")


@dataclass(frozen=True)
class Cell:
    config_id: str
    force_n: float
    direction: str
    condition: str
    success: int


@dataclass(frozen=True)
class Attribution:
    fresh: float
    vla_replay: float
    actuator_replay: float
    high_level_gain: float
    low_level_gain: float
    residual_success: float
    n_cells: int


def _paired_cells(rows: Iterable[dict]) -> list[dict[str, Cell]]:
    by_key: dict[tuple[str, float, str], dict[str, Cell]] = {}
    for r in rows:
        c = Cell(
            config_id=str(r["config_id"]),
            force_n=float(r["force_n"]),
            direction=str(r["direction"]),
            condition=str(r["condition"]),
            success=int(bool(r["success"])),
        )
        if c.condition not in CONDITIONS:
            continue
        key = (c.config_id, c.force_n, c.direction)
        slot = by_key.setdefault(key, {})
        if c.condition in slot:
            raise ValueError(f"duplicate row for {key} / {c.condition}")
        slot[c.condition] = c
    return [v for v in by_key.values() if set(v) == set(CONDITIONS)]


def attribution(rows: Iterable[dict]) -> Attribution:
    cells = _paired_cells(rows)
    if not cells:
        raise ValueError("no complete fresh/vla_replay/actuator_replay cells")
    means = {
        cond: float(np.mean([c[cond].success for c in cells]))
        for cond in CONDITIONS
    }
    return Attribution(
        fresh=means["fresh"],
        vla_replay=means["vla_replay"],
        actuator_replay=means["actuator_replay"],
        high_level_gain=means["fresh"] - means["vla_replay"],
        low_level_gain=means["vla_replay"] - means["actuator_replay"],
        residual_success=means["actuator_replay"],
        n_cells=len(cells),
    )


def clustered_bootstrap(rows: Iterable[dict], *, n_boot: int = 10000, seed: int = 20260824) -> dict[str, tuple[float, float, float]]:
    """Bootstrap physical configs, preserving the complete force/direction panel."""
    rows = list(rows)
    configs = sorted({str(r["config_id"]) for r in rows})
    if len(configs) < 2:
        raise ValueError("need >=2 config clusters")
    by_cfg = {c: [r for r in rows if str(r["config_id"]) == c] for c in configs}
    point = attribution(rows)
    rng = np.random.default_rng(seed)
    hg, lg = [], []
    for _ in range(n_boot):
        sampled = rng.choice(configs, size=len(configs), replace=True)
        boot: list[dict] = []
        for j, c in enumerate(sampled):
            for r in by_cfg[str(c)]:
                x = dict(r)
                x["config_id"] = f"{c}__boot{j}"
                boot.append(x)
        a = attribution(boot)
        hg.append(a.high_level_gain)
        lg.append(a.low_level_gain)

    def pack(point_value: float, xs: list[float]) -> tuple[float, float, float]:
        lo, hi = np.quantile(xs, [0.025, 0.975])
        return float(point_value), float(lo), float(hi)

    return {
        "high_level_gain": pack(point.high_level_gain, hg),
        "low_level_gain": pack(point.low_level_gain, lg),
    }


def p0_fidelity(rows: Iterable[dict], *, min_success: float = 0.90, max_drop: float = 0.10) -> dict:
    """Technical gate on unperturbed replay only; do not inspect perturbation outcomes here."""
    rows = [r for r in rows if float(r.get("force_n", 0.0)) == 0.0]
    cells = _paired_cells(rows)
    if not cells:
        raise ValueError("no complete unperturbed replay cells")
    flat = [c[cond].__dict__ for c in cells for cond in CONDITIONS]
    a = attribution(flat)
    ok = (
        a.fresh >= min_success
        and a.vla_replay >= min_success
        and a.actuator_replay >= min_success
        and (a.fresh - a.vla_replay) <= max_drop
        and (a.fresh - a.actuator_replay) <= max_drop
    )
    return {
        "n_cells": a.n_cells,
        "fresh": a.fresh,
        "vla_replay": a.vla_replay,
        "actuator_replay": a.actuator_replay,
        "pass": bool(ok),
    }


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
