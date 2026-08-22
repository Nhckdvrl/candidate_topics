from __future__ import annotations

from math import sqrt
from statistics import median
from typing import Sequence
import random


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        v = values[order[pos]]
        while end < len(order) and values[order[end]] == v:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for j in range(pos, end):
            out[order[j]] = avg_rank
        pos = end
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if den == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Dependency-free Kendall tau-b with ties."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    den = sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / den if den else float("nan")


def bootstrap_ci(values: Sequence[float], seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    vals = [float(v) for v in values if v == v]
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * (len(boots) - 1))]
    hi = boots[int((1 - alpha / 2) * (len(boots) - 1))]
    return lo, hi


def summarize(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values if v == v]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "median": median(vals), "min": min(vals), "max": max(vals)}
