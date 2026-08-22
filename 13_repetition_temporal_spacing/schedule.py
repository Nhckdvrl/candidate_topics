#!/usr/bin/env python3
"""Build exact fixed-multiset temporal-spacing schedules for Topic 13.

The repeated conditions share:
  * the same repeat slots,
  * the same unique-document IDs at every non-repeat slot,
  * the same repeated-document IDs and exact multiplicities.
Only the assignment of repeated identities to repeat slots changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


CONDITIONS = ("fresh", "clustered", "random", "even")


@dataclass(frozen=True)
class ScheduleSpec:
    total_blocks: int
    repeat_fraction_target: float
    repeat_count: int
    repeat_docs: int
    repeat_slots: int
    unique_slots: int
    realized_repeat_fraction: float
    required_corpus_blocks: int
    seed: int


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def spacing_stats(schedule: np.ndarray, repeat_doc_ids: np.ndarray) -> dict[str, float]:
    gaps: list[int] = []
    spans: list[int] = []
    cvs: list[float] = []
    for doc_id in repeat_doc_ids.tolist():
        pos = np.flatnonzero(schedule == doc_id)
        if len(pos) < 2:
            continue
        d = np.diff(pos).astype(np.float64)
        gaps.extend(d.tolist())
        spans.append(int(pos[-1] - pos[0]))
        if float(d.mean()) > 0:
            cvs.append(float(d.std() / d.mean()))
    g = np.asarray(gaps, dtype=np.float64)
    return {
        "mean_gap_blocks": float(g.mean()) if len(g) else float("nan"),
        "median_gap_blocks": float(np.median(g)) if len(g) else float("nan"),
        "p10_gap_blocks": float(np.percentile(g, 10)) if len(g) else float("nan"),
        "p90_gap_blocks": float(np.percentile(g, 90)) if len(g) else float("nan"),
        "mean_span_blocks": float(np.mean(spans)) if spans else float("nan"),
        "mean_gap_cv": float(np.mean(cvs)) if cvs else float("nan"),
    }


def _build_repeat_assignment(kind: str, repeat_doc_ids: np.ndarray, repeat_count: int, rng: np.random.Generator) -> np.ndarray:
    k = len(repeat_doc_ids)
    if kind == "clustered":
        order = repeat_doc_ids.copy()
        rng.shuffle(order)
        return np.repeat(order, repeat_count)
    if kind == "random":
        x = np.repeat(repeat_doc_ids, repeat_count)
        rng.shuffle(x)
        return x
    if kind == "even":
        base = repeat_doc_ids.copy()
        rng.shuffle(base)
        rounds = []
        for r in range(repeat_count):
            shift = r % max(1, k)
            rounds.append(np.roll(base, shift))
        return np.concatenate(rounds)
    raise ValueError(kind)


def build_schedules(total_blocks: int, repeat_fraction: float, repeat_count: int, seed: int) -> tuple[ScheduleSpec, dict[str, np.ndarray], dict]:
    if total_blocks < 100:
        raise ValueError("total_blocks too small")
    if not 0 < repeat_fraction < 0.5:
        raise ValueError("repeat_fraction must be in (0, 0.5)")
    if repeat_count < 2:
        raise ValueError("repeat_count must be >= 2")

    rng = np.random.default_rng(seed)
    target_repeat_slots = int(round(total_blocks * repeat_fraction))
    repeat_docs = target_repeat_slots // repeat_count
    if repeat_docs < 2:
        raise ValueError("repeat pool has fewer than two documents; increase budget or lower repeat_count")
    repeat_slots_n = repeat_docs * repeat_count
    unique_slots_n = total_blocks - repeat_slots_n

    repeat_doc_ids = np.arange(0, repeat_docs, dtype=np.int64)
    unique_doc_ids = np.arange(repeat_docs, repeat_docs + unique_slots_n, dtype=np.int64)
    fresh_doc_ids = np.arange(repeat_docs + unique_slots_n, repeat_docs + unique_slots_n + repeat_slots_n, dtype=np.int64)
    required = int(fresh_doc_ids[-1]) + 1

    repeat_slots = np.sort(rng.choice(total_blocks, size=repeat_slots_n, replace=False)).astype(np.int64)
    is_repeat = np.zeros(total_blocks, dtype=bool)
    is_repeat[repeat_slots] = True
    unique_slots = np.flatnonzero(~is_repeat)

    unique_assignment = unique_doc_ids.copy()
    rng.shuffle(unique_assignment)

    schedules: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        s = np.empty(total_blocks, dtype=np.int64)
        s[unique_slots] = unique_assignment
        local_rng = np.random.default_rng(seed + {"fresh": 101, "clustered": 211, "random": 307, "even": 401}[condition])
        if condition == "fresh":
            x = fresh_doc_ids.copy()
            local_rng.shuffle(x)
        else:
            x = _build_repeat_assignment(condition, repeat_doc_ids, repeat_count, local_rng)
        assert len(x) == len(repeat_slots)
        s[repeat_slots] = x
        schedules[condition] = s

    spec = ScheduleSpec(
        total_blocks=total_blocks,
        repeat_fraction_target=repeat_fraction,
        repeat_count=repeat_count,
        repeat_docs=repeat_docs,
        repeat_slots=repeat_slots_n,
        unique_slots=unique_slots_n,
        realized_repeat_fraction=repeat_slots_n / total_blocks,
        required_corpus_blocks=required,
        seed=seed,
    )

    audit: dict = {"spec": asdict(spec), "conditions": {}}
    shared_unique_reference = schedules["random"][unique_slots]
    for condition, s in schedules.items():
        audit["conditions"][condition] = {
            "sha256": sha256_array(s),
            "unique_slots_sha256": sha256_array(s[unique_slots]),
            "repeat_slot_positions_sha256": sha256_array(repeat_slots),
        }
        if condition != "fresh":
            vals, counts = np.unique(s[repeat_slots], return_counts=True)
            assert np.array_equal(vals, repeat_doc_ids)
            assert np.all(counts == repeat_count)
            assert np.array_equal(s[unique_slots], shared_unique_reference)
            audit["conditions"][condition]["spacing"] = spacing_stats(s, repeat_doc_ids)
            audit["conditions"][condition]["repeated_multiset_sha256"] = sha256_array(np.sort(s[repeat_slots]))

    hashes = {audit["conditions"][c]["repeated_multiset_sha256"] for c in ("clustered", "random", "even")}
    assert len(hashes) == 1
    assert len({audit["conditions"][c]["unique_slots_sha256"] for c in CONDITIONS}) == 1

    cg = audit["conditions"]["clustered"]["spacing"]["mean_gap_blocks"]
    rg = audit["conditions"]["random"]["spacing"]["mean_gap_blocks"]
    eg = audit["conditions"]["even"]["spacing"]["mean_gap_blocks"]
    if not (cg < eg):
        raise RuntimeError(f"spacing construction failed: clustered mean gap={cg}, even={eg}")
    audit["spacing_order_diagnostic"] = {"clustered": cg, "random": rg, "even": eg}
    return spec, schedules, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-blocks", type=int, required=True)
    p.add_argument("--repeat-fraction", type=float, default=0.10)
    p.add_argument("--repeat-count", type=int, default=1386)
    p.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()

    _, schedules, audit = build_schedules(args.total_blocks, args.repeat_fraction, args.repeat_count, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition, arr in schedules.items():
        np.save(args.out_dir / f"{condition}.npy", arr.astype(np.int32))
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit["spec"], indent=2, sort_keys=True))
    print("spacing:", json.dumps(audit["spacing_order_diagnostic"], sort_keys=True))


if __name__ == "__main__":
    main()
