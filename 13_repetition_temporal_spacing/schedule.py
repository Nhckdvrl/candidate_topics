#!/usr/bin/env python3
"""Build fixed-multiset temporal-spacing schedules for Topic 13.

Causal invariant for repeated conditions:
  * same repeated documents and exact multiplicities,
  * same non-repeated document at every non-repeat slot,
  * same repeat-slot positions,
  * at most ONE repeat slot in each optimizer step.

Thus clustered vs even changes when a repeated identity reappears across optimizer
updates, without changing within-step duplicate multiplicity/batch diversity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONDITIONS = ("fresh", "clustered", "random", "even")
SCHEDULE_SCHEMA_VERSION = 2


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
    blocks_per_optimizer_step: int
    optimizer_steps: int
    max_repeat_slots_per_optimizer_step: int
    schema_version: int = SCHEDULE_SCHEMA_VERSION


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def spacing_stats(schedule: np.ndarray, repeat_doc_ids: np.ndarray, blocks_per_step: int) -> dict[str, float]:
    block_gaps: list[int] = []
    step_gaps: list[int] = []
    block_spans: list[int] = []
    step_spans: list[int] = []
    cvs: list[float] = []
    for doc_id in repeat_doc_ids.tolist():
        pos = np.flatnonzero(schedule == doc_id)
        if len(pos) < 2:
            continue
        db = np.diff(pos).astype(np.float64)
        steps = pos // blocks_per_step
        ds = np.diff(steps).astype(np.float64)
        block_gaps.extend(db.tolist())
        step_gaps.extend(ds.tolist())
        block_spans.append(int(pos[-1] - pos[0]))
        step_spans.append(int(steps[-1] - steps[0]))
        if float(ds.mean()) > 0:
            cvs.append(float(ds.std() / ds.mean()))
    bg = np.asarray(block_gaps, dtype=np.float64)
    sg = np.asarray(step_gaps, dtype=np.float64)
    return {
        "mean_gap_blocks": float(bg.mean()),
        "median_gap_blocks": float(np.median(bg)),
        "mean_gap_optimizer_steps": float(sg.mean()),
        "median_gap_optimizer_steps": float(np.median(sg)),
        "p10_gap_optimizer_steps": float(np.percentile(sg, 10)),
        "p90_gap_optimizer_steps": float(np.percentile(sg, 90)),
        "mean_span_blocks": float(np.mean(block_spans)),
        "mean_span_optimizer_steps": float(np.mean(step_spans)),
        "mean_step_gap_cv": float(np.mean(cvs)),
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
        return np.tile(base, repeat_count)
    raise ValueError(kind)


def _choose_repeat_slots(total_blocks: int, repeat_slots_n: int, blocks_per_step: int, rng: np.random.Generator) -> np.ndarray:
    n_steps = math.ceil(total_blocks / blocks_per_step)
    if repeat_slots_n > n_steps:
        raise ValueError(
            f"Need {repeat_slots_n} repeat slots but only {n_steps} optimizer steps. "
            "Reduce effective batch size or repetition budget so each optimizer step can contain at most one repeat slot."
        )
    chosen_steps = np.sort(rng.choice(n_steps, size=repeat_slots_n, replace=False))
    slots = np.empty(repeat_slots_n, dtype=np.int64)
    for i, step in enumerate(chosen_steps.tolist()):
        start = step * blocks_per_step
        stop = min(total_blocks, start + blocks_per_step)
        slots[i] = int(rng.integers(start, stop))
    slots.sort()
    assert len(np.unique(slots // blocks_per_step)) == repeat_slots_n
    return slots


def build_schedules(total_blocks: int, repeat_fraction: float, repeat_count: int, seed: int, blocks_per_optimizer_step: int = 8) -> tuple[ScheduleSpec, dict[str, np.ndarray], dict]:
    if total_blocks < 100:
        raise ValueError("total_blocks too small")
    if not 0 < repeat_fraction < 0.5:
        raise ValueError("repeat_fraction must be in (0, 0.5)")
    if repeat_count < 2:
        raise ValueError("repeat_count must be >= 2")
    if blocks_per_optimizer_step < 1:
        raise ValueError("blocks_per_optimizer_step must be >= 1")

    rng = np.random.default_rng(seed)
    target_repeat_slots = int(round(total_blocks * repeat_fraction))
    repeat_docs = target_repeat_slots // repeat_count
    if repeat_docs < 2:
        raise ValueError("repeat pool has fewer than two documents")
    repeat_slots_n = repeat_docs * repeat_count
    unique_slots_n = total_blocks - repeat_slots_n
    required = repeat_docs + unique_slots_n + repeat_slots_n

    corpus_perm = rng.permutation(required).astype(np.int64)
    repeat_doc_ids = corpus_perm[:repeat_docs]
    unique_doc_ids = corpus_perm[repeat_docs : repeat_docs + unique_slots_n]
    fresh_doc_ids = corpus_perm[repeat_docs + unique_slots_n :]
    assert len(fresh_doc_ids) == repeat_slots_n

    repeat_slots = _choose_repeat_slots(total_blocks, repeat_slots_n, blocks_per_optimizer_step, rng)
    is_repeat = np.zeros(total_blocks, dtype=bool)
    is_repeat[repeat_slots] = True
    unique_slots = np.flatnonzero(~is_repeat)

    unique_assignment = unique_doc_ids.copy()
    rng.shuffle(unique_assignment)

    schedules: dict[str, np.ndarray] = {}
    offsets = {"fresh": 101, "clustered": 211, "random": 307, "even": 401}
    for condition in CONDITIONS:
        s = np.empty(total_blocks, dtype=np.int64)
        s[unique_slots] = unique_assignment
        local_rng = np.random.default_rng(seed + offsets[condition])
        if condition == "fresh":
            x = fresh_doc_ids.copy()
            local_rng.shuffle(x)
        else:
            x = _build_repeat_assignment(condition, repeat_doc_ids, repeat_count, local_rng)
        s[repeat_slots] = x
        schedules[condition] = s

    n_steps = math.ceil(total_blocks / blocks_per_optimizer_step)
    spec = ScheduleSpec(total_blocks, repeat_fraction, repeat_count, repeat_docs, repeat_slots_n, unique_slots_n, repeat_slots_n / total_blocks, required, seed, blocks_per_optimizer_step, n_steps, 1)

    audit: dict = {
        "spec": asdict(spec),
        "repeat_doc_ids_sha256": sha256_array(np.sort(repeat_doc_ids)),
        "repeat_slots_sha256": sha256_array(repeat_slots),
        "conditions": {},
    }
    ref_unique = schedules["random"][unique_slots]
    repeat_step_ids = repeat_slots // blocks_per_optimizer_step
    assert len(np.unique(repeat_step_ids)) == len(repeat_step_ids)

    for condition, s in schedules.items():
        row = {
            "sha256": sha256_array(s),
            "unique_slots_sha256": sha256_array(s[unique_slots]),
            "repeat_slot_positions_sha256": sha256_array(repeat_slots),
            "max_repeat_slots_same_optimizer_step": 1,
        }
        if condition != "fresh":
            vals, counts = np.unique(s[repeat_slots], return_counts=True)
            assert np.array_equal(np.sort(vals), np.sort(repeat_doc_ids))
            assert np.all(counts == repeat_count)
            assert np.array_equal(s[unique_slots], ref_unique)
            row["spacing"] = spacing_stats(s, repeat_doc_ids, blocks_per_optimizer_step)
            row["repeated_multiset_sha256"] = sha256_array(np.sort(s[repeat_slots]))
        audit["conditions"][condition] = row

    hashes = {audit["conditions"][c]["repeated_multiset_sha256"] for c in ("clustered", "random", "even")}
    assert len(hashes) == 1
    assert len({audit["conditions"][c]["unique_slots_sha256"] for c in CONDITIONS}) == 1
    cg = audit["conditions"]["clustered"]["spacing"]["mean_gap_optimizer_steps"]
    eg = audit["conditions"]["even"]["spacing"]["mean_gap_optimizer_steps"]
    if not (cg < eg):
        raise RuntimeError(f"spacing construction failed at optimizer-step scale: clustered={cg}, even={eg}")
    audit["spacing_order_diagnostic_optimizer_steps"] = {"clustered": cg, "even": eg}
    return spec, schedules, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-blocks", type=int, required=True)
    p.add_argument("--repeat-fraction", type=float, default=0.10)
    p.add_argument("--repeat-count", type=int, default=1386)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--blocks-per-optimizer-step", type=int, default=8)
    args = p.parse_args()
    _, schedules, audit = build_schedules(args.total_blocks, args.repeat_fraction, args.repeat_count, args.seed, args.blocks_per_optimizer_step)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition, arr in schedules.items():
        np.save(args.out_dir / f"{condition}.npy", arr.astype(np.int32))
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit["spec"], indent=2, sort_keys=True))
    print("optimizer-step spacing:", json.dumps(audit["spacing_order_diagnostic_optimizer_steps"], sort_keys=True))

if __name__ == "__main__":
    main()
