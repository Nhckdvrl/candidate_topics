"""Validation and aggregation for repeated hidden-state feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeaturePanel:
    state_id: np.ndarray
    checkpoint: np.ndarray
    sim_state_hash: np.ndarray
    feature: np.ndarray
    n_feature_seeds: np.ndarray


def load_feature_npz(path: str | Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    required = {"state_id", "checkpoint", "sim_state_hash", "feature_seed", "feature"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(f"missing feature arrays: {sorted(missing)}")
    out = {k: np.asarray(d[k]) for k in required}
    n = len(out["state_id"])
    if any(len(v) != n for k, v in out.items() if k != "feature") or len(out["feature"]) != n:
        raise ValueError("feature arrays are not row-aligned")
    if out["feature"].ndim != 2:
        raise ValueError("feature must be [N,D]")
    if not np.isfinite(out["feature"]).all():
        raise ValueError("feature contains non-finite values")
    return out


def aggregate_feature_replicates(raw: dict[str, np.ndarray], *, min_seeds: int = 4) -> FeaturePanel:
    state_id = np.asarray(raw["state_id"]).astype(str)
    checkpoint = np.asarray(raw["checkpoint"]).astype(str)
    hashes = np.asarray(raw["sim_state_hash"]).astype(str)
    fseed = np.asarray(raw["feature_seed"]).astype(int)
    feat = np.asarray(raw["feature"], float)
    if not (len(state_id) == len(checkpoint) == len(hashes) == len(fseed) == len(feat)):
        raise ValueError("feature rows do not align")

    for sid in np.unique(state_id):
        if len(np.unique(hashes[state_id == sid])) != 1:
            raise ValueError(f"sim_state_hash mismatch for feature state {sid}")

    cps = sorted(np.unique(checkpoint))
    rows_sid, rows_cp, rows_hash, rows_f, rows_n = [], [], [], [], []
    for sid in sorted(np.unique(state_id)):
        idx_state = state_id == sid
        present = sorted(np.unique(checkpoint[idx_state]))
        if present != cps:
            raise ValueError(f"incomplete checkpoint feature panel for state {sid}")
        seed_sets = {}
        for cp in cps:
            idx = idx_state & (checkpoint == cp)
            seeds = tuple(sorted(fseed[idx].tolist()))
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"duplicate feature_seed for state {sid}, checkpoint {cp}")
            if len(seeds) < min_seeds:
                raise ValueError(f"only {len(seeds)} feature seeds for state {sid}, checkpoint {cp}")
            seed_sets[cp] = seeds
        reference = seed_sets[cps[0]]
        if any(seed_sets[cp] != reference for cp in cps[1:]):
            raise ValueError(f"feature_seed sets differ across checkpoints for state {sid}")

        h = np.unique(hashes[idx_state])[0]
        for cp in cps:
            idx = idx_state & (checkpoint == cp)
            rows_sid.append(sid)
            rows_cp.append(cp)
            rows_hash.append(h)
            rows_f.append(feat[idx].mean(axis=0))
            rows_n.append(int(idx.sum()))

    return FeaturePanel(
        state_id=np.asarray(rows_sid),
        checkpoint=np.asarray(rows_cp),
        sim_state_hash=np.asarray(rows_hash),
        feature=np.stack(rows_f, axis=0),
        n_feature_seeds=np.asarray(rows_n, int),
    )


def save_aggregated_feature_panel(panel: FeaturePanel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state_id=panel.state_id,
        checkpoint=panel.checkpoint,
        sim_state_hash=panel.sim_state_hash,
        feature=panel.feature.astype(np.float32),
        n_feature_seeds=panel.n_feature_seeds,
    )


def concat_feature_npz(paths) -> dict[str, np.ndarray]:
    parts = [load_feature_npz(p) for p in paths]
    if not parts:
        raise ValueError("no feature files")
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
