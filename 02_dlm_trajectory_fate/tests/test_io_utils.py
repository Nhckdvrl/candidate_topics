import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from io_utils import load_shards


def _write(path, ids):
    meta = {
        "model": "x",
        "num_examples": 4,
        "num_shards": 2,
        "steps": 4,
        "gen_length": 4,
        "block_length": 2,
        "temperature": 0.0,
        "prompt_style": "midtruth",
        "seed": 0,
        "surface_only": True,
        "n_regions": 1,
    }
    np.savez_compressed(
        path,
        problem_id=np.array(ids),
        capture_steps=np.array([0, 3]),
        hidden_indices=np.array([1]),
        correct_strict=np.zeros((2, 4), dtype=bool),
        observed_strict=np.ones((2, 4), dtype=bool),
        correct_fallback=np.zeros((2, 4), dtype=bool),
        observed_fallback=np.ones((2, 4), dtype=bool),
        answer_all=np.full((2, 4), ""),
        entropy=np.full((2, 2), np.nan),
        selected_prob=np.full((2, 2), np.nan),
        clean_maxprob=np.full((2, 2), np.nan),
        frac_unmasked=np.full((2, 2), np.nan),
        prompt_tokens=np.array([10, 11]),
        metadata_json=np.array(json.dumps(meta, sort_keys=True)),
    )


def test_load_shards_checks_and_sorts(tmp_path):
    _write(tmp_path / "shard_00_of_02.npz", [2, 0])
    _write(tmp_path / "shard_01_of_02.npz", [3, 1])
    data = load_shards(tmp_path)
    assert data["problem_id"].tolist() == [0, 1, 2, 3]


def test_require_hidden_rejects_surface_only(tmp_path):
    _write(tmp_path / "shard_00_of_02.npz", [0, 1])
    _write(tmp_path / "shard_01_of_02.npz", [2, 3])
    with pytest.raises(ValueError, match="Hidden states are missing"):
        load_shards(tmp_path, require_hidden=True)
