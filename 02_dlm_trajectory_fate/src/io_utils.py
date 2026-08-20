from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATIC_KEYS = {"capture_steps", "hidden_indices", "metadata_json"}


def load_shards(
    input_dir: Path,
    *,
    require_hidden: bool = False,
) -> dict[str, np.ndarray | dict]:
    files = sorted(input_dir.glob("shard_??_of_??.npz"))
    if not files:
        raise FileNotFoundError(f"No shard NPZ files in {input_dir}")

    loaded: list[dict[str, np.ndarray]] = []
    metadata: list[dict] = []
    for f in files:
        with np.load(f, allow_pickle=False) as b:
            blob = {k: b[k].copy() for k in b.files}
        if "metadata_json" not in blob:
            raise ValueError(
                f"{f} is from an old/incompatible run: metadata_json missing"
            )
        metadata.append(json.loads(str(blob["metadata_json"].item())))
        loaded.append(blob)

    meta0 = metadata[0]
    for f, meta in zip(files[1:], metadata[1:]):
        if meta != meta0:
            raise ValueError(
                f"Shard metadata mismatch: {files[0].name} vs {f.name}"
            )

    expected_shards = int(meta0["num_shards"])
    if len(files) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} shard files, found {len(files)} in {input_dir}"
        )

    for key in ["capture_steps", "hidden_indices"]:
        for f, b in zip(files[1:], loaded[1:]):
            if not np.array_equal(loaded[0][key], b[key]):
                raise ValueError(
                    f"Shard mismatch for {key}: {files[0].name} vs {f.name}"
                )

    common_keys = set(loaded[0])
    for b in loaded[1:]:
        common_keys &= set(b)

    if require_hidden and "hidden" not in common_keys:
        raise ValueError(
            "Hidden states are missing. This looks like a --surface-only run."
        )

    out: dict[str, np.ndarray | dict] = {
        "capture_steps": loaded[0]["capture_steps"],
        "hidden_indices": loaded[0]["hidden_indices"],
        "metadata": meta0,
    }

    for key in sorted(common_keys - STATIC_KEYS):
        arrays = [b[key] for b in loaded]
        if arrays[0].ndim == 0:
            continue
        out[key] = np.concatenate(arrays, axis=0)

    problem_id = np.asarray(out["problem_id"])
    if len(np.unique(problem_id)) != len(problem_id):
        raise ValueError("Duplicate problem_id values across shards")

    expected_examples = int(meta0["num_examples"])
    if len(problem_id) != expected_examples:
        raise ValueError(
            f"Expected {expected_examples} examples, loaded {len(problem_id)}"
        )

    order = np.argsort(problem_id)
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0:
            if value.shape[0] == len(order) and key not in {
                "capture_steps",
                "hidden_indices",
            }:
                out[key] = value[order]

    return out
