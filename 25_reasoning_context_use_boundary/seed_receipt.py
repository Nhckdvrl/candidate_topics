#!/usr/bin/env python3
"""Check the frozen Weakest-Link Qwen3-8B reproduction receipt.

This script does not rescore model text with a new evaluator. It reads the
`correct` field written by the pinned upstream Weakest-Link Python runners,
which already use the paper's official Exact Match implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PINNED_UPSTREAM_COMMIT = "9b01abaad354208a6a8fb26c58eb5c330036fb94"
PUBLISHED_GOLD_ONLY = {
    "Qwen3-8B": 0.4246,
    "Qwen3-8B-Think": 0.4470,
}
BUCKETS = {
    "beginning": "b",
    "midsection": "m",
    "tail": "t",
}
DISTANCES = (1, 2, 3, 4, 5)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            rows.append(obj)
    return rows


def _bank_ids(path: Path) -> List[str]:
    rows = _read_jsonl(path)
    ids: List[str] = []
    for i, row in enumerate(rows):
        qid = row.get("question_id", row.get("id"))
        if qid is None:
            raise ValueError(f"Bank row {i} has no question_id/id")
        ids.append(str(qid))
    if len(set(ids)) != len(ids):
        raise ValueError("Bank contains duplicate question IDs")
    return ids


def _accuracy(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    vals = [1.0 if bool(r.get("correct")) else 0.0 for r in rows]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _validate_result_file(
    path: Path,
    expected_ids: List[str],
    *,
    expected_thinking: bool,
    expected_prompt_id: int,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "complete": False,
    }
    if not path.exists():
        return None, diag

    rows = _read_jsonl(path)
    row_ids = [str(r.get("question_id", "")) for r in rows]
    id_set = set(row_ids)
    expected_set = set(expected_ids)
    duplicates = len(id_set) != len(row_ids)

    thinking_ok = all(bool(r.get("enable_thinking")) == expected_thinking for r in rows)
    prompt_ok = all(int(r.get("prompt_id", -999)) == expected_prompt_id for r in rows)
    correct_field_ok = all(isinstance(r.get("correct"), bool) for r in rows)

    diag.update(
        {
            "n": len(rows),
            "expected_n": len(expected_ids),
            "duplicate_ids": duplicates,
            "missing_ids": len(expected_set - id_set),
            "extra_ids": len(id_set - expected_set),
            "thinking_flag_ok": thinking_ok,
            "prompt_id_ok": prompt_ok,
            "correct_field_ok": correct_field_ok,
            "accuracy": _accuracy(rows),
        }
    )
    diag["complete"] = bool(
        len(rows) == len(expected_ids)
        and not duplicates
        and id_set == expected_set
        and thinking_ok
        and prompt_ok
        and correct_field_ok
    )
    return rows, diag


def _spread_path(root: Path, model_dir: str, bucket: str, distance: int) -> Path:
    letter = BUCKETS[bucket]
    return (
        root
        / "spread"
        / model_dir
        / bucket
        / f"gold_at_{letter}_dist{distance}_na.jsonl"
    )


def _gold_path(root: Path, model_dir: str) -> Path:
    return root / "gold_only" / model_dir / "gold_only_results.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-file", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/receipt/summary.json",
        help="Receipt summary JSON path.",
    )
    args = parser.parse_args()

    bank_file = Path(args.bank_file).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_ids = _bank_ids(bank_file)
    diagnostics: Dict[str, Any] = {}

    all_rows: Dict[str, List[Dict[str, Any]]] = {}

    for model_dir, think in (("Qwen3-8B", False), ("Qwen3-8B-Think", True)):
        key = f"gold_only/{model_dir}"
        rows, diag = _validate_result_file(
            _gold_path(results_root, model_dir),
            expected_ids,
            expected_thinking=think,
            expected_prompt_id=0,
        )
        diagnostics[key] = diag
        if rows is not None:
            all_rows[key] = rows

    spread_cell_rows: Dict[str, List[Dict[str, Any]]] = {
        "Qwen3-8B": [],
        "Qwen3-8B-Think": [],
    }
    spread_bucket_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "Qwen3-8B": {b: [] for b in BUCKETS},
        "Qwen3-8B-Think": {b: [] for b in BUCKETS},
    }

    for model_dir, think in (("Qwen3-8B", False), ("Qwen3-8B-Think", True)):
        for bucket in BUCKETS:
            for distance in DISTANCES:
                path = _spread_path(results_root, model_dir, bucket, distance)
                key = f"spread/{model_dir}/{bucket}/d{distance}"
                rows, diag = _validate_result_file(
                    path,
                    expected_ids,
                    expected_thinking=think,
                    expected_prompt_id=22,
                )
                diagnostics[key] = diag
                if rows is not None:
                    spread_cell_rows[model_dir].extend(rows)
                    spread_bucket_rows[model_dir][bucket].extend(rows)

    complete = all(d.get("complete", False) for d in diagnostics.values())

    gold_no = diagnostics.get("gold_only/Qwen3-8B", {}).get("accuracy")
    gold_think = diagnostics.get("gold_only/Qwen3-8B-Think", {}).get("accuracy")
    noisy_no = _accuracy(spread_cell_rows["Qwen3-8B"])
    noisy_think = _accuracy(spread_cell_rows["Qwen3-8B-Think"])

    def _ge(a: Optional[float], b: Optional[float]) -> bool:
        return a is not None and b is not None and a >= b

    def _gt(a: Optional[float], b: Optional[float]) -> bool:
        return a is not None and b is not None and a > b

    relations = {
        "gold_think_ge_nonthink": _ge(gold_think, gold_no),
        "noisy_think_ge_think_gold_only": _ge(noisy_think, gold_think),
        "noisy_think_gt_noisy_nonthink": _gt(noisy_think, noisy_no),
    }

    verdict = (
        "SEED_RELATION_REPRODUCED"
        if complete and all(relations.values())
        else "SEED_RELATION_NOT_REPRODUCED"
    )

    summary: Dict[str, Any] = {
        "verdict": verdict,
        "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
        "bank_file": str(bank_file),
        "bank_n": len(expected_ids),
        "published_gold_only": PUBLISHED_GOLD_ONLY,
        "observed": {
            "gold_only": {
                "Qwen3-8B": gold_no,
                "Qwen3-8B-Think": gold_think,
            },
            "published_gold_only_abs_error": {
                "Qwen3-8B": (
                    None if gold_no is None else abs(gold_no - PUBLISHED_GOLD_ONLY["Qwen3-8B"])
                ),
                "Qwen3-8B-Think": (
                    None
                    if gold_think is None
                    else abs(gold_think - PUBLISHED_GOLD_ONLY["Qwen3-8B-Think"])
                ),
            },
            "spread_na_pooled": {
                "Qwen3-8B": noisy_no,
                "Qwen3-8B-Think": noisy_think,
            },
            "spread_na_by_bucket": {
                model: {bucket: _accuracy(rows) for bucket, rows in by_bucket.items()}
                for model, by_bucket in spread_bucket_rows.items()
            },
        },
        "relations": relations,
        "complete_support": complete,
        "diagnostics": diagnostics,
        "note": (
            "Published gold-only values are recorded, not tolerance-gated. "
            "The receipt gate uses complete support plus the seed paper's qualitative relations."
        ),
    }

    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
