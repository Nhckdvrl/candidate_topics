#!/usr/bin/env python3
"""Normalize public MCQ datasets into Topic-04 JSONL.

Primary use:
    MMLU-Pro test, exact K=10.

The benchmark split is used as a supervised-learning stimulus pool; this script
does not claim held-out benchmark evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_index(labels: list[str], answer_key: str) -> int:
    key = str(answer_key).strip()
    if key in labels:
        return labels.index(key)
    # Some ARC variants use integer-like labels.
    if key.isdigit():
        if key in labels:
            return labels.index(key)
        idx = int(key)
        if 0 <= idx < len(labels):
            return idx
        if 1 <= idx <= len(labels):
            return idx - 1
    raise ValueError(f"Cannot map answer key {answer_key!r} into labels {labels!r}")


def load_mmlu_pro(split: str) -> list[dict]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    out = []
    for r in ds:
        out.append(
            {
                "id": f"mmlu_pro:{r['question_id']}",
                "dataset": "mmlu_pro",
                "category": str(r.get("category", "unknown")),
                "source": str(r.get("src", "")),
                "question": str(r["question"]),
                "choices": [str(x) for x in r["options"]],
                "answer": int(r["answer_index"]),
            }
        )
    return out


def load_mmlu(split: str) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split=split)
    out = []
    for i, r in enumerate(ds):
        answer = int(r["answer"])
        out.append(
            {
                "id": f"mmlu:{split}:{i}",
                "dataset": "mmlu",
                "category": str(r.get("subject", "unknown")),
                "question": str(r["question"]),
                "choices": [str(x) for x in r["choices"]],
                "answer": answer,
            }
        )
    return out


def load_arc(split: str) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    out = []
    for r in ds:
        labels = [str(x) for x in r["choices"]["label"]]
        choices = [str(x) for x in r["choices"]["text"]]
        out.append(
            {
                "id": f"arc_challenge:{r['id']}",
                "dataset": "arc_challenge",
                "category": "science",
                "question": str(r["question"]),
                "choices": choices,
                "answer": answer_index(labels, r["answerKey"]),
            }
        )
    return out


def load_openbookqa(split: str) -> list[dict]:
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    out = []
    for r in ds:
        labels = [str(x) for x in r["choices"]["label"]]
        choices = [str(x) for x in r["choices"]["text"]]
        out.append(
            {
                "id": f"openbookqa:{r['id']}",
                "dataset": "openbookqa",
                "category": "science",
                "question": str(r["question_stem"]),
                "choices": choices,
                "answer": answer_index(labels, r["answerKey"]),
            }
        )
    return out


def load_medmcqa(split: str) -> list[dict]:
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    out = []
    for r in ds:
        choices = [str(r[k]) for k in ("opa", "opb", "opc", "opd")]
        out.append(
            {
                "id": f"medmcqa:{r['id']}",
                "dataset": "medmcqa",
                "category": str(r.get("subject_name", "medicine")),
                "question": str(r["question"]),
                "choices": choices,
                "answer": int(r["cop"]),
            }
        )
    return out


LOADERS = {
    "mmlu_pro": load_mmlu_pro,
    "mmlu": load_mmlu,
    "arc_challenge": load_arc,
    "openbookqa": load_openbookqa,
    "medmcqa": load_medmcqa,
}


def validate(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        if r["id"] in seen:
            raise ValueError(f"Duplicate ID: {r['id']}")
        seen.add(r["id"])
        k = len(r["choices"])
        a = int(r["answer"])
        if k < 2 or not 0 <= a < k:
            raise ValueError(f"Bad choices/answer for {r['id']}")
        if any(not str(x).strip() for x in r["choices"]):
            continue
        if not str(r["question"]).strip():
            continue
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--require-k", type=int, default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = validate(LOADERS[args.dataset](args.split))
    if args.require_k is not None:
        rows = [r for r in rows if len(r["choices"]) == args.require_k]
    if args.max_items is not None:
        rows = rows[: args.max_items]
    write_jsonl(args.output, rows)
    by_k: dict[int, int] = {}
    for r in rows:
        by_k[len(r["choices"])] = by_k.get(len(r["choices"]), 0) + 1
    print(json.dumps({"n": len(rows), "choice_counts": by_k}, indent=2))


if __name__ == "__main__":
    main()
