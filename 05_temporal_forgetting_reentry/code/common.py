#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Any

BOXED_RE = re.compile(r"\\boxed\s*\{", re.I)
FINAL_MARKER_RE = re.compile(r"(?:final\s+answer|answer\s+is|therefore.*answer)\s*[:=]?", re.I)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "correct"}:
            return True
        if s in {"0", "false", "f", "no", "n", "incorrect", "wrong"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {x!r}")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def split_reasoning_steps(trace: str) -> list[str]:
    """Transparent step splitter; intentionally not LLM-dependent.

    Prefer paragraph/line boundaries. If the response is one paragraph, fall
    back to sentence-ish boundaries. This is not claimed to recover semantic
    proof steps perfectly; the runbook requires a manual audit before G0.
    """
    trace = str(trace).strip()
    if not trace:
        return []
    lines = [x.strip() for x in re.split(r"\n+", trace) if x.strip()]
    if len(lines) >= 2:
        return lines
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", trace) if x.strip()]


def last_boxed_content(text: str) -> str | None:
    """Extract the content of the last balanced \\boxed{...}."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None
    left = text.find("{", idx)
    if left < 0:
        return None
    depth = 0
    for i in range(left, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[left + 1 : i].strip()
    return None


def normalize_answer_text(x: Any) -> str:
    s = str(x if x is not None else "").strip().lower()
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s)
    return s.strip(".$")


def explicit_answer_leak_reasons(prefix: str, gold_answer: Any) -> list[str]:
    """Conservative answer-leak detector.

    We do *not* ban every occurrence of the gold number/expression because a
    legitimate intermediate derivation may pass through the final value. We do
    ban explicit final-answer constructions and boxed gold answers.
    """
    reasons: list[str] = []
    gold = normalize_answer_text(gold_answer)
    boxed = last_boxed_content(prefix)
    if boxed is not None:
        if not gold or normalize_answer_text(boxed) == gold:
            reasons.append("boxed_answer")
        else:
            reasons.append("boxed_expression")

    for m in FINAL_MARKER_RE.finditer(prefix):
        tail = normalize_answer_text(prefix[m.end() : m.end() + 160])
        if not gold or (gold and gold in tail):
            reasons.append("explicit_final_answer_marker")
            break
    return sorted(set(reasons))


def stable_hash_int(text: str) -> int:
    import hashlib
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
