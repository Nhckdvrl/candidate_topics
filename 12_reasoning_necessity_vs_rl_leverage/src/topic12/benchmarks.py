from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class Example:
    task: str
    uid: str
    problem: str
    gold: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_key(seed: int, uid: str) -> str:
    return hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).hexdigest()


def stable_subset(examples: list[Example], n: int | None, seed: int) -> list[Example]:
    """Order-independent deterministic sampling.

    Selecting by a hash of stable example IDs avoids accidental dependence on a
    dataset library's row order. Every GPU worker therefore receives the exact
    same ledger without coordination.
    """
    ordered = sorted(examples, key=lambda x: _stable_key(seed, x.uid))
    if n is None or n <= 0 or n >= len(ordered):
        return ordered
    return ordered[:n]


def _uid(task: str, problem: str, explicit: str | None = None) -> str:
    if explicit:
        return f"{task}:{explicit}"
    digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:20]
    return f"{task}:{digest}"


def load_math500(n: int | None, seed: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows: list[Example] = []
    for row in ds:
        problem = str(row["problem"])
        # Current MATH-500 exposes `answer`; fall back to solution if needed.
        gold = str(row.get("answer") or row.get("solution") or "")
        explicit = row.get("unique_id") or row.get("id")
        meta = {
            k: row.get(k)
            for k in ("subject", "level", "solution")
            if k in row
        }
        rows.append(
            Example(
                task="math500",
                uid=_uid("math500", problem, str(explicit) if explicit is not None else None),
                problem=problem,
                gold=gold,
                metadata=meta,
            )
        )
    return stable_subset(rows, n, seed)


def load_gsm8k(n: int | None, seed: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows: list[Example] = []
    for row in ds:
        problem = str(row["question"])
        raw = str(row["answer"])
        gold = raw.split("####")[-1].strip()
        rows.append(
            Example(
                task="gsm8k",
                uid=_uid("gsm8k", problem),
                problem=problem,
                gold=gold,
                metadata={"reference_solution": raw},
            )
        )
    return stable_subset(rows, n, seed)


LOADERS = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
}


def load_tasks(task_names: Iterable[str], n_per_task: int | None, seed: int) -> dict[str, list[Example]]:
    result: dict[str, list[Example]] = {}
    for name in task_names:
        key = name.strip().lower()
        if key not in LOADERS:
            raise ValueError(f"Unsupported task {key!r}; choose from {sorted(LOADERS)}")
        result[key] = LOADERS[key](n_per_task, seed)
    return result


PROMPT_TEMPLATE = (
    "Solve the following mathematics problem. Reason step by step. "
    "Put the final answer in \\boxed{{}}.\n\n"
    "Problem:\n{problem}\n\nSolution:\n"
)


def make_prompt(example: Example) -> str:
    return PROMPT_TEMPLATE.format(problem=example.problem)


def _fallback_last_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    # Numeric fallback for GSM8K-like responses.
    nums = re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/[0-9]+)?", text)
    return nums[-1].replace(",", "") if nums else text.strip()


def grade_math(response: str, gold: str) -> tuple[bool, bool, str | None]:
    """Return (correct, parse_ok, error).

    Math-Verify is the locked primary grader. The fallback only prevents a
    parser exception from crashing an expensive sweep; fallback usage is
    recorded and audited by `check_integrity.py`.
    """
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold)
        pred_parsed = parse(response)
        parse_ok = bool(gold_parsed) and bool(pred_parsed)
        if parse_ok:
            return bool(verify(gold_parsed, pred_parsed)), True, None
        return (
            _fallback_last_answer(response) == _fallback_last_answer(gold),
            False,
            "math_verify_parse_empty",
        )
    except Exception as exc:  # preserve the expensive generation, never hide it
        return (
            _fallback_last_answer(response) == _fallback_last_answer(gold),
            False,
            f"{type(exc).__name__}: {exc}",
        )
