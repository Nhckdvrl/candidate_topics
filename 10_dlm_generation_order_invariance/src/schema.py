from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PuzzleSpec:
    protocol_version: str
    puzzle_id: str
    puzzle: str
    solution: str
    split: str
    blank_indices: list[int]
    candidate_counts: dict[str, int]
    transforms: list[dict[str, Any]]


@dataclass
class TraceRecord:
    puzzle_id: str
    variant_id: str
    split: str
    remasking: str
    puzzle: str
    solution: str
    transform: dict[str, Any] | None
    blank_indices: list[int]
    predicted_digits: list[int]
    finalization_step: dict[str, int]
    confidence_at_finalization: dict[str, float]
    valid_solution: bool
    exact_solution: bool
    metadata: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "to_json"):
                f.write(record.to_json())
            elif hasattr(record, "__dataclass_fields__"):
                f.write(json.dumps(asdict(record), sort_keys=True))
            else:
                f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
