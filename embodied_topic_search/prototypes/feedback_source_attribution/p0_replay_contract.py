from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _arr(x: Any) -> tuple[float, ...]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise ValueError("command contains non-finite values")
    return tuple(float(v) for v in a)


@dataclass(frozen=True)
class VlaCommand:
    """The high-level command at the VLA -> whole-body-controller seam."""

    target_upper_body_pose: tuple[float, ...]
    navigate_cmd: tuple[float, ...]
    base_height_command: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VlaCommand":
        upper = payload["target_upper_body_pose"]
        if isinstance(upper, dict):
            upper = [upper[k] for k in sorted(upper)]
        return cls(_arr(upper), _arr(payload["navigate_cmd"]), _arr(payload["base_height_command"]))


@dataclass(frozen=True)
class ActuatorReference:
    """The post-WBC reference at the whole-body-controller -> actuator seam."""

    target_q: tuple[float, ...]
    left_hand_q: tuple[float, ...]
    right_hand_q: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ActuatorReference":
        return cls(_arr(payload["target_q"]), _arr(payload["left_hand_q"]), _arr(payload["right_hand_q"]))


@dataclass(frozen=True)
class TapeRow:
    step: int
    vla: VlaCommand
    actuator: ActuatorReference


class ReplayTape:
    """Lossless command tape for validating the two causal cut points.

    Record both seams on an unperturbed canonical rollout, then replay one seam at
    a time from the same realised initial state. This module intentionally contains
    no simulator or policy code: it makes serialization semantics explicit before
    any scientific perturbation result is inspected.
    """

    def __init__(self, rows: Iterable[TapeRow]):
        self.rows = tuple(rows)
        steps = [r.step for r in self.rows]
        if steps != list(range(len(steps))):
            raise ValueError(f"tape steps must be contiguous from zero, got {steps[:8]}")

    def __len__(self) -> int:
        return len(self.rows)

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for row in self.rows:
                f.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayTape":
        rows: list[TapeRow] = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(TapeRow(
                step=int(obj["step"]),
                vla=VlaCommand(**{k: tuple(v) for k, v in obj["vla"].items()}),
                actuator=ActuatorReference(**{k: tuple(v) for k, v in obj["actuator"].items()}),
            ))
        return cls(rows)

    def exact_equal(self, other: "ReplayTape") -> bool:
        return self.rows == other.rows
