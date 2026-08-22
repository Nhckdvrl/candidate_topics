from __future__ import annotations

from dataclasses import dataclass
import random


def _rc(i: int) -> tuple[int, int]:
    return divmod(i, 4)


def _idx(r: int, c: int) -> int:
    return r * 4 + c


def _group_perm(rng: random.Random) -> tuple[int, ...]:
    groups = [0, 1]
    rng.shuffle(groups)
    out: list[int] = []
    for group in groups:
        within = [0, 1]
        rng.shuffle(within)
        out.extend(group * 2 + x for x in within)
    return tuple(out)


@dataclass(frozen=True)
class Sudoku4Transform:
    row_perm: tuple[int, ...]
    col_perm: tuple[int, ...]
    transpose: bool = False

    def __post_init__(self) -> None:
        if sorted(self.row_perm) != list(range(4)) or sorted(self.col_perm) != list(range(4)):
            raise ValueError("row/column permutations must be bijections")

    def map_index(self, old_index: int) -> int:
        r, c = _rc(old_index)
        if self.transpose:
            r, c = c, r
        return _idx(self.row_perm[r], self.col_perm[c])

    def apply(self, grid: str) -> str:
        if len(grid) != 16:
            raise ValueError("4x4 grid must contain 16 characters")
        out = ["0"] * 16
        for old_i, value in enumerate(grid):
            out[self.map_index(old_i)] = value
        return "".join(out)

    def as_dict(self) -> dict:
        return {"row_perm": list(self.row_perm), "col_perm": list(self.col_perm), "transpose": self.transpose}

    @classmethod
    def from_dict(cls, d: dict) -> "Sudoku4Transform":
        return cls(tuple(d["row_perm"]), tuple(d["col_perm"]), bool(d.get("transpose", False)))


def random_spatial_transform(rng: random.Random) -> Sudoku4Transform:
    while True:
        t = Sudoku4Transform(_group_perm(rng), _group_perm(rng), bool(rng.getrandbits(1)))
        if any(t.map_index(i) != i for i in range(16)):
            return t


def blank_indices(puzzle: str) -> list[int]:
    return [i for i, c in enumerate(puzzle) if c == "0"]
