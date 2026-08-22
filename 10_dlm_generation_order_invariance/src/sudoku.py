from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import random

Grid = tuple[int, ...]
ROWS = tuple(range(9))
COLS = tuple(range(9))
DIGITS = frozenset(range(1, 10))


def _validate_grid(grid: Sequence[int], allow_zero: bool = True) -> None:
    if len(grid) != 81:
        raise ValueError(f"expected 81 cells, got {len(grid)}")
    allowed = set(range(10)) if allow_zero else set(range(1, 10))
    if any(int(x) not in allowed for x in grid):
        raise ValueError("grid contains invalid digit")


def rc(index: int) -> tuple[int, int]:
    return divmod(index, 9)


def idx(row: int, col: int) -> int:
    return row * 9 + col


def peers(index: int) -> set[int]:
    r, c = rc(index)
    box_r, box_c = (r // 3) * 3, (c // 3) * 3
    out = {idx(r, cc) for cc in COLS} | {idx(rr, c) for rr in ROWS}
    out |= {idx(rr, cc) for rr in range(box_r, box_r + 3) for cc in range(box_c, box_c + 3)}
    out.discard(index)
    return out


PEERS = tuple(frozenset(peers(i)) for i in range(81))


def candidates(grid: Sequence[int], index: int) -> frozenset[int]:
    _validate_grid(grid)
    if grid[index] != 0:
        return frozenset({int(grid[index])})
    used = {int(grid[j]) for j in PEERS[index] if grid[j]}
    return frozenset(DIGITS - used)


def candidate_counts(grid: Sequence[int]) -> dict[int, int]:
    return {i: len(candidates(grid, i)) for i, v in enumerate(grid) if int(v) == 0}


def is_valid_solution(grid: Sequence[int]) -> bool:
    try:
        _validate_grid(grid, allow_zero=False)
    except ValueError:
        return False
    g = tuple(int(x) for x in grid)
    target = DIGITS
    for r in ROWS:
        if frozenset(g[idx(r, c)] for c in COLS) != target:
            return False
    for c in COLS:
        if frozenset(g[idx(r, c)] for r in ROWS) != target:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if frozenset(g[idx(r, c)] for r in range(br, br + 3) for c in range(bc, bc + 3)) != target:
                return False
    return True


def solve(grid: Sequence[int], limit: int = 2) -> list[Grid]:
    """Backtracking solver; returns at most ``limit`` solutions."""
    _validate_grid(grid)
    state = [int(x) for x in grid]
    solutions: list[Grid] = []

    def rec() -> None:
        if len(solutions) >= limit:
            return
        best_i = -1
        best_cands: frozenset[int] | None = None
        for i, value in enumerate(state):
            if value != 0:
                continue
            cands = candidates(state, i)
            if not cands:
                return
            if best_cands is None or len(cands) < len(best_cands):
                best_i, best_cands = i, cands
                if len(cands) == 1:
                    break
        if best_i < 0:
            solutions.append(tuple(state))
            return
        assert best_cands is not None
        for value in sorted(best_cands):
            state[best_i] = value
            rec()
            state[best_i] = 0
            if len(solutions) >= limit:
                return

    rec()
    return solutions


def unique_solution(grid: Sequence[int]) -> Grid | None:
    sols = solve(grid, limit=2)
    return sols[0] if len(sols) == 1 else None


def canonical_solution() -> Grid:
    return tuple(((r * 3 + r // 3 + c) % 9) + 1 for r in ROWS for c in COLS)


def _shuffled_group_perm(rng: random.Random) -> tuple[int, ...]:
    groups = [0, 1, 2]
    rng.shuffle(groups)
    out: list[int] = []
    for group in groups:
        within = [0, 1, 2]
        rng.shuffle(within)
        out.extend(group * 3 + x for x in within)
    return tuple(out)


@dataclass(frozen=True)
class SudokuTransform:
    """Exact Sudoku automorphism.

    ``row_perm``/``col_perm`` map old coordinates to new coordinates. The
    primary experiment keeps ``digit_perm`` as identity so the only token-level
    change is spatial serialization, not digit identity.
    """

    row_perm: tuple[int, ...]
    col_perm: tuple[int, ...]
    transpose: bool = False
    digit_perm: tuple[int, ...] = tuple(range(10))

    def __post_init__(self) -> None:
        if sorted(self.row_perm) != list(range(9)) or sorted(self.col_perm) != list(range(9)):
            raise ValueError("row/column permutations must be bijections")
        if len(self.digit_perm) != 10 or self.digit_perm[0] != 0 or sorted(self.digit_perm[1:]) != list(range(1, 10)):
            raise ValueError("digit permutation must map 1..9 bijectively and keep 0 fixed")

    def map_index(self, old_index: int) -> int:
        r, c = rc(old_index)
        if self.transpose:
            r, c = c, r
        return idx(self.row_perm[r], self.col_perm[c])

    def apply(self, grid: Sequence[int]) -> Grid:
        _validate_grid(grid)
        out = [0] * 81
        for old_i, value in enumerate(grid):
            out[self.map_index(old_i)] = self.digit_perm[int(value)]
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "row_perm": list(self.row_perm),
            "col_perm": list(self.col_perm),
            "transpose": self.transpose,
            "digit_perm": list(self.digit_perm),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SudokuTransform":
        return cls(tuple(d["row_perm"]), tuple(d["col_perm"]), bool(d.get("transpose", False)), tuple(d.get("digit_perm", range(10))))


def random_spatial_transform(rng: random.Random, force_nonidentity: bool = True) -> SudokuTransform:
    while True:
        t = SudokuTransform(
            row_perm=_shuffled_group_perm(rng),
            col_perm=_shuffled_group_perm(rng),
            transpose=bool(rng.getrandbits(1)),
        )
        if not force_nonidentity or any(t.map_index(i) != i for i in range(81)):
            return t


def random_full_transform(rng: random.Random) -> SudokuTransform:
    base = random_spatial_transform(rng)
    digits = list(range(1, 10))
    rng.shuffle(digits)
    return SudokuTransform(base.row_perm, base.col_perm, base.transpose, tuple([0] + digits))


def random_solution(rng: random.Random) -> Grid:
    return random_full_transform(rng).apply(canonical_solution())


def make_unique_puzzle(rng: random.Random, blanks: int = 45) -> tuple[Grid, Grid]:
    """Generate a deterministic-by-seed unique puzzle by clue removal."""
    if not 1 <= blanks <= 64:
        raise ValueError("blanks must be in [1, 64]")
    solution = random_solution(rng)
    puzzle = list(solution)
    order = list(range(81))
    rng.shuffle(order)
    removed = 0
    for i in order:
        if removed >= blanks:
            break
        old = puzzle[i]
        puzzle[i] = 0
        if unique_solution(puzzle) is None:
            puzzle[i] = old
        else:
            removed += 1
    if removed != blanks:
        raise RuntimeError(f"could only remove {removed}/{blanks} clues while preserving uniqueness")
    return tuple(puzzle), solution


def format_puzzle(grid: Sequence[int]) -> str:
    _validate_grid(grid)
    rows = []
    for r in ROWS:
        rows.append(" ".join(str(grid[idx(r, c)]) if grid[idx(r, c)] else "." for c in COLS))
    return "\n".join(rows)


def blank_indices(grid: Sequence[int]) -> tuple[int, ...]:
    return tuple(i for i, v in enumerate(grid) if int(v) == 0)


def boundary_distance(index: int) -> int:
    return min(index, 80 - index)
