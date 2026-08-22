import random

from sudoku import (
    blank_indices,
    candidate_counts,
    canonical_solution,
    is_valid_solution,
    make_unique_puzzle,
    random_solution,
    random_spatial_transform,
    solve,
)


def test_canonical_solution_is_valid():
    assert is_valid_solution(canonical_solution())


def test_random_solution_is_valid_deterministic_and_seed_sensitive():
    a = random_solution(random.Random(123))
    b = random_solution(random.Random(123))
    c = random_solution(random.Random(124))
    assert a == b
    assert a != c
    assert is_valid_solution(a) and is_valid_solution(c)


def test_generated_puzzle_unique():
    puzzle, solution = make_unique_puzzle(random.Random(7), blanks=35)
    assert len(blank_indices(puzzle)) == 35
    assert solve(puzzle, limit=2) == [solution]


def test_spatial_transform_preserves_solution_and_blanks():
    rng = random.Random(11)
    puzzle, solution = make_unique_puzzle(rng, blanks=30)
    t = random_spatial_transform(rng)
    tp, ts = t.apply(puzzle), t.apply(solution)
    assert is_valid_solution(ts)
    assert solve(tp, limit=2) == [ts]
    assert {t.map_index(i) for i in blank_indices(puzzle)} == set(blank_indices(tp))
    assert len({t.map_index(i) for i in range(81)}) == 81


def test_candidate_counts_are_isomorphism_invariant_cellwise():
    rng = random.Random(19)
    puzzle, _ = make_unique_puzzle(rng, blanks=30)
    t = random_spatial_transform(rng)
    tp = t.apply(puzzle)
    a, b = candidate_counts(puzzle), candidate_counts(tp)
    for i, count in a.items():
        assert b[t.map_index(i)] == count
