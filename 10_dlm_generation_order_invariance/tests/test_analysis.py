import pytest

from analyze_g0 import analyze
from sudoku import SudokuTransform


def _rec(pid, variant, steps, exact=True, transform=None, remasking="low_confidence", native=1.0):
    return {
        "puzzle_id": pid,
        "variant_id": variant,
        "split": "discovery",
        "remasking": remasking,
        "exact_solution": exact,
        "finalization_step": {str(k): v for k, v in steps.items()},
        "transform": transform,
        "metadata": {"native_digit_argmax_fraction": native},
    }


def test_analysis_perfect_isomorphism_tau_and_repeat():
    t = SudokuTransform(tuple(range(9)), tuple(range(9)), False).as_dict()
    records = [
        _rec("p", "identity", {0: 1, 1: 2, 2: 3}),
        _rec("p", "identity-repeat", {0: 1, 1: 2, 2: 3}),
        _rec("p", "iso-0", {0: 1, 1: 2, 2: 3}, transform=t),
    ]
    manifest = [{"puzzle_id": "p", "candidate_counts": {"0": 1, "1": 2, "2": 3}}]
    out = analyze(records, manifest, "discovery", {
        "min_identity_exact_puzzles_for_interpretation": 1,
        "min_both_exact_pairs_for_order_analysis": 1,
        "min_both_exact_puzzles_for_order_analysis": 1,
        "same_serialization_tau_floor": 0.95,
    })
    assert out["tau_iso_per_puzzle"]["mean"] == 1.0
    assert out["easy_first_candidate_count_spearman_per_puzzle"]["mean"] == 1.0
    assert out["same_serialization_repeat_tau"]["mean"] == 1.0
    assert out["decision_flags"]["same_serialization_order_stable"]


def test_solve_flip_is_kept_as_non_equivariance_evidence():
    t = SudokuTransform(tuple(range(9)), tuple(range(9)), False).as_dict()
    records = [
        _rec("p", "identity", {0: 1, 1: 2, 2: 3}, exact=True),
        _rec("p", "iso-0", {0: 1, 1: 2, 2: 3}, exact=False, transform=t),
    ]
    manifest = [{"puzzle_id": "p", "candidate_counts": {"0": 1, "1": 2, "2": 3}}]
    out = analyze(records, manifest, "discovery")
    assert out["solve_flip_count"] == 1
    assert out["solve_flip_rate"] == 1.0
    assert out["solve_flip_directions"]["identity_correct_isomorph_wrong"] == 1
    assert out["n_both_exact_isomorph_pairs"] == 0


def test_surface_position_null_is_reported_separately():
    reverse = tuple(reversed(range(9)))
    t = SudokuTransform(reverse, reverse, False)
    # Three mapped blank cells. The model preserves logical order perfectly after
    # mapping, even though absolute row-major positions reverse.
    old = {0: 1, 10: 2, 20: 3}
    new = {t.map_index(i): step for i, step in old.items()}
    records = [
        _rec("p", "identity", old),
        _rec("p", "iso-0", new, transform=t.as_dict()),
    ]
    manifest = [{"puzzle_id": "p", "candidate_counts": {str(i): 1 for i in old}}]
    out = analyze(records, manifest, "discovery")
    assert out["tau_iso_per_puzzle"]["mean"] == 1.0
    assert out["surface_order_positional_null_per_puzzle"]["mean"] == -1.0
    assert out["tau_excess_over_surface_null_per_puzzle"]["mean"] == 2.0


def test_duplicate_trace_keys_fail_loudly():
    rec = _rec("p", "identity", {0: 1, 1: 2})
    with pytest.raises(ValueError, match="duplicate trace key"):
        analyze([rec, dict(rec)], [{"puzzle_id": "p", "candidate_counts": {}}], "discovery")
