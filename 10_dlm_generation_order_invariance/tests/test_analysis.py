from analyze_g0 import analyze
from sudoku import SudokuTransform


def _rec(pid, variant, steps, exact=True, transform=None, remasking="low_confidence"):
    return {
        "puzzle_id": pid,
        "variant_id": variant,
        "split": "discovery",
        "remasking": remasking,
        "exact_solution": exact,
        "finalization_step": {str(k): v for k, v in steps.items()},
        "transform": transform,
    }


def test_analysis_perfect_isomorphism_tau():
    t = SudokuTransform(tuple(range(9)), tuple(range(9)), False).as_dict()
    records = [
        _rec("p", "identity", {0: 1, 1: 2, 2: 3}),
        _rec("p", "iso-0", {0: 1, 1: 2, 2: 3}, transform=t),
    ]
    manifest = [{"puzzle_id": "p", "candidate_counts": {"0": 1, "1": 2, "2": 3}}]
    out = analyze(records, manifest, "discovery")
    assert out["tau_iso"]["mean"] == 1.0
    assert out["seed_replication_candidate_count_vs_finalization_spearman"] == 1.0
