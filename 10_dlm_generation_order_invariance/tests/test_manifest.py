from make_manifest import build_manifest
from sudoku import SudokuTransform


def test_manifest_is_deterministic_and_spatial_only():
    a = build_manifest(seed=123, n_discovery=2, n_confirmation=1, blanks=25, transforms_per_puzzle=2)
    b = build_manifest(seed=123, n_discovery=2, n_confirmation=1, blanks=25, transforms_per_puzzle=2)
    assert a == b
    assert [x.split for x in a] == ["discovery", "discovery", "confirmation"]
    for rec in a:
        for td in rec.transforms:
            t = SudokuTransform.from_dict(td)
            assert t.digit_perm == tuple(range(10))
