from src.analyze import diversity_amplification, pooled_transfer, extract_test_step1


def row(condition, diversity, pair_id, active):
    return {
        "phase": "test", "trial": 1, "condition": condition, "diversity": diversity,
        "pair_id": pair_id, "active": active, "valid_action": True,
    }


def test_primary_metric_detects_diversity_amplification():
    rows = []
    # Concentrated: C-U = 0.25. Distributed: C-U = 0.75. Amplification = 0.5.
    for i in range(4):
        rows += [row("controllable", "concentrated", i, 1), row("uncontrollable", "concentrated", i, i != 0)]
        rows += [row("controllable", "distributed", i, 1), row("uncontrollable", "distributed", i, i == 0)]
    cells = extract_test_step1(rows)
    assert abs(pooled_transfer(cells) - 0.5) < 1e-9
    assert abs(diversity_amplification(cells) - 0.5) < 1e-9
