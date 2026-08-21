import math

from memory_interference.metrics import interference_asymmetry, log_auc, summarize


def test_asymmetry_sign_convention():
    assert interference_asymmetry(0.9, 0.4) == 0.5
    assert interference_asymmetry(0.4, 0.9) == -0.5


def test_log_auc_constant_curve():
    auc = log_auc({1: 1.0, 3: 1.0, 7: 1.0})
    expected = math.log10(8) - math.log10(2)
    assert abs(auc - expected) < 1e-9


def test_summary_pairs_ri_pi():
    rows = []
    for condition, corrects in [("RI", [1, 1]), ("PI", [1, 0])]:
        for i, correct in enumerate(corrects):
            rows.append(
                {
                    "model": "m",
                    "condition": condition,
                    "num_updates": 2,
                    "correct": bool(correct),
                    "target_rank": 1 if correct else 2,
                    "skipped": False,
                    "episode_id": i,
                    "query_key": "x",
                }
            )
    summary = summarize(rows)
    cell = next(x for x in summary if x["num_updates"] == 2)
    assert cell["ri_accuracy"] == 1.0
    assert cell["pi_accuracy"] == 0.5
    assert cell["I"] == 0.5
