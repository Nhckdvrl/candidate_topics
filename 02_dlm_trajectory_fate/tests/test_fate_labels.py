import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from fate_labels import fate_from_correctness


def test_transient_controls_and_final_outcome_controls():
    c = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    steps = np.array([0, 2, 5])
    f = fate_from_correctness(c, steps)

    assert f["recoverable"][0, 0] == 1
    assert f["transient_recovery"][0, 0] == 1
    assert f["finish_correct_from_wrong"][0, 0] == 0
    assert f["recoverable"][1, 0] == 0
    assert f["transient_recovery"][1, 0] == 0
    assert f["will_overwrite"][2, 0] == 1
    assert f["transient_overwrite"][2, 0] == 1
    assert f["finish_wrong_from_correct"][2, 0] == 0
    assert f["will_overwrite"][3, 0] == 0
    assert f["transient_overwrite"][3, 0] == 0
    assert f["finish_correct_from_wrong"][4, 0] == 1
    assert f["transient_recovery"][4, 0] == -1
    assert f["finish_wrong_from_correct"][5, 0] == 1
    assert f["transient_overwrite"][5, 0] == -1


def test_unobserved_is_not_incorrect():
    c = np.array([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=bool)
    observed = np.array(
        [
            [1, 0, 1, 1],
            [1, 0, 1, 1],
        ],
        dtype=bool,
    )
    f = fate_from_correctness(c, np.array([0, 1, 2]), observed)
    assert f["will_overwrite"][0, 0] == 0
    assert f["current_correct"][0, 1] == -1
    assert f["recoverable"][1, 0] == 1
    assert f["recovery_lead"][1, 0] == 2


def test_leads_and_commitment():
    c = np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]], dtype=bool)
    f = fate_from_correctness(c, np.array([0]))
    assert f["recovery_lead"][0, 0] == 1
    assert f["final_commitment_lead"][0, 0] == 3
    assert f["overwrite_lead"][1, 0] == 1
    assert f["final_commitment_lead"][1, 0] == 3
