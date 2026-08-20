import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from fate_labels import fate_from_correctness
from gsm8k_utils import extract_boxed_number, extract_number, extract_number_strict, numerically_equal


def test_answer_extraction():
    assert extract_number("work... #### 1,234") == "1234"
    assert extract_number("no marker 2 then 7") == "7"
    assert extract_number_strict("no marker 2 then 7") is None
    assert extract_boxed_number(r"<answer>\boxed{42}</answer>") == "42"
    assert numerically_equal("3.0", "3")


def test_fates_recovery_and_overwrite():
    c = np.array([
        [0,0,1,1,0,1],
        [0,0,0,0,0,0],
        [1,1,1,1,1,1],
    ], dtype=bool)
    steps = np.array([0,2,4,5])
    f = fate_from_correctness(c, steps)
    assert f["recoverable"][0,0] == 1
    assert f["recovery_lead"][0,0] == 2
    assert f["recoverable"][1,0] == 0
    assert f["will_overwrite"][0,1] == 1
    assert f["overwrite_lead"][0,1] == 2
    assert f["will_overwrite"][2,1] == 0
    assert f["will_overwrite"][0,3] == 0
