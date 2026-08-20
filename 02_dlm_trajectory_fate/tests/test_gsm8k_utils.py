import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from gsm8k_utils import (
    extract_boxed_number,
    extract_number,
    extract_number_strict,
    numerically_equal,
)


def test_answer_extraction():
    assert extract_number("work... #### 1,234") == "1234"
    assert extract_number("no marker 2 then 7") == "7"
    assert extract_number_strict("no marker 2 then 7") is None
    assert extract_boxed_number(r"<answer>\boxed{42}</answer>") == "42"
    assert extract_boxed_number(r"\boxed{1}\n\boxed{-3.5}") == "-3.5"
    assert numerically_equal("3.0", "3")
