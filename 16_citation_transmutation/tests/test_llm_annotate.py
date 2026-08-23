import pytest

from llm_annotate import extract_json_object, map_certainty_to_underlying


def test_extract_json_object_accepts_fenced_noise():
    obj = extract_json_object('```json\n{"stronger":"A"}\n```')
    assert obj["stronger"] == "A"


@pytest.mark.parametrize(
    "stronger,a_is_source,expected",
    [
        ("A", True, "DOWN"),
        ("B", True, "UP"),
        ("A", False, "UP"),
        ("B", False, "DOWN"),
        ("SAME", True, "SAME"),
    ],
)
def test_certainty_mapping(stronger, a_is_source, expected):
    assert map_certainty_to_underlying(stronger, a_is_source) == expected
