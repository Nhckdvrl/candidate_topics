from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g0_position_dissociation import (
    build_target_program,
    make_context_pair,
    normalize_line,
    parse_list,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=text.replace("\n", " \n ").split())


def test_generated_program_executes():
    p = build_target_program(random.Random(7), 3, 8)
    ns = {}
    exec(p.code, ns)
    assert ns["target_3"](p.input_x) == p.output
    assert len(p.trace) == 8


def test_parse_list_rejects_non_integer_or_missing():
    assert parse_list("answer: [1, -2, 3]") == [1, -2, 3]
    assert parse_list("answer: [1.0, 2]") is None
    assert parse_list("no list") is None


def test_normalize_line_requires_assignment():
    assert normalize_line("```python\n arr[2]   =   arr[1] + 4\n```") == "arr[2] = arr[1] + 4"
    assert normalize_line("I cannot find it") == ""


def test_context_pair_is_same_content_with_middle_target():
    tok = FakeTokenizer()
    p = build_target_program(random.Random(11), 4, 8)
    pair = make_context_pair(tok, p.code, random.Random(19), 2500)
    assert pair.start.count(p.code) == 1
    assert pair.middle.count(p.code) == 1
    assert pair.distractor_sha256
    assert pair.start_target_center_fraction < 0.12
    assert 0.40 <= pair.middle_target_center_fraction <= 0.60
    assert abs(pair.start_tokens - pair.middle_tokens) <= 16
