import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from graph_parser import parse_first_fork
from prompt_utils import full_prompt


def test_parse_official_style_two_chain_question():
    q = (
        "Consider a system of variables where each variable is defined as follows: "
        "p = 3; c = p + 4; d = c + 5; x = p + 8; y = x + 2. "
        "If p = 3, determine the value of d."
    )
    f = parse_first_fork(q)
    assert f.premise == "p"
    assert f.target == "d"
    assert {f.candidate_a, f.candidate_b} == {"c", "x"}
    assert f.viable == "c"


def test_prompt_matches_official_template_shape():
    p = full_prompt("Q?")
    assert "### Instruction:" in p
    assert "\\boxed{}" in p
    assert "### Input:\nQ?" in p
    assert p.endswith("step by step:\n1.")
