import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from graph_parser import parse_first_fork
from prompt_utils import full_prompt
from analyze_sampled_branches import first_branch


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


def test_first_branch_parser():
    assert first_branch("Reasoning\n1. q = p + 3 = 8.\n2. r = q + 2") == "q"
    assert first_branch("No numbered branch here") is None
