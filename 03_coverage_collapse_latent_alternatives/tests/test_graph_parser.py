import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from graph_parser import parse_equations, parse_first_fork, unique_terminal_from
from prepare_forks import choose_unused_control_target


def test_parse_first_fork_and_matched_alternative_target():
    q = (
        "Consider a system of variables where each variable is defined as follows: "
        "p = 3; c = p + 4; d = c + 5; x = p + 8; y = x + 2. "
        "If p = 3, determine the value of d."
    )
    f = parse_first_fork(q)
    assert f.premise == "p"
    assert {f.candidate_a, f.candidate_b} == {"c", "x"}
    assert f.viable == "c"
    assert f.target == "d"
    assert f.alternative_target == "y"


def test_unique_terminal_from_chain():
    rules = {"p": "3", "c": "p + 1", "d": "c + 1", "x": "p + 1", "y": "x + 1"}
    assert unique_terminal_from("c", rules) == "d"
    assert unique_terminal_from("x", rules) == "y"


def test_control_target_is_unused_single_letter():
    q = (
        "Consider a system of variables where each variable is defined as follows: "
        "p = 3; c = p + 4; d = c + 5; x = p + 8; y = x + 2. "
        "If p = 3, determine the value of d."
    )
    control = choose_unused_control_target(q, problem_id=7)
    assert len(control) == 1
    assert control not in parse_equations(q)
