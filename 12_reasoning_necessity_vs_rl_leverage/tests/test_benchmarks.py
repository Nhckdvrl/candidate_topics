import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic12.benchmarks import Example, grade_math, make_prompt, stable_subset


def test_qwen_seed_prompt_is_frozen():
    ex = Example("math500", "x", "Compute 2+2.", "4", {})
    prompt = make_prompt(ex)
    assert prompt.startswith("<|im_start|>system\nPlease reason step by step")
    assert "<|im_start|>user\nCompute 2+2.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_stable_subset_is_nested_for_sample_expansion():
    xs = [Example("t", str(i), str(i), str(i), {}) for i in range(20)]
    a = stable_subset(xs, 8, 123)
    b = stable_subset(xs, 12, 123)
    assert [x.uid for x in a] == [x.uid for x in b[:8]]


def test_math500_bare_gold_is_normalized_for_math_verify():
    correct, parse_ok, error = grade_math(r"The answer is \boxed{(-1, 6)}.", "(-1,6)")
    assert correct is True
    assert parse_ok is True
    assert error is None
