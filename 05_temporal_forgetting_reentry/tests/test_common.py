import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from common import as_bool, last_boxed_content, explicit_answer_leak_reasons, wilson_interval


class CommonTests(unittest.TestCase):
    def test_false_string_is_false(self):
        self.assertFalse(as_bool("False"))
        self.assertFalse(as_bool("0"))
        self.assertTrue(as_bool("True"))

    def test_balanced_box(self):
        self.assertEqual(last_boxed_content(r"x then \\boxed{\\frac{1}{2}}"), r"\\frac{1}{2}")

    def test_leak_detector_is_not_raw_substring_filter(self):
        self.assertEqual(explicit_answer_leak_reasons("We obtain 12 as an intermediate value.", "12"), [])
        self.assertTrue(explicit_answer_leak_reasons("Therefore the final answer is 12.", "12"))

    def test_wilson_bounds(self):
        lo, hi = wilson_interval(12, 16)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)
        self.assertGreaterEqual(lo, 0)
        self.assertLessEqual(hi, 1)


if __name__ == "__main__":
    unittest.main()
