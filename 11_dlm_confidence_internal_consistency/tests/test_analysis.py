import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze import orientation_effect


class AnalysisTest(unittest.TestCase):
    def test_effect_signs(self):
        values = {"CC": 0.95, "IC": 0.70, "CW": 0.90, "IW": 0.65}
        effects = orientation_effect(values)
        self.assertAlmostEqual(effects["delta_consistency"], 0.25)
        self.assertAlmostEqual(effects["delta_correctness"], 0.05)
        self.assertGreater(effects["coherent_wrong_minus_incoherent_correct"], 0)


if __name__ == "__main__":
    unittest.main()
