import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze import orientation_effect, sign_flip_pvalue, verdict


class AnalysisTest(unittest.TestCase):
    def test_effect_signs(self):
        values = {"CC": 0.95, "IC": 0.70, "CW": 0.90, "IW": 0.65}
        effects = orientation_effect(values)
        self.assertAlmostEqual(effects["delta_consistency"], 0.25)
        self.assertAlmostEqual(effects["delta_correctness"], 0.05)
        self.assertGreater(effects["coherent_wrong_minus_incoherent_correct"], 0)

    def test_sign_flip_detects_strong_positive(self):
        x = np.linspace(0.1, 0.3, 20)
        p = sign_flip_pvalue(x, np.random.default_rng(1), 5000)
        self.assertLess(p, 0.01)

    @staticmethod
    def _metrics(cons_ci=(0.05, 0.15), compare_ci=(0.02, 0.10), full_ci=(0.03, 0.12), tail_ci=(0.01, 0.08)):
        return {
            "confidence_result_late": {
                "effects": {
                    "delta_consistency": {"ci95": list(cons_ci)},
                    "coherent_wrong_minus_incoherent_correct": {"ci95": list(compare_ci)},
                }
            },
            "confidence_tail": {"effects": {"delta_consistency": {"ci95": list(tail_ci)}}},
            "confidence_full": {"effects": {"delta_consistency": {"ci95": list(full_ci)}}},
        }

    def test_protocol_failure_blocks_interpretation(self):
        protocol = {"gap_ci95": [-0.01, 0.03]}
        v, _ = verdict(protocol, self._metrics())
        self.assertEqual(v, "INVALID_PROTOCOL_DO_NOT_INTERPRET")

    def test_strong_signal_requires_late_result_and_cw_ic(self):
        protocol = {"gap_ci95": [0.10, 0.30]}
        v, _ = verdict(protocol, self._metrics())
        self.assertEqual(v, "GO_STRONG_STRUCTURAL_SIGNAL")

    def test_nonpositive_late_result_kills_topic(self):
        protocol = {"gap_ci95": [0.10, 0.30]}
        v, _ = verdict(protocol, self._metrics(cons_ci=(-0.10, -0.01)))
        self.assertEqual(v, "KILL_NO_INTERNAL_CONSISTENCY_SIGNAL")


if __name__ == "__main__":
    unittest.main()
