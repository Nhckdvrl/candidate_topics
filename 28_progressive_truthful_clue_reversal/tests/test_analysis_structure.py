import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "analyze_reversal_structure.py"
SPEC = importlib.util.spec_from_file_location("analyze_reversal_structure", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class TestTextFeatures(unittest.TestCase):
    def test_contains_token_sequence(self):
        self.assertTrue(analysis.contains_token_sequence("Near New York City", "New York"))
        self.assertFalse(analysis.contains_token_sequence("Yorkshire", "York"))

    def test_capitalized_introduction(self):
        spans = analysis.introduced_capitalized_spans(
            "Seti built a tomb for Nefertari.", "This ruler fought at Kadesh."
        )
        self.assertEqual(spans, ("nefertari", "seti"))

    def test_structural_features(self):
        feats = analysis.structural_features(
            'In 1969, "Apollo 11" reached the Moon (NASA).',
            "Earlier evidence.",
            ("moon",),
            {"apollo": 2.0, "reached": 1.5, "moon": 2.5, "nasa": 3.0},
        )
        self.assertTrue(feats["has_year"])
        self.assertTrue(feats["has_number"])
        self.assertTrue(feats["has_quote"])
        self.assertTrue(feats["has_parenthetical"])
        self.assertTrue(feats["gold_exact_in_new"])


class TestRecovery(unittest.TestCase):
    def test_first_future_recovery_respects_trajectory(self):
        df = pd.DataFrame(
            {
                "config": ["a", "a", "a", "b", "b"],
                "agent_type": ["ai"] * 5,
                "qid": ["q1", "q1", "q1", "q2", "q2"],
                "clue_idx": [1, 2, 3, 1, 2],
                "correct": [1, 0, 1, 1, 0],
            }
        )
        out = analysis.add_first_future_recovery(df)
        self.assertEqual(out.loc[1, "first_future_correct_clue"], 3)
        self.assertEqual(out.loc[1, "recovery_lag_clues"], 1)
        self.assertTrue(pd.isna(out.loc[4, "first_future_correct_clue"]))

    def test_transition_table_reproduces_adjacent_reversal_and_recovery(self):
        full = "Alpha clue. Beta clue. Gamma clue."
        spans = np.asarray([[0, 12], [12, 23], [23, 35]], dtype=np.int32)
        clean = pd.DataFrame(
            {
                "config": ["m"] * 3,
                "agent_type": ["ai"] * 3,
                "qid": ["q1"] * 3,
                "clue_idx": [1, 2, 3],
                "correct": [1, 0, 1],
                "prediction": ["gold", "beta", "gold"],
                "strict_alias_correct": [True, False, True],
                "clue_text": ["Alpha clue.", "Alpha clue. Beta clue.", full],
                "full_quiz_question": [full] * 3,
                "clue_spans": [spans] * 3,
                "alias_norms": [("gold",)] * 3,
                "category": ["Test"] * 3,
            }
        )
        out = analysis.build_transition_table(clean, {"clue": 1.0, "beta": 2.0})
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["reversal"]), 1)
        self.assertTrue(bool(out.iloc[0]["immediate_recovery"]))
        self.assertTrue(bool(out.iloc[0]["eventual_recovery"]))
        self.assertEqual(out.iloc[0]["new_clue_text"], "Beta clue.")


class TestSummaries(unittest.TestCase):
    def test_cluster_bootstrap_binary_difference(self):
        df = pd.DataFrame(
            {
                "qid": ["q1", "q1", "q2", "q2", "q3", "q3"],
                "introduced_name_any": [False, True] * 3,
                "reversal": [0, 1, 0, 1, 0, 0],
            }
        )
        observed, lo, hi = analysis.cluster_bootstrap_binary_difference(
            df, "introduced_name_any", reps=100, seed=1
        )
        self.assertAlmostEqual(observed, 2 / 3)
        self.assertLessEqual(lo, observed)
        self.assertGreaterEqual(hi, observed)


if __name__ == "__main__":
    unittest.main()
