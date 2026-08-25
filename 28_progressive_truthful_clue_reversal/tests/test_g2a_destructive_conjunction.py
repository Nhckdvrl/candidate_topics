import unittest

import pandas as pd

import g2a_destructive_conjunction as g2a


class TestClearWrong(unittest.TestCase):
    def test_alias_related_rejects_shortening_and_plural(self):
        self.assertTrue(g2a.prediction_alias_related("Mozart", ("wolfgang amadeus mozart",)))
        self.assertTrue(g2a.prediction_alias_related("Tornado", ("tornadoes",)))
        self.assertTrue(g2a.prediction_alias_related("iPhone 5", ("iphone",)))

    def test_distinct_competitor_is_clear(self):
        self.assertFalse(g2a.prediction_alias_related("Barry Goldwater", ("joe biden",)))


class TestCellsAndStatistics(unittest.TestCase):
    def test_combine_results_marks_destructive_cell(self):
        paired = pd.DataFrame(
            {
                "boundary_id": ["q1_2_3"],
                "qid": ["q1"],
                "aliases": [("gold",)],
                "o1_correct": [True],
                "o2_correct": [False],
                "o1_prediction_norm": ["gold"],
                "o2_prediction": ["competitor"],
                "o2_prediction_norm": ["competitor"],
            }
        )
        c = pd.DataFrame(
            {
                "boundary_id": ["q1_2_3"],
                "prediction": ["gold"],
                "prediction_norm": ["gold"],
                "correct": [True],
                "valid": [True],
                "input_tokens": [10],
                "output_tokens": [1],
                "single_line": [True],
            }
        )
        row = g2a.combine_results(paired, c).iloc[0]
        self.assertTrue(row.jointly_sufficient)
        self.assertTrue(row.destructive_exact)
        self.assertTrue(row.destructive_clear)
        self.assertEqual(row.three_state_pattern, "110")

    def test_cluster_bootstrap_conditional_rate(self):
        df = pd.DataFrame(
            {
                "qid": ["q1", "q1", "q2", "q3"],
                "numer": [1, 0, 1, 0],
                "denom": [1, 1, 1, 0],
            }
        )
        observed, lo, hi = g2a.cluster_bootstrap_conditional_rate(
            df, "numer", "denom", reps=200
        )
        self.assertAlmostEqual(observed, 2 / 3)
        self.assertLessEqual(lo, observed)
        self.assertGreaterEqual(hi, observed)


if __name__ == "__main__":
    unittest.main()
