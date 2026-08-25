import unittest

import pandas as pd

import g1_order_swap as g1


class TestPromptAndPairing(unittest.TestCase):
    def test_prompt_numbers_clues_without_changing_text(self):
        prompt = g1.build_user_prompt(["First clue.", "Second clue."])
        self.assertIn("1. First clue.", prompt)
        self.assertIn("2. Second clue.", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_pair_outputs_defines_reversals_and_final_harm(self):
        panel = pd.DataFrame(
            {
                "boundary_id": ["q1_2_3"],
                "qid": ["q1"],
                "aliases": [("gold",)],
            }
        )
        states = pd.DataFrame(
            {
                "boundary_id": ["q1_2_3"] * 4,
                "state": ["o1", "o2", "s1", "s2"],
                "prediction": ["gold", "wrong", "gold", "gold"],
                "prediction_norm": ["gold", "wrong", "gold", "gold"],
                "correct": [True, False, True, True],
                "valid": [True] * 4,
            }
        )
        out = g1.pair_outputs(panel, states).iloc[0]
        self.assertTrue(out.original_reversal)
        self.assertFalse(out.swap_reversal)
        self.assertTrue(out.original_only_final_harm)
        self.assertEqual(out.four_state_pattern, "1011")


class TestStatistics(unittest.TestCase):
    def test_cluster_bootstrap_paired_uses_left_minus_right(self):
        df = pd.DataFrame(
            {
                "qid": ["q1", "q1", "q2", "q3"],
                "left": [1, 1, 0, 1],
                "right": [0, 1, 0, 0],
            }
        )
        observed, lo, hi = g1.cluster_bootstrap_paired(df, "left", "right", reps=200)
        self.assertAlmostEqual(observed, 0.5)
        self.assertLessEqual(lo, observed)
        self.assertGreaterEqual(hi, observed)

    def test_debug_run_never_gets_scientific_verdict(self):
        paired = pd.DataFrame(
            {
                "qid": [f"q{i}" for i in range(4)],
                "all_valid": [True] * 4,
                "o1_correct": [True] * 4,
                "o2_correct": [False] * 4,
                "s1_correct": [True] * 4,
                "s2_correct": [True] * 4,
                "common_belief": [True] * 4,
                "original_reversal": [True] * 4,
                "swap_reversal": [False] * 4,
                "original_final_error": [True] * 4,
                "swap_final_error": [False] * 4,
                "original_only_final_harm": [True] * 4,
                "swap_only_final_harm": [False] * 4,
                "order_independent_final_error": [False] * 4,
            }
        )
        states = pd.DataFrame({"single_line": [True] * 16})
        self.assertEqual(g1.summarize(paired, states, debug=True)["verdict"], "DEBUG_NO_VERDICT")


if __name__ == "__main__":
    unittest.main()
