import sys
import unittest
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[1]
if str(TOPIC_DIR) not in sys.path:
    sys.path.insert(0, str(TOPIC_DIR))

import g0_atomic_vs_composed as g0


class Topic25HelperTests(unittest.TestCase):
    def test_placeholder_resolution_uses_only_earlier_gold_answer(self):
        decomp = [
            {"question": "Who founded ExampleCo?", "answer": "Ada"},
            {"question": "Where was #1 born?", "answer": "London"},
        ]
        self.assertEqual(
            g0._render_atomic_question(decomp, 1),
            "Where was Ada born?",
        )

    def test_future_or_self_placeholder_is_rejected(self):
        decomp = [
            {"question": "Where was #1 born?", "answer": "London"},
            {"question": "Who is #1?", "answer": "Ada"},
        ]
        with self.assertRaises(ValueError):
            g0._render_atomic_question(decomp, 0)

    def test_stable_rank_is_deterministic_and_identity_sensitive(self):
        self.assertEqual(g0._stable_rank("x"), g0._stable_rank("x"))
        self.assertNotEqual(g0._stable_rank("x"), g0._stable_rank("y"))

    def test_percentile_interpolates_endpoints(self):
        vals = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(g0._percentile(vals, 0.0), 0.0)
        self.assertEqual(g0._percentile(vals, 1.0), 3.0)
        self.assertAlmostEqual(g0._percentile(vals, 0.5), 1.5)

    def test_eligible_case_requires_exact_support_to_both_bank_golds(self):
        bank = {
            "question_id": "2hop__1_2",
            "question": "Where was the founder born?",
            "answers": ["London"],
            "gold_docs": [
                {"text": "Ada founded ExampleCo."},
                {"text": "Ada was born in London."},
            ],
        }
        source = {
            "id": "2hop__1_2",
            "question": "Where was the founder born?",
            "paragraphs": [
                {"paragraph_text": "Ada founded ExampleCo."},
                {"paragraph_text": "Ada was born in London."},
            ],
            "question_decomposition": [
                {
                    "question": "Who founded ExampleCo?",
                    "answer": "Ada",
                    "paragraph_support_idx": 0,
                },
                {
                    "question": "Where was #1 born?",
                    "answer": "London",
                    "paragraph_support_idx": 1,
                },
            ],
        }
        case, reason = g0._build_eligible_case(bank, source, "id")
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(case)
        self.assertEqual(case["support_gold_indices"], [0, 1])
        self.assertEqual(case["atomic_questions"][1], "Where was Ada born?")

    def test_eligible_case_rejects_support_not_in_bank_gold(self):
        bank = {
            "question_id": "2hop__1_2",
            "question": "Q",
            "answers": ["A"],
            "gold_docs": [{"text": "gold one"}, {"text": "gold two"}],
        }
        source = {
            "id": "2hop__1_2",
            "question": "Q",
            "paragraphs": [
                {"paragraph_text": "not a bank gold"},
                {"paragraph_text": "gold two"},
            ],
            "question_decomposition": [
                {"question": "q1", "answer": "a1", "paragraph_support_idx": 0},
                {"question": "q2", "answer": "a2", "paragraph_support_idx": 1},
            ],
        }
        case, reason = g0._build_eligible_case(bank, source, "id")
        self.assertIsNone(case)
        self.assertEqual(reason, "support_not_unique_bank_gold")


if __name__ == "__main__":
    unittest.main()
