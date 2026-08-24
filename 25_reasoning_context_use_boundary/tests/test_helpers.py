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

    def test_shared_query_interface_keeps_composed_dependency(self):
        decomp = [
            {"question": "ExampleCo >> founder", "answer": "Ada"},
            {"question": "#1 >> birthplace", "answer": "London"},
        ]
        atomic = g0._format_atomic_query(g0._render_atomic_question(decomp, 1))
        composed = g0._format_composed_query(decomp)
        self.assertTrue(atomic.startswith(g0._QUERY_PREFIX))
        self.assertTrue(composed.startswith(g0._QUERY_PREFIX))
        self.assertIn("Step 1: Ada >> birthplace", atomic)
        self.assertNotIn("#1", atomic)
        self.assertIn("Step 1: ExampleCo >> founder", composed)
        self.assertIn("Step 2: #1 >> birthplace", composed)
        self.assertIn("#1 denotes the answer to Step 1", composed)

    def test_stable_rank_is_deterministic_and_identity_sensitive(self):
        self.assertEqual(g0._stable_rank("x"), g0._stable_rank("x"))
        self.assertNotEqual(g0._stable_rank("x"), g0._stable_rank("y"))

    def test_percentile_interpolates_endpoints(self):
        vals = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(g0._percentile(vals, 0.0), 0.0)
        self.assertEqual(g0._percentile(vals, 1.0), 3.0)
        self.assertAlmostEqual(g0._percentile(vals, 0.5), 1.5)

    def _clean_bank_and_source(self):
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
                    "question": "ExampleCo >> founder",
                    "answer": "Ada",
                    "paragraph_support_idx": 0,
                },
                {
                    "question": "#1 >> birthplace",
                    "answer": "London",
                    "paragraph_support_idx": 1,
                },
            ],
        }
        return bank, source

    def test_eligible_case_requires_exact_support_to_both_bank_golds(self):
        bank, source = self._clean_bank_and_source()
        case, reason = g0._build_eligible_case(bank, source, "id")
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(case)
        self.assertEqual(case["support_gold_indices"], [0, 1])
        self.assertIn("Step 1: Ada >> birthplace", case["atomic_queries"][1])
        self.assertIn("Step 2: #1 >> birthplace", case["composed_query"])
        self.assertEqual(case["composed_answer"], "London")

    def test_eligible_case_rejects_missing_step_dependency(self):
        bank, source = self._clean_bank_and_source()
        source["question_decomposition"][1]["question"] = "Ada >> birthplace"
        case, reason = g0._build_eligible_case(bank, source, "id")
        self.assertIsNone(case)
        self.assertEqual(reason, "step1_has_no_dependency")

    def test_eligible_case_rejects_final_answer_mismatch(self):
        bank, source = self._clean_bank_and_source()
        source["question_decomposition"][1]["answer"] = "Paris"
        case, reason = g0._build_eligible_case(bank, source, "id")
        self.assertIsNone(case)
        self.assertEqual(reason, "final_answer_mismatch")

    def test_eligible_case_rejects_support_not_in_bank_gold(self):
        bank, source = self._clean_bank_and_source()
        source["paragraphs"][0]["paragraph_text"] = "not a bank gold"
        case, reason = g0._build_eligible_case(bank, source, "id")
        self.assertIsNone(case)
        self.assertEqual(reason, "support_not_unique_bank_gold")


if __name__ == "__main__":
    unittest.main()
