import importlib.util
import pathlib
import sys
import unittest

import pandas as pd

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "g0_progressive_reversal.py"
SPEC = importlib.util.spec_from_file_location("g0_progressive_reversal", MODULE_PATH)
g0 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = g0
SPEC.loader.exec_module(g0)


class TestHelpers(unittest.TestCase):
    def test_parse_qc_id(self):
        self.assertEqual(g0.parse_qc_id("q123_4"), ("q123", 4))
        self.assertEqual(g0.parse_qc_id("bad"), (None, None))

    def test_normalize_answer(self):
        self.assertEqual(g0.normalize_answer("The, Magna Carta!"), "magna carta")
        self.assertTrue(g0.alias_exact("The Magna Carta", ("magna carta",)))
        self.assertFalse(g0.alias_exact("Magna Cart", ("magna carta",)))

    def test_extract_added_clue(self):
        q = "AAA BBB CCC"
        spans = [[0, 3], [4, 7], [8, 11]]
        self.assertEqual(g0.extract_added_clue(q, spans, 2), "BBB")
        self.assertIsNone(g0.extract_added_clue(q, spans, 4))


class TestCleaningAndTransitions(unittest.TestCase):
    def make_questions(self):
        full = "clue one clue two clue three"
        spans = [[0, 8], [9, 17], [18, 28]]
        rows = []
        for i, text in [
            (1, "clue one"),
            (2, "clue one clue two"),
            (3, "clue one clue two clue three"),
        ]:
            rows.append(
                {
                    "qc_id": f"q1_{i}",
                    "clue_text": text,
                    "n_clues": i,
                    "clean_answers": ["gold answer"],
                    "alias_norms": ("gold answer",),
                    "orig_qid": "q1",
                    "full_quiz_question": full,
                    "clue_spans": spans,
                    "orig_answer_string": "gold answer",
                    "metadata": {"category": "Test", "subcategory": "Unit"},
                }
            )
        return pd.DataFrame(rows)

    def test_human_excluded_and_reversal_detected(self):
        questions = self.make_questions()
        responses = pd.DataFrame(
            [
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_1",
                    "answer": "gold answer",
                    "prediction": "gold answer",
                    "score": 1.0,
                },
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_2",
                    "answer": "gold answer",
                    "prediction": "wrong",
                    "score": 0.0,
                },
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_3",
                    "answer": "gold answer",
                    "prediction": "gold answer",
                    "score": 1.0,
                },
                {
                    "config": "humans",
                    "agent_type": "human_team",
                    "qc_id": "q1_1",
                    "answer": "gold answer",
                    "prediction": "gold answer",
                    "score": 1.0,
                },
            ]
        )
        clean, audit = g0.clean_and_join(responses, questions, include_human=False)
        self.assertEqual(audit["human_rows_seen"], 1)
        self.assertEqual(set(clean["agent_type"]), {"llm"})

        traj, events, counts = g0.analyze_transitions(clean)
        self.assertEqual(counts["official_reversal_events"], 1)
        self.assertEqual(counts["eligible_consecutive_transitions_from_correct"], 1)
        self.assertAlmostEqual(counts["reversal_rate_given_current_correct"], 1.0)
        self.assertEqual(len(events), 1)
        self.assertTrue(bool(events.iloc[0]["strict_alias_stable"]))
        self.assertEqual(events.iloc[0]["new_clue_text"], "clue two")
        self.assertTrue(bool(traj.iloc[0]["recovered_immediately"]))

    def test_conflicting_duplicate_cell_dropped(self):
        questions = self.make_questions()
        responses = pd.DataFrame(
            [
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_1",
                    "answer": "gold answer",
                    "prediction": "gold answer",
                    "score": 1.0,
                },
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_1",
                    "answer": "gold answer",
                    "prediction": "different",
                    "score": 1.0,
                },
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_2",
                    "answer": "gold answer",
                    "prediction": "wrong",
                    "score": 0.0,
                },
            ]
        )
        clean, audit = g0.clean_and_join(responses, questions, include_human=False)
        self.assertEqual(audit["ambiguous_duplicate_cells_dropped"], 1)
        self.assertEqual(clean["clue_idx"].tolist(), [2])

    def test_gap_pair_not_primary(self):
        questions = self.make_questions()
        responses = pd.DataFrame(
            [
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_1",
                    "answer": "gold answer",
                    "prediction": "gold answer",
                    "score": 1.0,
                },
                {
                    "config": "model_a",
                    "agent_type": "llm",
                    "qc_id": "q1_3",
                    "answer": "gold answer",
                    "prediction": "wrong",
                    "score": 0.0,
                },
            ]
        )
        clean, _ = g0.clean_and_join(responses, questions, include_human=False)
        _, events, counts = g0.analyze_transitions(clean)
        self.assertEqual(counts["official_reversal_events"], 0)
        self.assertEqual(counts["gap_pairs_excluded_from_primary"], 1)
        self.assertTrue(events.empty)


class TestGates(unittest.TestCase):
    def test_go_gate(self):
        audit = {
            "question_join_coverage_before_contract_filter": 1.0,
            "clean_joined_rows": 10000,
        }
        counts = {
            "official_reversal_events": 120,
            "eligible_consecutive_transitions_from_correct": 4000,
            "reversal_rate_given_current_correct": 0.03,
        }
        events = pd.DataFrame(
            {
                "qid": [f"q{i % 60}" for i in range(120)],
                "config": [f"m{i % 6}" for i in range(120)],
                "strict_alias_stable": [True] * 40 + [False] * 80,
            }
        )
        verdict, gates = g0.evaluate_gates(
            audit, counts, events, include_human=False, debug_subset=False
        )
        self.assertEqual(verdict, "GO_REVERSAL_OBJECT")
        self.assertTrue(all(x["pass"] for x in gates.values()))

    def test_subset_is_debug_only(self):
        verdict, _ = g0.evaluate_gates(
            {}, {}, pd.DataFrame(), include_human=False, debug_subset=True
        )
        self.assertEqual(verdict, "DEBUG_ONLY_CONFIG_SUBSET")


if __name__ == "__main__":
    unittest.main()
