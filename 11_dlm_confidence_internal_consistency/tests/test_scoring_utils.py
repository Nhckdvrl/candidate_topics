import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score_llada import exact_length_batches, token_positions_for_char_spans


class ScoringUtilsTest(unittest.TestCase):
    def test_char_spans_map_to_overlapping_tokens(self):
        offsets = [(0, 4), (5, 7), (8, 9), (10, 12)]
        self.assertEqual(token_positions_for_char_spans(offsets, [[5, 7], [10, 12]]), [1, 3])

    def test_exact_length_batches_never_pad(self):
        rows = [
            {"input_ids": [1, 2]},
            {"input_ids": [3, 4, 5]},
            {"input_ids": [6, 7]},
            {"input_ids": [8, 9, 10]},
        ]
        batches = list(exact_length_batches(rows, 2))
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(len({len(x["input_ids"]) for x in batch}), 1)


if __name__ == "__main__":
    unittest.main()
