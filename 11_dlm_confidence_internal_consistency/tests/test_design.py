import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_design import build_pair, validate_pair


class DesignTest(unittest.TestCase):
    def test_factorial_is_exact_in_text_space(self):
        pair = build_pair(0, random.Random(7), 20, 89)
        validate_pair(pair)
        by_orientation = {}
        for sample in pair:
            by_orientation.setdefault(sample.orientation, {})[sample.cell] = sample
        for cells in by_orientation.values():
            cc, ic, cw, iw = (cells[k] for k in ("CC", "IC", "CW", "IW"))
            self.assertEqual(cc.prompt, ic.prompt)
            self.assertEqual(cw.prompt, iw.prompt)
            self.assertEqual(cc.continuation_text, ic.continuation_text)
            self.assertEqual(cw.continuation_text, iw.continuation_text)
            self.assertEqual(cc.result_char_spans, ic.result_char_spans)
            self.assertEqual(cc.result_char_spans, cw.result_char_spans)
            self.assertEqual(cc.result_char_spans, iw.result_char_spans)
            self.assertNotEqual(cc.announcement_text, ic.announcement_text)
            self.assertNotEqual(cw.announcement_text, iw.announcement_text)
            self.assertEqual(cc.announcement_text + cc.continuation_text, cw.announcement_text + cw.continuation_text)
            self.assertEqual(ic.announcement_text + ic.continuation_text, iw.announcement_text + iw.continuation_text)
            self.assertNotEqual(cc.prompt, cw.prompt)
            self.assertNotEqual(ic.prompt, iw.prompt)

    def test_result_spans_point_only_to_numbers(self):
        pair = build_pair(1, random.Random(13), 20, 89)
        for sample in pair:
            self.assertEqual(len(sample.result_char_spans), 4)
            for start, end in sample.result_char_spans:
                self.assertTrue(sample.continuation_text[start:end].isdigit())

    def test_mirror_reverses_anchor_roles(self):
        pair = build_pair(3, random.Random(11), 20, 89)
        by_orientation = {}
        for sample in pair:
            by_orientation.setdefault(sample.orientation, {})[sample.cell] = sample
        self.assertEqual(by_orientation[0]["CC"].branch_anchor, by_orientation[1]["CC"].alternate_anchor)
        self.assertEqual(by_orientation[0]["CC"].alternate_anchor, by_orientation[1]["CC"].branch_anchor)


if __name__ == "__main__":
    unittest.main()
