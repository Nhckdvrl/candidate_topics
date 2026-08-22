import random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from build_design import build_pair,validate_pair,_standalone_integer_occurs

class DesignV3Test(unittest.TestCase):
    def test_exact_future_factorial(self):
        pair=build_pair(0,random.Random(7),20,89); validate_pair(pair); g={}
        for s in pair: g.setdefault(s.orientation,{})[s.cell]=s
        for cells in g.values():
            cc,ic,cw,iw=(cells[k] for k in ("CC","IC","CW","IW"))
            self.assertEqual(len({x.trajectory_text for x in cells.values()}),1)
            self.assertEqual(cc.prompt,ic.prompt); self.assertEqual(cw.prompt,iw.prompt)
            self.assertEqual(cc.check_text,cw.check_text); self.assertEqual(ic.check_text,iw.check_text)
            self.assertNotEqual(cc.prompt,cw.prompt); self.assertNotEqual(cc.check_text,ic.check_text)
            self.assertTrue(cc.trajectory_text.endswith("\n"))
            self.assertFalse(_standalone_integer_occurs(cc.prompt,cc.prompt_anchor))
            self.assertFalse(_standalone_integer_occurs(cc.check_text,cc.check_anchor))
    def test_mirror_reverses_branch(self):
        pair=build_pair(1,random.Random(11),20,89); g={}
        for s in pair:g.setdefault(s.orientation,{})[s.cell]=s
        self.assertEqual(g[0]["CC"].branch_anchor,g[1]["CC"].alternate_anchor)
        self.assertEqual(g[0]["CC"].alternate_anchor,g[1]["CC"].branch_anchor)
    def test_result_spans_are_four_numbers(self):
        pair=build_pair(2,random.Random(19),20,89)
        for s in pair:
            self.assertEqual(len(s.result_char_spans),4)
            for a,b in s.result_char_spans:self.assertTrue(s.trajectory_text[a:b].isdigit())

if __name__=="__main__":unittest.main()
