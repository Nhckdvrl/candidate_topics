import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from score_llada import hamming,token_positions_for_char_spans

class ScoringUtilsTest(unittest.TestCase):
    def test_hamming(self):
        self.assertEqual(hamming([1,2,3],[1,9,3]),1); self.assertGreater(hamming([1],[1,2]),100)
    def test_span_mapping(self):
        offsets=[(0,2),(2,3),(4,6),(7,9)]; self.assertEqual(token_positions_for_char_spans(offsets,[(4,6)]),[2])

if __name__=="__main__":unittest.main()
